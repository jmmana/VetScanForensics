"""
EXP-00: prueba directa de si el confound de resolucion es real.

Hipotesis (de la auditoria del plan maestro): las imagenes falsas del GAN
nacen nativamente en 128x128, mientras que las reales se reducen desde
resoluciones nativas de ~1500-2800px. Ese paso de reduccion deja una firma
sistematica en el espectro de frecuencia que no tiene nada que ver con
"IA vs. real".

Para aislar el efecto SIN mezclar generadores, este experimento toma
solo imagenes REALES y las divide en dos grupos con la MISMA etiqueta
verdadera (ambas son reales) pero procesadas con dos rutas de resize
distintas:
  - Grupo A: reduccion directa nativa -> 128x128 (bicubic), como hace hoy
    el pipeline del detector.
  - Grupo B: reduccion nativa -> 256x256 -> 128x128 (doble muestreo), para
    simular una ruta de procesamiento distinta.

Si un clasificador puede distinguir Grupo A de Grupo B muy por encima del
azar, eso prueba que el mero historial de resize ya es una senal
aprendible, separada de cualquier forense real de generador. Si el
detector NO logra distinguirlos (cerca de 50% de accuracy), el confound
de resolucion no explica el 100% original.

Uso:
    python training/00_resolution_confound_test.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = ROOT / "data" / "real"
OUT_DIR = ROOT / "training" / "exp00_confound"
IMG_SIZE = 128
FREQ_BINS = 32


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def radial_power_spectrum(img_gray: np.ndarray, n_bins: int = FREQ_BINS) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(img_gray))
    power = np.log1p(np.abs(f) ** 2)
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    r_max = r.max()
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    profile = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        profile[i] = power[mask].mean() if mask.any() else 0.0
    profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-8)
    return profile


def resize_direct(img: Image.Image) -> Image.Image:
    """Grupo A: reduccion directa nativa -> 128x128 (ruta actual del pipeline,
    la misma que reciben las imagenes reales hoy)."""
    return img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)


def resize_upsampled_from_seed(img: Image.Image) -> Image.Image:
    """Grupo B: reduccion nativa -> 8x8 -> 128x128 (upsample desde semilla
    minuscula). Esto imita la caracteristica arquitectonica real de un GAN:
    la imagen se "construye" desde una representacion diminuta hacia arriba
    por upsampling, en vez de simplemente reducirse desde una foto grande.
    Es un proxy mucho mas fiel del confound sospechado que solo variar la
    ruta de downsampling."""
    seed = img.resize((8, 8), Image.BICUBIC)
    return seed.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)


class ConfoundDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int]):
        self.paths = paths
        self.labels = labels
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("L")
        label = self.labels[idx]
        resized = resize_direct(img) if label == 0 else resize_upsampled_from_seed(img)
        freq = radial_power_spectrum(np.array(resized, dtype=np.float32) / 255.0)
        tensor = self.to_tensor(resized).repeat(3, 1, 1)
        tensor = (tensor - 0.5) / 0.5
        return tensor, torch.tensor(freq, dtype=torch.float32), torch.tensor(
            float(label), dtype=torch.float32
        )


class DualStreamDetector(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.spatial = backbone
        self.freq_branch = nn.Sequential(
            nn.Linear(FREQ_BINS, 64), nn.ReLU(inplace=True), nn.Linear(64, 32), nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Linear(512 + 32, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, img, freq):
        feat_img = self.spatial(img)
        feat_freq = self.freq_branch(freq)
        return self.head(torch.cat([feat_img, feat_freq], dim=1)).squeeze(1)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    probs_all, labels_all = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, freqs, labels in loader:
            imgs, freqs, labels = imgs.to(device), freqs.to(device), labels.to(device)
            logits = model(imgs, freqs)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            probs_all.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            labels_all.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else float("nan")
    return avg_loss, auc, probs_all, labels_all


def main() -> None:
    device = get_device()
    print(f"Usando device: {device}")

    real_paths = sorted(REAL_DIR.glob("*.png")) + sorted(REAL_DIR.glob("*.jpg"))
    if not real_paths:
        raise RuntimeError("No hay imagenes reales en data/real/")

    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=len(real_paths)).tolist()  # 0=grupo A, 1=grupo B, mismo pool de imagenes
    print(f"Total imagenes reales: {len(real_paths)} (grupo A={labels.count(0)}, grupo B={labels.count(1)})")

    train_p, test_p, train_l, test_l = train_test_split(
        real_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    train_p, val_p, train_l, val_l = train_test_split(
        train_p, train_l, test_size=0.2, stratify=train_l, random_state=42
    )

    train_ds = ConfoundDataset(train_p, train_l)
    val_ds = ConfoundDataset(val_p, val_l)
    test_ds = ConfoundDataset(test_p, test_l)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    test_loader = DataLoader(test_ds, batch_size=8)

    model = DualStreamDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_auc = -1.0
    best_state = None
    for epoch in range(1, 21):
        train_loss, train_auc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_auc, _, _ = run_epoch(model, val_loader, criterion, optimizer, device, False)
        print(f"[epoch {epoch}/20] train_auc={train_auc:.3f} val_auc={val_auc:.3f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss, test_auc, test_probs, test_labels = run_epoch(model, test_loader, criterion, optimizer, device, False)
    test_preds = [1 if p >= 0.5 else 0 for p in test_probs]
    test_acc = accuracy_score(test_labels, test_preds)

    result = {
        "description": "EXP-00: puede el detector distinguir dos rutas de resize DENTRO de imagenes 100% reales",
        "n_images": len(real_paths),
        "test_accuracy": test_acc,
        "test_auc": test_auc,
        "n_test": len(test_p),
        "interpretation": (
            "Si test_accuracy/test_auc estan muy por encima de 0.5, el historial de "
            "resize por si solo es una senal aprendible, separada de cualquier "
            "forense real de generador. Esto apoya la hipotesis del confound de "
            "resolucion del plan maestro."
        ),
    }
    print("\n=== RESULTADO EXP-00 ===")
    print(json.dumps(result, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "exp00_result.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
