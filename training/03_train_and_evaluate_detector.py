"""
Entrena el detector real-vs-falsa (fraude en imagen veterinaria) y produce
las figuras/metricas que el paper reporta.

Arquitectura: dos ramas que se concatenan antes de la capa final, siguiendo
la misma idea de MedForensics/DSKI (MICCAI 2025) simplificada a dos senales
en vez de tres:
  1. Rama espacial: ResNet18 preentrenada en ImageNet (features visuales).
  2. Rama de frecuencia: espectro de potencia radialmente promediado (FFT),
     que es donde quedan los artefactos periodicos del upsampling del GAN.

Uso:
    python training/03_train_and_evaluate_detector.py --epochs 25
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = ROOT / "data" / "real"
FAKE_DIR = ROOT / "data" / "fake"
CKPT_DIR = ROOT / "training" / "checkpoints"
FIG_DIR = ROOT / "paper" / "figures"
IMG_SIZE = 128
FREQ_BINS = 32


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def radial_power_spectrum(img_gray: np.ndarray, n_bins: int = FREQ_BINS) -> np.ndarray:
    """Espectro de potencia FFT promediado radialmente (detecta patrones
    periodicos tipo 'checkerboard' que deja el upsampling de un GAN)."""
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


class RadiographDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], train: bool):
        self.paths = paths
        self.labels = labels
        aug = (
            [transforms.RandomHorizontalFlip(p=0.5)]
            if train
            else []
        )
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                *aug,
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("L")
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        freq = radial_power_spectrum(np.array(img_resized, dtype=np.float32) / 255.0)
        tensor = self.transform(img)
        tensor3 = tensor.repeat(3, 1, 1)  # ResNet espera 3 canales
        tensor3 = (tensor3 - 0.5) / 0.5
        return tensor3, torch.tensor(freq, dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.float32
        )


class DualStreamDetector(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.spatial = backbone
        self.freq_branch = nn.Sequential(
            nn.Linear(FREQ_BINS, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, img: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        feat_img = self.spatial(img)
        feat_freq = self.freq_branch(freq)
        combined = torch.cat([feat_img, feat_freq], dim=1)
        return self.head(combined).squeeze(1)


def load_dataset_splits():
    real_paths = sorted(REAL_DIR.glob("*.png")) + sorted(REAL_DIR.glob("*.jpg"))
    fake_paths = sorted(FAKE_DIR.glob("*.png"))
    if not real_paths or not fake_paths:
        raise RuntimeError(
            "Faltan imagenes. Corre primero 01_prepare_real_data.py y "
            "02_train_generator_and_make_fakes.py"
        )

    paths = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths)  # 0=real, 1=fake

    train_p, test_p, train_l, test_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    train_p, val_p, train_l, val_l = train_test_split(
        train_p, train_l, test_size=0.2, stratify=train_l, random_state=42
    )
    return (train_p, train_l), (val_p, val_l), (test_p, test_l)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, freqs, labels in loader:
            imgs, freqs, labels = imgs.to(device), freqs.to(device), labels.to(device)
            logits = model(imgs, freqs)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    return avg_loss, auc, all_probs, all_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = get_device()
    print(f"Usando device: {device}")

    (train_p, train_l), (val_p, val_l), (test_p, test_l) = load_dataset_splits()
    print(f"train={len(train_p)} val={len(val_p)} test={len(test_p)}")

    train_ds = RadiographDataset(train_p, train_l, train=True)
    val_ds = RadiographDataset(val_p, val_l, train=False)
    test_ds = RadiographDataset(test_p, test_l, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = DualStreamDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = -1.0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_auc, _, _ = run_epoch(model, val_loader, criterion, optimizer, device, False)
        print(
            f"[epoch {epoch}/{args.epochs}] train_loss={train_loss:.3f} train_auc={train_auc:.3f} "
            f"val_loss={val_loss:.3f} val_auc={val_auc:.3f}"
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), CKPT_DIR / "detector_best.pth")

    model.load_state_dict(torch.load(CKPT_DIR / "detector_best.pth", map_location=device))
    test_loss, test_auc, test_probs, test_labels = run_epoch(
        model, test_loader, criterion, optimizer, device, False
    )
    test_preds = [1 if p >= 0.5 else 0 for p in test_probs]

    metrics = {
        "test_loss": test_loss,
        "test_auc": test_auc,
        "accuracy": accuracy_score(test_labels, test_preds),
        "precision": precision_score(test_labels, test_preds, zero_division=0),
        "recall": recall_score(test_labels, test_preds, zero_division=0),
        "f1": f1_score(test_labels, test_preds, zero_division=0),
        "n_train": len(train_p),
        "n_val": len(val_p),
        "n_test": len(test_p),
    }
    print("\n=== METRICAS FINALES (test) ===")
    print(json.dumps(metrics, indent=2))

    (ROOT / "training").mkdir(exist_ok=True)
    with open(ROOT / "training" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    make_figures(test_labels, test_probs, test_preds)


def make_figures(labels, probs, preds) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(labels, probs)
    auc_val = roc_auc_score(labels, probs)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curva ROC — detector real vs. falsa")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=150)
    plt.close()

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.xticks([0, 1], ["Real", "Falsa"])
    plt.yticks([0, 1], ["Real", "Falsa"])
    plt.xlabel("Prediccion")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusion")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Figuras guardadas en {FIG_DIR}")


if __name__ == "__main__":
    main()
