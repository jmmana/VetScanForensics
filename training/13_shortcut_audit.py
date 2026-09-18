"""EXP-13: control pareado del shortcut de resize usando el detector de EXP-00.

Primero se construyen las rutas A/B de EXP-00; despues se aplica la
correccion a ambas. Asi se mide si elimina una huella preexistente.
Mismos grupos, splits, inicializacion e hiperparametros en ambos brazos.
"""

import importlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from preprocessing_common import random_resize_round_trip

exp00 = importlib.import_module('00_resolution_confound_test')
# Reutilizar exactamente las implementaciones del control original.
DualStreamDetector = exp00.DualStreamDetector
radial_power_spectrum = exp00.radial_power_spectrum
run_epoch = exp00.run_epoch
get_device = exp00.get_device
REAL_DIR = exp00.REAL_DIR
OUT_DIR = Path(__file__).resolve().parent / 'exp13_shortcut_audit'


class AuditDataset(exp00.ConfoundDataset):
    def __init__(self, paths, labels, apply_fix, train):
        super().__init__(paths, labels)
        self.apply_fix = apply_fix
        self.train = train

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as source:
            img = source.convert('L')
        label = self.labels[idx]
        img = exp00.resize_direct(img) if label == 0 else exp00.resize_upsampled_from_seed(img)
        if self.apply_fix:
            seed = int(torch.randint(0, 2**32, ()).item()) if self.train else 42 + idx
            img = random_resize_round_trip(img, np.random.default_rng(seed))
        freq = radial_power_spectrum(np.array(img, dtype=np.float32) / 255.0)
        tensor = (self.to_tensor(img).repeat(3, 1, 1) - 0.5) / 0.5
        return tensor, torch.tensor(freq, dtype=torch.float32), torch.tensor(float(label))


def run_experiment(apply_fix: bool):
    torch.manual_seed(42)
    np.random.seed(42)
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

    train_ds = AuditDataset(train_p, train_l, apply_fix, train=True)
    val_ds = AuditDataset(val_p, val_l, apply_fix, train=False)
    test_ds = AuditDataset(test_p, test_l, apply_fix, train=False)

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

    print(f'fix={apply_fix}: test_auc={test_auc:.6f}, test_accuracy={test_acc:.6f}')
    return float(test_auc)


def main():
    without = run_experiment(False)
    with_fix = run_experiment(True)
    interpretation = (
        f'AUC sin correccion={without:.6f}; con correccion={with_fix:.6f}. '
        + ('Persiste una senal aprendible del historial de resize; no se demuestra '
           'neutralizacion del shortcut. El AUC del detector nuevo no basta para '
           'descartar este confound.' if abs(with_fix - 0.5) > 0.15 else
           'El resultado se acerca al azar en este test pequeno; no prueba por si '
           'solo ausencia de todo confound.')
        + ' Se aplico el rango prescrito 16..128 despues de construir las rutas A/B, '
          'antes de FFT y tensor, sin ajustar hiperparametros segun el test.'
    )
    result = dict(auc_without_fix=without, auc_with_fix=with_fix, interpretation=interpretation)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'exp13_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
