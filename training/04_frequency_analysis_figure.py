"""
Genera la figura comparativa de espectro de frecuencia promedio (real vs.
falsa). Es la evidencia visual central del paper: muestra el patron
periodico que deja el proceso de upsampling del generador, invisible al
ojo humano en la imagen original pero claro en el dominio de frecuencia.

Uso:
    python training/04_frequency_analysis_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = ROOT / "data" / "real"
FAKE_DIR = ROOT / "data" / "fake"
FIG_DIR = ROOT / "paper" / "figures"
IMG_SIZE = 128


def avg_log_spectrum(paths: list[Path]) -> np.ndarray:
    acc = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float64)
    n = 0
    for p in paths:
        img = Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        f = np.fft.fftshift(np.fft.fft2(arr))
        acc += np.log1p(np.abs(f) ** 2)
        n += 1
    return acc / max(n, 1)


def main() -> None:
    real_paths = sorted(REAL_DIR.glob("*.png")) + sorted(REAL_DIR.glob("*.jpg"))
    fake_paths = sorted(FAKE_DIR.glob("*.png"))
    if not real_paths or not fake_paths:
        raise RuntimeError("Faltan imagenes reales o falsas, corre los scripts 01 y 02 primero.")

    real_spec = avg_log_spectrum(real_paths)
    fake_spec = avg_log_spectrum(fake_paths)
    diff = fake_spec - real_spec

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(
        axes, [real_spec, fake_spec, diff], ["Real (promedio)", "Falsa (promedio)", "Diferencia"]
    ):
        im = ax.imshow(data, cmap="inferno")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    out_path = FIG_DIR / "frequency_spectrum_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Figura guardada en {out_path}")


if __name__ == "__main__":
    main()
