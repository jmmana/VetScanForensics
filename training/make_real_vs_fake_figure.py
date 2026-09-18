"""
Genera la figura de ejemplo real-vs-falsa que todo paper de deteccion de
imagenes sinteticas necesita: unas cuantas radiografias reales al lado de
unas cuantas generadas por el DCGAN, para que el lector vea de inmediato
de que estamos hablando.

Paleta validada (colorblind-safe) del skill de dataviz: azul para reales,
naranja para falsas, usados consistentemente en todas las figuras del paper.

Uso:
    python training/make_real_vs_fake_figure.py
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

REAL_COLOR = "#2a78d6"   # slot 1 del palette validado
FAKE_COLOR = "#eb6834"   # slot 2 del palette validado
INK = "#0b0b0b"
N_EXAMPLES = 4


def load_square(path: Path, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.BICUBIC)
    return np.array(img)


def main() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.edgecolor"] = INK
    plt.rcParams["text.color"] = INK

    real_paths = sorted(REAL_DIR.glob("*.png"))[:N_EXAMPLES]
    fake_paths = sorted(FAKE_DIR.glob("*.png"))[:N_EXAMPLES]
    if not real_paths or not fake_paths:
        raise RuntimeError("Faltan imagenes en data/real/ o data/fake/")

    fig, axes = plt.subplots(2, N_EXAMPLES, figsize=(2.2 * N_EXAMPLES, 5.0))

    for col, path in enumerate(real_paths):
        ax = axes[0, col]
        ax.imshow(load_square(path), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(REAL_COLOR)
            spine.set_linewidth(3)

    for col, path in enumerate(fake_paths):
        ax = axes[1, col]
        ax.imshow(load_square(path), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(FAKE_COLOR)
            spine.set_linewidth(3)

    axes[0, 0].set_ylabel("Real", fontsize=13, fontweight="bold", color=REAL_COLOR, labelpad=10)
    axes[1, 0].set_ylabel("AI-generated (DCGAN)", fontsize=13, fontweight="bold", color=FAKE_COLOR, labelpad=10)

    fig.suptitle(
        "Real canine thoracic radiographs vs. AI-generated counterparts",
        fontsize=13, color=INK, y=1.02,
    )
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "real_vs_fake_examples.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Figura guardada en {out_path}")


if __name__ == "__main__":
    main()
