"""
Diagrama de arquitectura de VetRad-ForensicsNet: tres ramas (espacial,
frecuencia, residual de ruido) fusionadas por atencion antes de la cabeza
de clasificacion. Es un esquema conceptual, no requiere resultados
experimentales.

Uso:
    python training/make_architecture_diagram.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"

# Paleta validada (skill de dataviz)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
INK = "#0b0b0b"
GRAY = "#898781"


def box(ax, xy, w, h, text, color, fontsize=10, text_color="white"):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.15,
    )
    ax.add_patch(rect)
    rect2 = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=2, edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect2)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=INK, weight="bold", wrap=True)


def arrow(ax, xy1, xy2, color=GRAY):
    ax.annotate("", xy=xy2, xytext=xy1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))


def main() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Input
    box(ax, (4, 7), 2, 0.7, "Input radiograph\n(real or synthetic)", INK, fontsize=9)

    # Three streams
    box(ax, (0.3, 5), 2.6, 0.9, "Spatial stream\nResNet-18/50, EfficientNet,\nConvNeXt, or ViT", BLUE, fontsize=8)
    box(ax, (3.7, 5), 2.6, 0.9, "Frequency stream\nFFT / DCT / wavelet\nradial-binned profile", ORANGE, fontsize=8)
    box(ax, (7.1, 5), 2.6, 0.9, "Noise-residual stream\nimage minus denoised\nversion of itself", AQUA, fontsize=8)

    arrow(ax, (5, 7), (1.6, 5.9))
    arrow(ax, (5, 7), (5, 5.9))
    arrow(ax, (5, 7), (8.4, 5.9))

    # Fusion
    box(ax, (2.5, 3.2), 5, 0.9, "Attention-based fusion", VIOLET, fontsize=10)
    arrow(ax, (1.6, 5), (3.5, 4.1))
    arrow(ax, (5, 5), (5, 4.1))
    arrow(ax, (8.4, 5), (6.5, 4.1))

    # Head
    box(ax, (3.5, 1.6), 3, 0.8, "Classification head\n(dropout + linear)", INK, fontsize=9)
    arrow(ax, (5, 3.2), (5, 2.4))

    # Output
    box(ax, (3.5, 0.2), 3, 0.7, "Real vs. AI-generated", GRAY, fontsize=9)
    arrow(ax, (5, 1.6), (5, 0.9))

    fig.suptitle("VetRad-ForensicsNet: three-stream architecture", fontsize=13, y=0.98)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "architecture_diagram.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Figura guardada en {out_path}")


if __name__ == "__main__":
    main()
