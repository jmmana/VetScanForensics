"""
Figura de EXP-00: muestra que dos grupos de imagenes 100% REALES, procesadas
con dos rutas de resize distintas, son indistinguibles a simple vista pero
100% separables para el detector (ver exp00_confound/exp00_result.json).

Uso:
    python training/make_exp00_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = ROOT / "data" / "real"
FIG_DIR = ROOT / "paper" / "figures"
RESULT_PATH = ROOT / "training" / "exp00_confound" / "exp00_result.json"

GROUP_A_COLOR = "#4a3aa7"  # slot 7 (violeta) del palette validado
GROUP_B_COLOR = "#1baf7a"  # slot 3 (aqua) del palette validado
INK = "#0b0b0b"
N_EXAMPLES = 4
IMG_SIZE = 128


def resize_direct(img: Image.Image) -> Image.Image:
    return img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)


def resize_upsampled_from_seed(img: Image.Image) -> Image.Image:
    seed = img.resize((8, 8), Image.BICUBIC)
    return seed.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)


def main() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    result = json.loads(RESULT_PATH.read_text()) if RESULT_PATH.exists() else None
    real_paths = sorted(REAL_DIR.glob("*.png"))[:N_EXAMPLES]

    fig, axes = plt.subplots(2, N_EXAMPLES, figsize=(2.2 * N_EXAMPLES, 5.0))
    for col, path in enumerate(real_paths):
        img = Image.open(path).convert("L")
        for row, (fn, color) in enumerate(
            [(resize_direct, GROUP_A_COLOR), (resize_upsampled_from_seed, GROUP_B_COLOR)]
        ):
            ax = axes[row, col]
            ax.imshow(np.array(fn(img)), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

    axes[0, 0].set_ylabel("Group A: direct resize", fontsize=11, fontweight="bold", color=GROUP_A_COLOR, labelpad=10)
    axes[1, 0].set_ylabel("Group B: 8x8-seed upsample", fontsize=11, fontweight="bold", color=GROUP_B_COLOR, labelpad=10)

    subtitle = "Both rows are 100% REAL images (same underlying radiographs)"
    if result:
        subtitle += f" — detector separates A vs. B at AUC={result['test_auc']:.3f}"
    fig.suptitle(subtitle, fontsize=12, color=INK, y=1.03)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "exp00_confound_examples.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Figura guardada en {out_path}")


if __name__ == "__main__":
    main()
