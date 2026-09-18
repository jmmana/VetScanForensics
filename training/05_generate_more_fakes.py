"""
Genera un lote grande de radiografias sinteticas adicionales a partir del
generador ya entrenado (training/checkpoints/generator.pth), sin volver a
entrenar. Se usa para inflar el dataset que se publica en Kaggle: mientras
mas grande, mas util para que otros lo reutilicen, aunque el conjunto de
entrenamiento del detector siga usando una muestra balanceada.

Uso:
    python training/05_generate_more_fakes.py --n-fake 2000
"""

import argparse
from pathlib import Path

import torch

import importlib.util

ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = ROOT / "training" / "checkpoints" / "generator.pth"
FAKE_DIR = ROOT / "data" / "fake"

spec = importlib.util.spec_from_file_location(
    "gan_module", ROOT / "training" / "02_train_generator_and_make_fakes.py"
)
gan_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gan_module)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fake", type=int, default=2000)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    device = gan_module.get_device()
    print(f"Usando device: {device}")

    if not CKPT_PATH.exists():
        raise RuntimeError(
            f"No encontre {CKPT_PATH}. Corre primero 02_train_generator_and_make_fakes.py"
        )

    gen = gan_module.Generator().to(device)
    gen.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    gen.eval()

    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    from torchvision.utils import save_image

    with torch.no_grad():
        for i in range(args.n_fake):
            noise = torch.randn(1, gan_module.LATENT_DIM, 1, 1, device=device)
            img = gen(noise).squeeze(0)
            img = (img + 1) / 2
            idx = args.start_index + i
            save_image(img, FAKE_DIR / f"fake_{idx:05d}.png")

    total = len(list(FAKE_DIR.glob("*.png")))
    print(f"Generadas {args.n_fake} imagenes adicionales. Total en {FAKE_DIR}: {total}")


if __name__ == "__main__":
    main()
