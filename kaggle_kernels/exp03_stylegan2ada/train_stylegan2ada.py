# rerun v2: dataset ya procesado completamente
"""
EXP-03: entrena StyleGAN2 con Adaptive Discriminator Augmentation (ADA) sobre
las 122 radiografias reales de gen_pool, via el paquete pip stylegan2_pytorch
(lucidrains), que incluye aumentacion diferenciable pensada para datasets
chicos, en el espiritu del metodo real de Karras et al. 2020 (arXiv:2006.06676).

Corre como Kaggle kernel con GPU. Input dataset: vet-radiographs-genpool-122.
Output: /kaggle/working/models/vetxray/*.pt (checkpoints) y
/kaggle/working/results/vetxray/*.jpg (muestras generadas durante el entrenamiento).
"""

import subprocess
import sys
from pathlib import Path

WORK_DIR = Path("/kaggle/working")
RUN_NAME = "vetxray"


def find_data_dir(slug: str = "vet-radiographs-genpool-122") -> Path:
    """La convencion de montaje de /kaggle/input/ ha cambiado entre entornos
    (a veces /kaggle/input/<slug>, a veces /kaggle/input/datasets/<owner>/<slug>).
    Buscar el directorio en vez de asumir una ruta fija evita romperse otra vez."""
    for candidate in Path("/kaggle/input").rglob(slug):
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    raise RuntimeError(f"No encontre un directorio '{slug}' con archivos bajo /kaggle/input")


DATA_DIR = find_data_dir()


def install_deps() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "stylegan2_pytorch"],
        check=True,
    )


def main() -> None:
    install_deps()
    print(f"Imagenes de entrada: {len(list(DATA_DIR.glob('*')))}")

    cmd = [
        "stylegan2_pytorch",
        "--data", str(DATA_DIR),
        "--results_dir", str(WORK_DIR / "results"),
        "--models_dir", str(WORK_DIR / "models"),
        "--name", RUN_NAME,
        "--image-size", "128",
        "--batch-size", "16",
        "--gradient-accumulate-every", "4",
        "--num-train-steps", "8000",
        "--num-workers", "2",
        "--aug-prob", "0.25",   # augmentacion diferenciable, analoga en espiritu a ADA
        "--save-every", "1000",
        "--evaluate-every", "1000",
    ]
    print("Comando:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    ckpts = sorted((WORK_DIR / "models" / RUN_NAME).glob("*.pt"))
    print(f"Checkpoints generados: {[c.name for c in ckpts]}")


if __name__ == "__main__":
    main()
