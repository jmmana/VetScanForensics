"""
Prepara las radiografías reales (perro, tórax, lateral) que sirven de base
para el detector de imágenes veterinarias falsas generadas por IA.

Fuente: "Radiographic Dataset for VHS determination learning process"
Flores Duenas, C.A.; Gaxiola Camacho, S.M.; Montano Gomez, M.F.
Mendeley Data, V1, DOI: 10.17632/ktx4cj55pn.1, licencia CC BY 4.0
https://data.mendeley.com/datasets/ktx4cj55pn/1

Mendeley sirve la descarga real vía JavaScript (no hay endpoint REST
estable para automatizarla), así que este script asume que ya bajaste
el zip manualmente desde el link de arriba (boton "Download all 152 files")
y lo dejaste en data/raw/.
"""

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
REAL_DIR = ROOT / "data" / "real"

SOURCE_CITATION = (
    "Flores Duenas, C.A.; Gaxiola Camacho, S.M.; Montano Gomez, M.F. (2022), "
    "\"Radiographic Dataset for VHS determination learning process\", Mendeley Data, V1, "
    "doi: 10.17632/ktx4cj55pn.1"
)


def find_source_zip() -> Path | None:
    candidates = list(RAW_DIR.glob("*.zip"))
    if not candidates:
        candidates = list((Path.home() / "Downloads").glob("*ktx4cj55pn*.zip"))
        candidates += list((Path.home() / "Downloads").glob("*VHS*.zip"))
    return candidates[0] if candidates else None


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REAL_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = find_source_zip()
    if zip_path is None:
        print(
            "No encontre el zip del dataset.\n\n"
            "Pasos manuales (una sola vez, toma 10 segundos):\n"
            "  1. Abre https://data.mendeley.com/datasets/ktx4cj55pn/1\n"
            "  2. Click en 'Download all 152 files (...)'\n"
            f"  3. Mueve el .zip descargado a: {RAW_DIR}\n"
            "  4. Vuelve a correr este script\n"
        )
        sys.exit(1)

    print(f"Extrayendo {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR / "extracted")

    images = sorted(
        p
        for p in (RAW_DIR / "extracted").rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        print("El zip no contiene imagenes PNG/JPG reconocibles. Revisa el contenido manualmente.")
        sys.exit(1)

    for i, img_path in enumerate(images):
        dest = REAL_DIR / f"real_{i:04d}{img_path.suffix.lower()}"
        shutil.copy(img_path, dest)

    print(f"Listo: {len(images)} radiografias reales copiadas a {REAL_DIR}")
    print(f"\nCitar siempre como:\n{SOURCE_CITATION}")


if __name__ == "__main__":
    main()
