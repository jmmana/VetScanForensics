"""Preprocesamiento compartido y validacion del holdout real."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def random_resize_round_trip(img: Image.Image, rng: np.random.Generator,
                            size_range=(16, 128), final_size=128) -> Image.Image:
    """Aplica el mismo resize aleatorio a ambas clases antes de las dos ramas."""
    intermediate = int(rng.integers(size_range[0], size_range[1] + 1))
    img = img.resize((intermediate, intermediate), Image.BICUBIC)
    return img.resize((final_size, final_size), Image.BICUBIC)


def load_real_split(split_file: Path, real_dir: Path):
    split = json.loads(Path(split_file).read_text())
    pool, holdout = split['gen_pool'], split['holdout']
    total = {p.name for pattern in ('*.png', '*.jpg') for p in real_dir.glob(pattern)}
    assert pool and holdout, 'Ambos grupos deben contener imagenes'
    assert len(set(pool)) == len(pool), 'Duplicados en gen_pool'
    assert len(set(holdout)) == len(holdout), 'Duplicados en holdout'
    assert not set(pool) & set(holdout), 'Overlap entre gen_pool y holdout'
    assert set(pool) | set(holdout) == total, 'El split no corresponde al pool real actual'
    assert len(pool) + len(holdout) == len(total), 'Total incorrecto'
    return split
