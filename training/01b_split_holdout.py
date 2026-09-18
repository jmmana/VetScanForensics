"""EXP-01: reserva un holdout que nunca participa en el entrenamiento del GAN."""

import json
from pathlib import Path

import numpy as np

from preprocessing_common import load_real_split

ROOT = Path(__file__).resolve().parent.parent


def main():
    real_dir = ROOT / 'data/real'
    target = ROOT / 'data/splits/real_split.json'
    if not target.exists():
        names = sorted(p.name for pattern in ('*.png', '*.jpg') for p in real_dir.glob(pattern))
        assert len(names) >= 2, 'Se necesitan al menos dos imagenes reales'
        shuffled = np.random.default_rng(42).permutation(names).tolist()
        n_pool = round(0.8 * len(names))
        split = dict(seed=42, gen_pool=sorted(shuffled[:n_pool]), holdout=sorted(shuffled[n_pool:]))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as f:
            json.dump(split, f, indent=2)
    else:
        print(f'Split existente conservado: {target}')
    split = load_real_split(target, real_dir)
    print(f"gen_pool={len(split['gen_pool'])}, holdout={len(split['holdout'])}; "
          'total verificado y sin overlap (asserts OK)')


if __name__ == '__main__':
    main()
