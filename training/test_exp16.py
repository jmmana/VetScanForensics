"""Verificación de features, cobertura, procedencia y CLI de EXP-16."""
import importlib.util
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision import models

import experiment_common as common
from posthoc_common import digest

spec = importlib.util.spec_from_file_location('latent16', common.HERE / '16_latent_space.py')
latent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(latent)


class LatentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.args = common.parser('test').parse_args([])
        cls.generators, _ = common.resolve_generators(None)
        cls.splits = latent.attribution.attribution_splits(cls.generators, cls.args)
        cls.rows = latent.all_points(cls.generators, cls.args, cls.splits)

    def test_embedding_is_64d_and_reconstructs_logits(self):
        constructor = models.resnet18
        with patch.object(common.base.models, 'resnet18', side_effect=lambda **kw: constructor(weights=None)):
            model = latent.attribution.AttributionModel(len(self.generators) + 1).eval()
        rows = self.rows[:2]
        features = latent.extract_embeddings(model, rows, self.args)
        self.assertEqual(features.shape, (2, 64))
        self.assertTrue(np.isfinite(features).all())
        group = ([common.ROOT / r['path'] for r in rows], [r['label'] for r in rows])
        image, frequency, _ = next(iter(latent.attribution.loader(group, self.args)))
        with torch.inference_mode():
            logits = model(image, frequency)
            reconstructed = model.head[-1](torch.from_numpy(features))
        torch.testing.assert_close(logits, reconstructed)
        np.testing.assert_array_equal(features, latent.extract_embeddings(model, rows, self.args))

    def test_manifest_covers_every_file_and_tracks_splits(self):
        real = common.load_real_split(self.args.split_file, common.base.REAL_DIR)
        expected = {'real': {p.resolve() for p in common.base.REAL_DIR.glob('*')
                             if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}}}
        expected.update({name: {p.resolve() for p in files} for name, files in self.generators.items()})
        for name, paths in expected.items():
            rows = [r for r in self.rows if r['generator'] == name]
            self.assertEqual(len(rows), len(paths))
            self.assertEqual({(common.ROOT / r['path']).resolve() for r in rows}, paths)
        for i, row in enumerate(self.rows):
            self.assertEqual(row['preprocessing_seed'], 42 + i)
            if row['generator'] == 'real':
                self.assertIn((common.ROOT / row['path']).name, real[row['split']])
                self.assertEqual(row['attribution_split'] == 'test', row['split'] == 'holdout')
        for name, (paths, _) in zip(('train', 'val', 'test'), self.splits):
            actual = {(common.ROOT / r['path']).resolve() for r in self.rows if r['attribution_split'] == name}
            self.assertEqual(actual, {p.resolve() for p in paths})

    def test_missing_generator_cli_without_traceback(self):
        program = '''
import importlib.util, sys, tempfile
from pathlib import Path
import experiment_common as common
spec = importlib.util.spec_from_file_location('latent', common.HERE / '16_latent_space.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
split_file = str(common.ROOT / 'data/splits/real_split.json')
with tempfile.TemporaryDirectory() as directory:
    common.HERE = common.ROOT = Path(directory)
    (common.HERE / 'GENERATORS.md').write_text('```json\\n{"stylegan2ada": {"directory": "missing"}}\\n```')
    sys.argv = ['exp16', '--generators', 'stylegan2ada', '--split-file', split_file]
    common.cli(module.main)
'''
        result = subprocess.run([sys.executable, '-c', program], cwd=common.HERE, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Corre primero el generador stylegan2ada', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_saved_embeddings_and_projections(self):
        directory = common.HERE / 'exp16_latent_space'
        if not (directory / 'embeddings.npz').exists():
            self.skipTest('Ejecutar EXP-16 completo para validar artefactos')
        with np.load(directory / 'embeddings.npz', allow_pickle=False) as saved:
            self.assertEqual(saved['embeddings'].shape, (len(self.rows), 64))
            self.assertTrue(np.isfinite(saved['embeddings']).all())
            for field, source in [('paths', 'path'), ('labels', 'label'), ('generator', 'generator'),
                                  ('split', 'split'), ('attribution_split', 'attribution_split'),
                                  ('preprocessing_seed', 'preprocessing_seed')]:
                np.testing.assert_array_equal(saved[field], [r[source] for r in self.rows])
            metadata = json.loads(str(saved['metadata_json']))
        self.assertEqual(metadata['checkpoint']['sha256'], digest(common.ROOT / metadata['checkpoint']['path']))
        self.assertEqual(metadata['checkpoint']['identity'], latent.expected_identity(self.generators, self.args, self.splits))
        with np.load(directory / 'projections.npz', allow_pickle=False) as saved:
            for method in ('umap', 'tsne'):
                self.assertEqual(saved[method].shape, (len(self.rows), 2))
                self.assertTrue(np.isfinite(saved[method]).all())
                self.assertTrue((np.ptp(saved[method], axis=0) > 0).all())
                self.assertGreater((directory / f'{method}_projection.png').stat().st_size, 10000)


if __name__ == '__main__':
    unittest.main()
