"""Controles de arquitectura, aislamiento y errores de EXP-19."""
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

spec = importlib.util.spec_from_file_location('attribution19', common.HERE / '19_generator_attribution.py')
attribution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attribution)


class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.args = common.parser('test').parse_args([])
        cls.generators, _ = common.resolve_generators(None)
        cls.splits = attribution.attribution_splits(cls.generators, cls.args)

    def test_output_shape_and_only_final_layer_changes(self):
        constructor = models.resnet18
        with patch.object(common.base.models, 'resnet18', side_effect=lambda **kw: constructor(weights=None)):
            reference = common.base.DualStreamDetector()
            for n in (2, 3, 4, 6):
                model = attribution.AttributionModel(n).eval()
                with torch.no_grad():
                    logits = model(torch.zeros(2, 3, 128, 128), torch.zeros(2, 32))
                self.assertEqual(tuple(logits.shape), (2, n))
                self.assertTrue(torch.isfinite(logits).all())
                torch.testing.assert_close(logits.softmax(1).sum(1), torch.ones(2))
                changed = [k for k, v in model.state_dict().items()
                           if v.shape != reference.state_dict()[k].shape]
                self.assertEqual(changed, ['head.3.weight', 'head.3.bias'])

    def test_holdout_and_per_generator_partitions(self):
        split = common.load_real_split(self.args.split_file, common.base.REAL_DIR)
        holdout = {common.base.REAL_DIR / p for p in split['holdout']}
        pool = {common.base.REAL_DIR / p for p in split['gen_pool']}
        for idx, (paths, labels) in enumerate(self.splits):
            reals = {p for p, y in zip(paths, labels) if y == 0}
            self.assertEqual(len(paths), len(set(paths)))
            if idx < 2:
                self.assertFalse(set(paths) & holdout)
                self.assertTrue(reals <= pool)
            else:
                self.assertEqual(reals, holdout)
                self.assertFalse(set(paths) & pool)
        for class_id, files in enumerate(self.generators.values(), 1):
            for actual, binary in zip(self.splits, common.split_for(files, self.args)):
                self.assertEqual([p for p, y in zip(*actual) if y == class_id],
                                 [p for p, y in zip(*binary) if y == 1])
        sets = [set(paths) for paths, _ in self.splits]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(sets[i] & sets[j])

    def test_missing_generator_cli_without_traceback(self):
        # Registro aislado: funciona incluso después de que llegue StyleGAN2-ADA.
        program = '''
import importlib.util, sys, tempfile
from pathlib import Path
import experiment_common as common
spec = importlib.util.spec_from_file_location('exp19', common.HERE / '19_generator_attribution.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
split_file = str(common.ROOT / 'data/splits/real_split.json')
with tempfile.TemporaryDirectory() as directory:
    common.HERE = common.ROOT = Path(directory)
    (common.HERE / 'GENERATORS.md').write_text('```json\\n{"stylegan2ada": {"directory": "missing"}}\\n```')
    sys.argv = ['exp19', '--generators', 'stylegan2ada', '--split-file', split_file]
    common.cli(module.main)
'''
        outcome = subprocess.run([sys.executable, '-c', program], cwd=common.HERE,
                                 capture_output=True, text=True)
        self.assertEqual(outcome.returncode, 2)
        self.assertIn('Corre primero el generador stylegan2ada', outcome.stderr)
        self.assertNotIn('Traceback', outcome.stderr)

    def test_saved_metrics_and_robustness(self):
        path = common.HERE / 'exp19_attribution/results.json'
        if not path.exists():
            self.skipTest('Ejecutar EXP-19 para validar resultados persistidos')
        result = json.loads(path.read_text())
        expected_conditions = [(f, s) for f, levels in attribution.robustness.LEVELS.items() for s in levels]
        self.assertEqual([(r['transform'], r['severity']) for r in result['degradation']], expected_conditions)
        self.assertEqual(len(result['predictions']), len(expected_conditions))
        for row, details in zip(result['degradation'], result['predictions']):
            actual = attribution.metrics(details, result['classes'])
            for name, value in actual.items():
                self.assertEqual(row[name], value)
            self.assertEqual(details['labels'], result['predictions'][0]['labels'])
            self.assertEqual(actual['n'], len(result['identity']['manifest']['test']))
            probabilities = np.asarray(details['probabilities'])
            self.assertEqual(probabilities.shape, (actual['n'], len(result['classes'])))
            np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
        self.assertEqual(result['evaluation']['confusion_matrix'], result['degradation'][0]['confusion_matrix'])


if __name__ == '__main__':
    unittest.main()
