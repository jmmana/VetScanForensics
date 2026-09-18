"""Controles de regresión de postproceso, métricas y atribuciones del checkpoint real."""
import csv
import importlib.util
import json
import unittest

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

import experiment_common as common


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, common.HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


robust = module('robust', '09_robustness.py')
explain = module('explain', '12_explainability.py')


class PosthocTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.model = common.base.DualStreamDetector().eval()
        cls.model.load_state_dict(torch.load(common.HERE / 'checkpoints/detector_best.pth', map_location='cpu', weights_only=True))
        cls.splits = common.load_dataset_splits()
        cls.dataset = common.base.RadiographDataset(*cls.splits[2], train=False)

    def test_identity_and_shared_hook(self):
        called = []
        def attack(image, idx):
            called.append(idx)
            return Image.new('L', image.size, 127)
        attacked = common.base.RadiographDataset(*self.splits[2], train=False, postprocess=attack)
        identity = common.base.RadiographDataset(*self.splits[2], train=False,
            postprocess=lambda image, idx: robust.transform(image, idx, 'clean', 0))
        for idx in (0, 30):  # real y falsa, conservando índices globales del test
            for a, b in zip(self.dataset[idx], identity[idx]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            image, freq, _ = attacked[idx]
            self.assertEqual(float(image.std()), 0)
            np.testing.assert_allclose(freq.numpy(), common.radial_power_spectrum(np.full((128, 128), 127 / 255, dtype=np.float32)))
        self.assertEqual(called, [0, 30])

    def test_transforms_change_pixels_and_noise_repeats(self):
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (128, 128), dtype=np.uint8))
        for family, levels in robust.LEVELS.items():
            for severity in levels:
                actual = robust.transform(image, 7, family, severity)
                self.assertEqual(actual.size, (128, 128))
                self.assertEqual(actual.mode, 'L')
                if family != 'clean':
                    self.assertFalse(np.array_equal(image, actual), (family, severity))
        np.testing.assert_array_equal(robust.transform(image, 7, 'noise', .05), robust.transform(image, 7, 'noise', .05))
        self.assertFalse(np.array_equal(robust.transform(image, 7, 'noise', .05), robust.transform(image, 8, 'noise', .05)))
        # Crop 60% del área: 99x99, no el incorrecto 77x77 (60% del lado).
        expected = image.crop((14, 14, 113, 113)).resize((128, 128), Image.Resampling.BICUBIC)
        np.testing.assert_array_equal(expected, robust.transform(image, 7, 'crop', .6))

    def test_saved_metrics_recomputed(self):
        directory = common.HERE / 'exp09_robustness'
        rows = json.loads((directory / 'results.json').read_text())
        metadata = json.loads((directory / 'metadata.json').read_text())
        with (directory / 'results.csv').open() as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 30)
        self.assertEqual(len(csv_rows), len(rows))
        for row, detail, csv_row in zip(rows, metadata['test_predictions'], csv_rows):
            labels, prob = detail['labels'], detail['probabilities']
            prediction = np.asarray(prob) >= .5
            self.assertEqual(row['n'], 89)
            for metric, actual in [('accuracy', accuracy_score(labels, prediction)),
                                   ('roc_auc', roc_auc_score(labels, prob)),
                                   ('balanced_accuracy', balanced_accuracy_score(labels, prediction))]:
                self.assertAlmostEqual(row[metric], actual)
                self.assertEqual(float(csv_row[metric]), row[metric])

    def test_gradcam_and_gradient_finite_difference(self):
        image, freq, _ = self.dataset[0]
        heat, raw, _ = explain.gradcam(self.model, image[None], freq[None])
        self.assertEqual(raw.shape, (4, 4))
        self.assertGreater(float(heat.std()), .01)
        freq = freq[None].double().requires_grad_(True)
        # Doble precisión permite diferencias finitas pequeñas para gradientes ~1e-3.
        self.model.double()
        try:
            image = image[None].double()
            gradient, = torch.autograd.grad(self.model(image, freq).sum(), freq)
            numerical = []
            with torch.no_grad():
                for i in range(32):
                    delta = torch.zeros_like(freq)
                    delta[0, i] = 1e-5
                    numerical.append(((self.model(image, freq + delta) - self.model(image, freq - delta)) / 2e-5).item())
            np.testing.assert_allclose(gradient.detach().numpy()[0], numerical, rtol=1e-4, atol=1e-8)
        finally:
            self.model.float()
        # Un objetivo independiente de las activaciones produce mapa nulo: debe fallar.
        with torch.no_grad():
            saved = self.model.head[-1].weight.clone()
            self.model.head[-1].weight.zero_()
        try:
            with self.assertRaisesRegex(ValueError, 'constante'):
                explain.gradcam(self.model, image.float(), freq.detach().float())
        finally:
            with torch.no_grad():
                self.model.head[-1].weight.copy_(saved)


if __name__ == '__main__':
    unittest.main()
