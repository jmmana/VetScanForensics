"""Checks de contrato y fugas; las métricas científicas salen solo de los CLI."""
import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
import experiment_common as exp


class ExperimentsTest(unittest.TestCase):
    def test_base_detector_and_dataset_are_preserved(self):
        torch.manual_seed(42)
        original = exp.base.DualStreamDetector().eval()
        torch.manual_seed(42)
        current = exp.Detector().eval()
        self.assertEqual(list(original.state_dict()), list(current.state_dict()))
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, current.state_dict()[name]), name)
        generators, _ = exp.resolve_generators(["dcgan"])
        path = generators["dcgan"][:1]
        before = exp.base.RadiographDataset(path, [1], False)[0]
        after = exp.FeatureDataset(path, [1], False, ["spatial", "fft"])[0]
        for left, right in zip(before, after):
            self.assertTrue(torch.equal(left, right))

    def test_registry_and_missing_generator(self):
        generators, missing = exp.resolve_generators(["dcgan"])
        self.assertGreater(len(generators["dcgan"]), 0)
        with self.assertRaisesRegex(ValueError, "Generador desconocido"):
            exp.resolve_generators(["nonexistent"])
        with patch.object(Path, "glob", return_value=[]):
            with self.assertRaisesRegex(ValueError, "Corre primero el generador stylegan2ada"):
                exp.resolve_generators(["stylegan2ada"])

    def test_split_reuses_original_and_is_disjoint(self):
        args = exp.parser("test").parse_args([])
        generators, _ = exp.resolve_generators(["dcgan"])
        with contextlib.redirect_stdout(io.StringIO()):
            actual = exp.split_for(generators["dcgan"], args)
            original = exp.base.load_dataset_splits()
        self.assertEqual(actual, original)
        groups = [set(paths) for paths, _ in actual]
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        holdout = exp.load_real_split(args.split_file, exp.base.REAL_DIR)["holdout"]
        self.assertEqual({p.name for p, label in zip(*actual[2]) if label == 0}, set(holdout))

    def test_frequency_profiles_and_all_feature_variants(self):
        gray = np.random.default_rng(42).random((128, 128), dtype=np.float32)
        profiles = [exp.radial_power_spectrum(gray), exp.dct_profile(gray), exp.wavelet_profile(gray)]
        for profile in profiles:
            self.assertEqual(profile.shape, (32,))
            self.assertTrue(np.isfinite(profile).all())
            self.assertGreaterEqual(profile.min(), 0)
            self.assertLessEqual(profile.max(), 1)
        for i in range(3):
            for j in range(i):
                self.assertFalse(np.allclose(profiles[i], profiles[j]))
        for features in exp.FEATURES:
            # Tiny spatial stub tests fusion/gradients without repeated ImageNet construction.
            with patch.object(exp, "spatial_backbone", return_value=(torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()), 3, {})):
                model = exp.Detector(features=features)
            width = 32 * len([f for f in model.features if f != "spatial"])
            output = model(torch.randn(2, 3, 128, 128), torch.randn(2, width))
            self.assertEqual(output.shape, (2,))
            output.sum().backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_backbone_shapes_without_downloading_weights(self):
        # Shape-only test, meta tensors: NOT pretrained training or reported metrics.
        names = {"resnet18": "resnet18", "resnet50": "resnet50", "efficientnet_b0": "efficientnet_b0",
                 "convnext_tiny": "convnext_tiny", "vit_small": "vit_b_16"}
        for requested, actual in names.items():
            constructor = getattr(exp.models, actual)
            with patch.object(exp.models, actual, side_effect=lambda weights, fn=constructor: fn(weights=None)):
                with torch.device("meta"):
                    model = exp.Detector(requested).eval()
                    output = model(torch.empty(2, 3, 128, 128), torch.empty(2, 32))
            self.assertEqual(output.shape, (2,))
            self.assertEqual(model.backbone_info["actual"], actual)

    def test_nine_metrics_against_known_confusion_matrix(self):
        args = exp.parser("test").parse_args([])
        model = exp.Detector(features="fft")
        labels = [0, 0, 0, 1, 1]
        probs = [0.1, 0.2, 0.8, 0.4, 0.9]
        with patch.object(exp, "loader", return_value=None), patch.object(exp.base, "run_epoch", return_value=(0.3, 0.0, probs, labels)):
            result = exp.evaluate(model, None, args)
        self.assertEqual(len(result["metrics"]), 9)
        self.assertEqual(result["confusion_matrix"], [[2, 1], [1, 1]])
        self.assertAlmostEqual(result["metrics"]["specificity"], 2 / 3)
        self.assertAlmostEqual(result["metrics"]["balanced_accuracy"], balanced_accuracy_score(labels, np.array(probs) >= 0.5))
        self.assertAlmostEqual(result["metrics"]["accuracy"], 3 / 5)


if __name__ == "__main__":
    unittest.main()
