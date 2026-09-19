"""Comprueba el adaptador de features contra las APIs de imágenes oficiales."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

spec = importlib.util.spec_from_file_location("exp17", Path(__file__).with_name("17_image_quality.py"))
exp17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp17)


class ImageQualityTests(unittest.TestCase):
    def test_saved_results_use_only_gen_pool(self):
        path = exp17.HERE / "exp17_image_quality/results.json"
        if not path.exists():
            self.skipTest("Ejecutar primero EXP-17 para auditar sus resultados")
        result = json.loads(path.read_text())
        split = json.loads((exp17.ROOT / "data/splits/real_split.json").read_text())
        reference = {Path(row["path"]).name for row in result["real_manifest"]}
        self.assertEqual(reference, set(split["gen_pool"]))
        self.assertFalse(reference & set(split["holdout"]))
        available, missing = exp17.resolve_generators(None)
        self.assertEqual(result["unavailable_generators"], missing)
        self.assertEqual({r["generator"] for r in result["results"]}, set(available))
        for row in result["results"]:
            self.assertEqual(row["n_real"], len(reference))
            self.assertEqual(row["n_fake"], len(available[row["generator"]]))
            self.assertEqual(len(row["fake_manifest"]), row["n_fake"])
            for field in ("paired_metrics", "ssim", "psnr", "lpips"):
                self.assertIsNone(row[field])
            self.assertTrue(all(exp17.np.isfinite(row[k]) for k in ("fid", "kid_mean", "kid_std")))
            values = row["fid_bootstrap"]["values"]
            self.assertEqual(len(values), result["protocol"]["fid_bootstraps"])
            exp17.np.testing.assert_allclose(row["fid_bootstrap"]["percentile_interval"],
                                            exp17.np.quantile(values, [0.025, 0.975]))

    def test_missing_generator_is_readable_error(self):
        _, missing = exp17.resolve_generators(None)
        if not missing:
            self.skipTest("Todos los generadores están disponibles")
        with tempfile.TemporaryDirectory() as tmp:
            run = subprocess.run([sys.executable, str(exp17.HERE / "17_image_quality.py"),
                                  "--generators", missing[0], "--output-dir", tmp],
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 2)
            self.assertIn(f"Corre primero el generador {missing[0]}", run.stderr)
            self.assertNotIn("Traceback", run.stderr)
            self.assertFalse((Path(tmp) / "results.json").exists())

    def test_preprocessing_and_shared_features_match_direct_metrics(self):
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

        torch.set_num_threads(2)
        torch.hub.set_dir(str(exp17.HERE / "checkpoints/exp17"))
        split = exp17.load_real_split(exp17.ROOT / "data/splits/real_split.json", exp17.ROOT / "data/real")
        real_paths = [exp17.ROOT / "data/real" / p for p in sorted(split["gen_pool"])[:3]]
        available, _ = exp17.resolve_generators(None)
        fake_paths = next(iter(available.values()))[:3]
        real = torch.stack([exp17.Images(real_paths)[i] for i in range(3)])
        fake = torch.stack([exp17.Images(fake_paths)[i] for i in range(3)])
        self.assertEqual(real.shape, (3, 3, 128, 128))
        self.assertEqual(real.dtype, torch.uint8)
        torch.testing.assert_close(real[:, 0], real[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(real[:, 1], real[:, 2], rtol=0, atol=0)
        extractor = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).eval()
        with torch.inference_mode():
            r, f = extractor(real)[0].double(), extractor(fake)[0].double()
            # El default reciente de FID activa antialias; nuestro protocolo
            # explícito usa el resize original TF de torch-fidelity, sin antialias.
            direct_fid = FrechetInceptionDistance(feature=2048, antialias=False)
            direct_fid.update(real, real=True)
            direct_fid.update(fake, real=False)
            expected = float(direct_fid.compute())
            self.assertAlmostEqual(exp17.calculate_fid(r, f), expected, delta=2e-4)
            direct_kid = KernelInceptionDistance(feature=direct_fid.inception, subsets=5, subset_size=2)
            direct_kid.update(real, real=True)
            direct_kid.update(fake, real=False)
            cached_kid = KernelInceptionDistance(feature=exp17.PrecomputedInception(), subsets=5, subset_size=2)
            cached_kid.update(r, real=True)
            cached_kid.update(f, real=False)
            torch.manual_seed(42)
            expected_kid = direct_kid.compute()
            torch.manual_seed(42)
            actual_kid = cached_kid.compute()
            for actual, expected in zip(actual_kid, expected_kid):
                self.assertAlmostEqual(float(actual), float(expected), delta=2e-5)


if __name__ == "__main__":
    unittest.main()
