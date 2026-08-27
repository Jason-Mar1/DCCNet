"""Configuration-only smoke tests; no model or Hugging Face download is invoked."""

from pathlib import Path
import sys
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG_DIR = ROOT / "configs"
EXPECTED_CONFIGS = {
    "shanghaitech_a_to_b.yaml",
    "shanghaitech_a_to_qnrf.yaml",
    "shanghaitech_b_to_a.yaml",
    "shanghaitech_b_to_qnrf.yaml",
    "shanghaitech_qnrf_to_a.yaml",
    "shanghaitech_qnrf_to_b.yaml",
    "jhu_str_to_sta.yaml",
    "jhu_sta_to_str.yaml",
}
DEPTH_REVISION = "1bee667edbfec7f9b345abaffec8c9c2bd5293c2"


class ReleaseConfigTest(unittest.TestCase):
    def test_only_published_protocol_configs_are_present(self):
        self.assertEqual({path.name for path in CONFIG_DIR.glob("*.yaml")}, EXPECTED_CONFIGS)
        self.assertEqual(list(CONFIG_DIR.glob("*.yml")), [])

    def test_paper_hyperparameters_and_checkpoint_paths(self):
        for filename in EXPECTED_CONFIGS:
            with self.subTest(filename=filename):
                with (CONFIG_DIR / filename).open("r", encoding="utf-8") as handle:
                    config = yaml.safe_load(handle)
                self.assertEqual(config["device"], "cuda:0")
                self.assertEqual(config["num_epochs"], 200)
                self.assertEqual(config["train_loader"]["batch_size"], 8)
                self.assertEqual(config["optimizer"]["name"], "adam")
                self.assertEqual(float(config["optimizer"]["params"]["lr"]), 1e-4)
                self.assertEqual(float(config["loss_weights"]["lambda_aux"]), 0.1)
                self.assertEqual(float(config["loss_weights"]["lambda_con"]), 0.01)
                self.assertEqual(float(config["foreground_threshold"]), 0.4)
                self.assertEqual(float(config["model"]["params"]["cls_thrs"]), 0.4)
                self.assertEqual(
                    float(config["train_dataset"]["params"]["foreground_threshold"]), 0.4
                )
                self.assertEqual(
                    float(config["val_dataset"]["params"]["foreground_threshold"]), 0.4
                )
                self.assertEqual(config["depth_anything"]["repo_id"], "LiheYoung/depth_anything_vits14")
                self.assertEqual(config["depth_anything"]["revision"], DEPTH_REVISION)
                self.assertEqual(config["depth_anything"]["encoder"], "vits")
                self.assertEqual(config["depth_anything"]["features"], 64)
                self.assertEqual(config["depth_anything"]["out_channels"], [48, 96, 192, 384])
                self.assertEqual(config["checkpoint"], "logs/{}/best.pth".format(config["version"]))

    def test_release_does_not_vendor_torchhub_or_download_in_this_test(self):
        self.assertFalse((ROOT / "torchhub").exists())
        # This test imports neither main.py nor a model class, so no depth
        # checkpoint/model architecture download can be triggered here.

    def test_main_parser_accepts_all_configs_without_constructing_a_model(self):
        # ``read_config`` only validates YAML; DepthAnything is constructed
        # later by build_model, so this remains an offline smoke test.
        from main import read_config

        for filename in EXPECTED_CONFIGS:
            with self.subTest(filename=filename):
                config, path = read_config(CONFIG_DIR / filename)
                self.assertEqual(path.name, filename)
                self.assertEqual(config["mode"], "final")

    def test_second_view_uses_additive_gaussian_noise(self):
        import torch
        from utils.misc import GaussianNoise

        torch.manual_seed(7)
        image = torch.full((3, 8, 8), 0.5)
        augmented = GaussianNoise(std=0.05)(image)
        self.assertFalse(torch.equal(image, augmented))
        self.assertGreater(float((augmented - image).abs().sum()), 0.0)

    def test_evaluation_skips_redundant_vgg_pretrained_download(self):
        from main import build_task

        for task, expected in (("train", None), ("test", False), ("vis", False)):
            with self.subTest(task=task):
                with mock.patch("main.build_model", side_effect=RuntimeError("stop after model call")) as builder:
                    with self.assertRaisesRegex(RuntimeError, "stop after model call"):
                        build_task({"seed": 2023}, task)
                builder.assert_called_once_with(
                    {"seed": 2023},
                    depth_checkpoint=None,
                    pretrained_backbone=expected,
                )


if __name__ == "__main__":
    unittest.main()
