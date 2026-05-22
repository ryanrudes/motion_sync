import unittest
from pathlib import Path

from pydantic import ValidationError

from retargeting.config import RetargetingConfig, load_config
from tests.support import MINIMAL_CONFIG


class TestLoadConfig(unittest.TestCase):
    def test_load_minimal_fixture(self):
        cfg = load_config(MINIMAL_CONFIG)
        self.assertIsInstance(cfg, RetargetingConfig)
        self.assertEqual(cfg.rate.video, 30.0)
        self.assertEqual(cfg.rate.mocap, 100.0)
        self.assertIn("Left_Shoe", cfg.bodies)
        self.assertIn("vicon/Left_Shoe/Left_Shoe", cfg.time_sync_solver.smplx_joints)

    def test_load_repo_default_config(self):
        repo_cfg = Path(__file__).resolve().parents[1] / "configs" / "retargeting.yaml"
        if not repo_cfg.is_file():
            self.skipTest("configs/retargeting.yaml not present")
        cfg = load_config(repo_cfg)
        self.assertGreater(len(cfg.bodies), 0)

    def test_invalid_yaml_raises(self):
        bad = Path(__file__).resolve().parent / "fixtures" / "_bad_config.yaml"
        bad.write_text("rate:\n  video: not_a_number\npaths:\n  smplx_models: x\n")
        try:
            with self.assertRaises(ValidationError):
                load_config(bad)
        finally:
            bad.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
