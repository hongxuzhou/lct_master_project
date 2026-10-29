import json
import tempfile
import unittest
from pathlib import Path

from repair_data_gen.shared_pipeline.split_config import load_config


DEV_CONFIG = Path(__file__).resolve().parents[1] / "splits" / "dev" / "config.json"


class SplitConfigTest(unittest.TestCase):
    def test_dev_config_loads(self):
        config = load_config(DEV_CONFIG)
        self.assertEqual(config.split, "dev")
        self.assertEqual(config.base_cap_per_type["content_word"], 60)
        self.assertEqual(config.candidate_plan["base_per_type"]["content_word"], 75)
        self.assertEqual(config.marker_policy.held_out_markers, ("sorry",))

    def test_marker_sets_must_be_disjoint(self):
        raw = json.loads(DEV_CONFIG.read_text(encoding="utf-8"))
        raw["marker_policy"]["markers"].append("sorry")
        with tempfile.TemporaryDirectory() as directory:
            split_dir = Path(directory) / "dev"
            split_dir.mkdir()
            path = split_dir / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disjoint"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
