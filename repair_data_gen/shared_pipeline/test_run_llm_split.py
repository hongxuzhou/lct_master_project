import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from repair_data_gen.rsr_generation import run_llm
from repair_data_gen.rsr_generation.audit import source_key


PROMPTS = Path(__file__).resolve().parents[1] / "rsr_generation" / "prompts" / "base"


def fake_reply(*_args, **_kwargs):
    answer = {
        "status": "no_site",
        "reason": "test",
        "devices": [],
    }
    envelope = {
        "is_error": False,
        "subtype": "success",
        "result": "```json\n" + json.dumps(answer) + "\n```",
        "num_turns": 1,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    return json.dumps(envelope), 0.1


class SplitRunTest(unittest.TestCase):
    def test_audit_source_key_prefers_request_id(self):
        row = {"request_id": "request-2", "repair_type": "tense", "doc_id": "p00/d0001"}
        self.assertEqual(source_key(row), "request-2")
        row["request_id"] = ""
        self.assertEqual(source_key(row), "legacy::tense::p00/d0001")

    def test_request_id_resume_and_results_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample.tsv"
            fields = ["request_id", "doc_id", "repair_type", "nl_clean", "sbn_clean"]
            with sample.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerow({
                    "request_id": "dev_base::content_word::p00/d0001",
                    "doc_id": "p00/d0001",
                    "repair_type": "content_word",
                    "nl_clean": "Test.",
                    "sbn_clean": "entity.n.01",
                })
            results = root / "dev-results"
            args = SimpleNamespace(
                docs=PROMPTS,
                sample=sample,
                only=[],
                model="fake/model",
                tag="dev",
                results_dir=results,
                config=None,
                backend="claude",
                workers=1,
                limit=0,
            )
            with patch.object(run_llm, "call_model", side_effect=fake_reply):
                run_llm.cmd_run(args)
                run_llm.cmd_run(args)
            output = results / "fake_model.dev.tsv"
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["request_id"], "dev_base::content_word::p00/d0001")
            self.assertTrue((results / "raw" / "fake_model.dev").is_dir())

    def test_legacy_sample_without_request_id_still_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample.tsv"
            fields = ["doc_id", "repair_type", "nl_clean", "sbn_clean"]
            with sample.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerow({
                    "doc_id": "p00/d0001",
                    "repair_type": "content_word",
                    "nl_clean": "Test.",
                    "sbn_clean": "entity.n.01",
                })
            results = root / "legacy-results"
            args = SimpleNamespace(
                docs=PROMPTS,
                sample=sample,
                only=[],
                model="fake/model",
                tag="legacy",
                results_dir=results,
                config=None,
                backend="claude",
                workers=1,
                limit=0,
            )
            with patch.object(run_llm, "call_model", side_effect=fake_reply):
                run_llm.cmd_run(args)
            with (results / "fake_model.legacy.tsv").open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                rows = list(reader)
                self.assertNotIn("request_id", reader.fieldnames)
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
