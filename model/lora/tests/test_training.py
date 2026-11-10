"""Numerical tests for loss aggregation, accumulation and checkpoint selection."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import condition_rows
from train import Recorder, collate, evaluate, keep_checkpoints, save_checkpoint, train_window


class ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 5)
        self.head = torch.nn.Linear(5, 11)

    def forward(self, input_ids, labels, **kwargs):
        logits = self.head(self.embedding(input_ids))[:, :-1]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 11), labels[:, 1:].reshape(-1), ignore_index=-100)
        return types.SimpleNamespace(loss=loss)

    def save_pretrained(self, path, **kwargs):
        torch.save(self.state_dict(), path / "adapter_model.pt")


EXAMPLES = [
    {"input_ids": [1, 2, 3], "labels": [-100, -100, 3]},
    {"input_ids": [2, 3, 4, 5, 6, 7], "labels": [-100, 3, 4, 5, 6, 7]},
    {"input_ids": [4, 5, 6, 8], "labels": [-100, -100, 6, 8]},
]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = ToyLM()
        self.device = torch.device("cpu")

    def test_eval_is_global_token_mean_and_padding_invariant(self):
        all_batch = collate(EXAMPLES, 0)
        expected = float(self.model(**all_batch).loss.detach())
        split = [collate(EXAMPLES[:2], 0), collate(EXAMPLES[2:], 0)]
        individual = [collate([e], 0) for e in EXAMPLES]
        for batches in (split, individual):
            result = evaluate(self.model, batches, self.device)
            self.assertAlmostEqual(result["eval_loss"], expected, places=6)
            self.assertEqual(result["eval_target_tokens"], 8)
        naive_mean = sum(float(self.model(**b).loss.detach()) for b in individual) / 3
        self.assertGreater(abs(naive_mean - expected), 0.001)

    def test_accumulation_matches_one_large_batch_including_partial_window(self):
        other = ToyLM()
        other.load_state_dict(self.model.state_dict())
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        other_optimizer = torch.optim.SGD(other.parameters(), lr=0.01)
        train_window(self.model, [collate([e], 0) for e in EXAMPLES], optimizer, self.device, 100)
        train_window(other, [collate(EXAMPLES, 0)], other_optimizer, self.device, 100)
        for first, second in zip(self.model.parameters(), other.parameters()):
            torch.testing.assert_close(first, second)

    def test_condition_pairing_changes_only_target(self):
        rows = [{"variant": "pmb_fluent", "text": "a", "sbn_clean": "x", "sbn_repair": "x"},
                {"variant": "rsr_base", "text": "b c", "sbn_clean": "y", "sbn_repair": "z"}]
        self.assertEqual(len(condition_rows(rows, "C1")), 1)
        c2, c3 = condition_rows(rows, "C2"), condition_rows(rows, "C3")
        self.assertEqual([r["text"] for r in c2], [r["text"] for r in c3])
        self.assertNotEqual(c2[-1]["target"], c3[-1]["target"])

    def test_checkpoint_roundtrip_and_retention_keep_best_and_last(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        tokenizer = types.SimpleNamespace(save_pretrained=lambda _: None)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            paths = [save_checkpoint(self.model, tokenizer, optimizer, scheduler, output,
                                     {"step": step, "eval_loss": loss})
                     for step, loss in ((1, 0.2), (2, 0.4), (3, 0.3))]
            keep_checkpoints(paths, paths[0], paths[-1])
            self.assertEqual([p.name for p in paths], ["checkpoint-1", "checkpoint-3"])
            self.assertEqual(json.loads((paths[0] / "state.json").read_text())["eval_loss"], 0.2)
            other = ToyLM()
            other.load_state_dict(torch.load(paths[0] / "adapter_model.pt", weights_only=True))
            torch.testing.assert_close(self.model(**collate(EXAMPLES, 0)).loss,
                                       other(**collate(EXAMPLES, 0)).loss)

    def test_wandb_failure_preserves_local_records(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Recorder(Path(directory), {}, "disabled", {})
            log.run = types.SimpleNamespace(log=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("network")))
            log.log("eval", step=10, eval_loss=0.5)
            log.close(0)
            records = [json.loads(line) for line in (Path(directory)/"metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["event"] for r in records], ["eval", "wandb_logging_failed"])
            self.assertEqual(records[0]["eval_loss"], 0.5)


if __name__ == "__main__":
    unittest.main()
