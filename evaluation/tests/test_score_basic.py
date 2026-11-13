"""Acceptance checks for the standalone graph scoring contract."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from formal_graph import make_scorer
from score_basic import score_rows


class BasicScoreTests(unittest.TestCase):
    def test_scored_incomplete_and_gold_error(self):
        gold = "person.n.01 run.v.01 Agent -1"
        refs = [dict(example_id=i, condition="test", gold_sbn=sbn)
                for i, sbn in (("same", gold), ("different", gold), ("blank", gold),
                               ("absent", gold), ("bad_gold", "not SBN"))]
        preds = [dict(example_id=i, condition="test", pred_sbn=sbn)
                 for i, sbn in (("same", "run.v.01 Agent +1 person.n.01"),
                                ("different", "person.n.01 run.v.01"),
                                ("blank", ""), ("bad_gold", gold))]
        same, different, blank, absent, bad_gold = score_rows(refs, preds, make_scorer())
        self.assertEqual(same["f1"], 1)
        self.assertEqual(different["score_status"], "scored")
        self.assertLess(different["f1"], 1)
        self.assertEqual(different["precision"], 1)
        self.assertLess(different["recall"], 1)
        self.assertEqual((same["relation_matched"], same["relation_total"], same["relation_recall"]),
                         (1, 1, 1))
        self.assertEqual((different["relation_matched"], different["relation_total"],
                          different["relation_recall"]), (0, 1, 0))
        self.assertIsNone(blank["relation_total"])
        self.assertIsNone(blank["precision"])
        self.assertIsNone(blank["recall"])
        self.assertEqual((blank["pred_status"], blank["f1"]), ("invalid", 0))
        self.assertEqual((absent["pred_status"], absent["f1"]), ("missing", 0))
        self.assertEqual((bad_gold["pred_status"], bad_gold["score_status"], bad_gold["f1"]),
                         ("valid", "gold_error", None))

    def test_unmatched_and_duplicate_keys_are_rejected(self):
        row = dict(example_id="x", condition="test", pred_sbn="person.n.01")
        with self.assertRaisesRegex(ValueError, "no matching reference"):
            list(score_rows([], [row], None))
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            list(score_rows([], [row, row], None))


if __name__ == "__main__":
    unittest.main()
