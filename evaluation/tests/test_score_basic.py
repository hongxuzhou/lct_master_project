"""Acceptance checks for the standalone graph scoring contract."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from formal_graph import Graph, full_score, make_scorer
from score_basic import require_panel, score_rows


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

    def test_scoped_condition_and_inversion(self):
        gold = ("female.n.02 order.v.01 Agent -1 CORRECTION <1 tea.n.01 ThemeOf -1 "
                "CONJUNCTION <2 coffee.n.01 ThemeOf -2")
        wrong_scope = gold.replace("Agent -1", "Agent -1 Theme +1").replace("tea.n.01 ThemeOf -1", "tea.n.01")
        # Appendix A: the first Theme is now asserted in the enclosing DRS.
        score = full_score(make_scorer(), Graph(gold), Graph(wrong_scope), True)
        self.assertLess(score["f1"], 1)
        self.assertEqual(score["relation_matched"], score["relation_total"] - 1)
        # CONJUNCTION adds its conditions to the parent DRS. Forward Theme
        # on the parent and inverse ThemeOf on the accepted object agree.
        accepted_forward = gold.replace("Agent -1", "Agent -1 Theme +2").replace("coffee.n.01 ThemeOf -2", "coffee.n.01")
        score = full_score(make_scorer(), Graph(gold), Graph(accepted_forward), True)
        self.assertEqual(score["f1"], 1)
        self.assertEqual(score["relation_recall"], 1)
        # Ordinary same-context role inversion and index renumbering agree.
        a = Graph("person.n.01 run.v.01 Agent -1")
        b = Graph("run.v.01 person.n.01 AgentOf -1")
        self.assertEqual(full_score(make_scorer(), a, b)["f1"], 1)

    def test_literal_case_is_not_normalized(self):
        gold = Graph('entity.n.01 Name "US"')
        pred = Graph('entity.n.01 Name "us"')
        result = full_score(make_scorer(), gold, pred, True)
        self.assertLess(result["f1"], 1)
        self.assertEqual(result["relation_recall"], 0)

    def test_instance_role_is_not_a_concept_declaration(self):
        gold = Graph("entity.n.01 InstanceOf +1 animal.n.01")
        pred = Graph("entity.n.01 animal.n.01")
        result = full_score(make_scorer(), gold, pred, True)
        self.assertEqual(result["relation_total"], 1)
        self.assertEqual(result["relation_matched"], 0)
        self.assertLess(result["f1"], 1)
        self.assertEqual(full_score(make_scorer(), gold, gold)["f1"], 1)

    def test_complete_basic_panel_required(self):
        refs = [dict(example_id=x, condition="C3") for x in ("a", "b")]
        require_panel(refs[::-1], refs)
        for actual in ([], refs[:1], refs + [dict(example_id="c", condition="C3")]):
            with self.assertRaisesRegex(ValueError, "missing|extra"):
                require_panel(actual, refs)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            require_panel(refs + refs[:1], refs)


if __name__ == "__main__":
    unittest.main()
