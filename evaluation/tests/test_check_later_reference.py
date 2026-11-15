"""Check member recovery, C–C outcomes and their denominators independently."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_later_reference import check_rows, summarise
from formal_local import Result, joint

GOLD = ("person.n.01 EQU speaker time.n.08 TPR now borrow.v.01 Agent -2 Time -1 "
        "CORRECTION <1 coat.n.01 ThemeOf -1 CONJUNCTION <2 jacket.n.01 ThemeOf -2 "
        "CONTINUATION <1 entity.n.01 Sub -2 Sub -1 Quantity 2 warm.a.01 AttributeOf -1")
PLAIN = GOLD.replace("CORRECTION <1 ", "").replace("CONJUNCTION <2 ", "")


class LaterReferenceTests(unittest.TestCase):
    def evaluate(self, sbn, condition="C3", side="rsr", status="valid"):
        gold = dict(example_id="x", pair_id="p", language_condition=side,
                    sbn_repair=GOLD if side == "rsr" else PLAIN,
                    reference_group_index=5, antecedent_indices=[3, 4],
                    withdrawn_antecedent_indices=[3] if side == "rsr" else [])
        row = dict(example_id="x", condition=condition, pred_status=status, pred_sbn=sbn)
        return next(check_rows([row], [gold]))

    def test_rsr_joint_and_partial_members(self):
        correct = self.evaluate(GOLD)
        self.assertEqual(correct["joint"]["status"], "pass")
        self.assertTrue(correct["sub_to_cc"])
        missing = self.evaluate(GOLD.replace("Sub -2 ", ""))
        self.assertEqual(missing["member_recovered"], [False, True])
        self.assertEqual(missing["members"]["status"], "fail")
        self.assertEqual(missing["joint"]["status"], "fail")
        extra = self.evaluate(GOLD.replace("Sub -2", "Sub -5 Sub -2"))
        self.assertEqual(extra["member_recovered"], [True, True])
        self.assertEqual(extra["extra_members"], 1)
        self.assertEqual(extra["joint"]["status"], "fail")
        wrong_status = self.evaluate(PLAIN)
        self.assertEqual(wrong_status["members"]["status"], "pass")
        self.assertEqual(wrong_status["cc_outcome"], "false_negative")
        self.assertEqual(wrong_status["joint"]["status"], "fail")

    def test_c2_rsr_and_fluent(self):
        c2 = self.evaluate(PLAIN, condition="C2")
        self.assertEqual(c2["members"]["status"], "pass")
        self.assertIsNone(c2["cc_outcome"])
        self.assertEqual(c2["joint"]["status"], "not_run")
        for condition in ("C2", "C3"):
            fluent = self.evaluate(GOLD, condition=condition, side="fluent")
            self.assertEqual(fluent["cc_outcome"], "false_positive")
            self.assertEqual(fluent["members"]["status"], "pass")

    def test_invalid_counts_only_in_end_to_end_denominator(self):
        valid = self.evaluate(GOLD)
        invalid = self.evaluate(None, status="invalid")
        summary, = summarise([valid, invalid])
        self.assertEqual(summary["false_negative"]["denominator"], 1)
        self.assertEqual(summary["members"]["denominator"], 2)
        self.assertEqual(summary["joint"]["rate"], 0.5)

    def test_ambiguous_member_goes_to_review(self):
        duplicate = GOLD + " coat.n.01"
        row = self.evaluate(duplicate)
        self.assertEqual(row["members"]["status"], "review")
        self.assertEqual(row["joint"]["status"], "review")

    def test_wrong_later_connection_is_failure_not_review(self):
        row = self.evaluate(GOLD.replace("AttributeOf", "ColourOf"))
        self.assertEqual(row["members"]["status"], "fail")
        self.assertEqual(row["joint"]["status"], "fail")
        # Without a unique identified group, don't invent per-member evidence.
        self.assertEqual(row["member_recovered"], [None, None])

    def test_joint_failure_cannot_be_overridden_by_review(self):
        failed = (Result("fail", "missing repair"), [])
        review = (Result("review", "ambiguous members"), [])
        self.assertEqual(joint(failed, review).status, "fail")
        self.assertEqual(joint(review, failed).status, "fail")
