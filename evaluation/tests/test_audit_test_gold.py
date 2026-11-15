"""Gold filtering must consider both targets and preserve comparison pairs."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_test_gold import FILES, audit_rows


class GoldAuditTests(unittest.TestCase):
    def test_bad_clean_target_removes_both_core_partners(self):
        # C3's repair target is usable; C2's clean target is not. Both must
        # leave the shared comparison panel, including the marked partner.
        repair = ("order.v.01 CORRECTION <1 coffee.n.01 ThemeOf -1 "
                  "CONJUNCTION <2 tea.n.01 ThemeOf -2")
        fluent = dict(example_id="f", doc_id="doc", variant="pmb_fluent", text="Tea.",
                      sbn_clean="tea.n.01", sbn_repair="tea.n.01")
        base = dict(example_id="b", item_id="pair", doc_id="doc", variant="rsr_base",
                    repair_type="content_word", text="Coffee, tea.",
                    sbn_clean="tea.n.01 EQU -99", sbn_repair=repair)
        marked = dict(base, example_id="m", variant="rsr_editing_expression",
                      text="Coffee, sorry, tea.", sbn_clean="tea.n.01")
        items, _ = audit_rows({FILES[0]: [fluent, base, marked], FILES[1]: [], FILES[2]: []})
        self.assertEqual([r["status"] for r in items], ["retain", "exclude", "exclude"])
        self.assertTrue(any(i["check"] == "sbn_clean" for i in items[1]["issues"]))

    def test_cycle_is_excluded_and_does_not_remove_unrelated_rows(self):
        cycle = "entity.n.01 SubOf +1 entity.n.01 ANA -1"
        rows = [dict(example_id="cycle", doc_id="bad", variant="pmb_fluent", text="Some.",
                     sbn_clean=cycle, sbn_repair=cycle),
                dict(example_id="good", doc_id="good", variant="pmb_fluent", text="Tea.",
                     sbn_clean="tea.n.01", sbn_repair="tea.n.01")]
        items, _ = audit_rows({FILES[0]: rows, FILES[1]: [], FILES[2]: []})
        self.assertEqual([r["status"] for r in items], ["exclude", "retain"])
        self.assertEqual(items[0]["issues"][0]["reason"], "cyclic graph")

    def test_missing_pair_and_global_duplicate_are_not_silent(self):
        row = dict(example_id="b", item_id="p", doc_id="d", variant="rsr_base",
                   repair_type="content_word", text="Tea.",
                   sbn_clean="tea.n.01", sbn_repair="tea.n.01")
        items, _ = audit_rows({FILES[0]: [row], FILES[1]: [], FILES[2]: []})
        self.assertEqual(items[0]["status"], "exclude")
        self.assertTrue(any(i["check"] == "pair_consistency" for i in items[0]["issues"]))
        with self.assertRaisesRegex(ValueError, "duplicate test ID"):
            audit_rows({FILES[0]: [row, row], FILES[1]: [], FILES[2]: []})


if __name__ == "__main__":
    unittest.main()
