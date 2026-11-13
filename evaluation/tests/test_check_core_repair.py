"""Core C3: distinguish detection from local correctness and its coverage."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_core_repair import check_rows, summarise

GOLD = ("female.n.02 order.v.01 Agent -1 CORRECTION <1 coffee.n.01 ThemeOf -1 "
        "CONJUNCTION <2 tea.n.01 ThemeOf -2")
WITHDRAWN_SUBJECT = (
    "order.v.01 CORRECTION <1 female.n.02 AgentOf -1 coffee.n.01 ThemeOf -2 "
    "CONJUNCTION <2 tea.n.01 ThemeOf -3"
)


class CoreRepairTests(unittest.TestCase):
    def test_scope_detection_and_denominators(self):
        examples = [
            ("correct", GOLD, "valid", "content_word"),
            ("wrong_scope", WITHDRAWN_SUBJECT, "valid", "content_word"),
            ("reversed", GOLD.replace("coffee", "@").replace("tea", "coffee").replace("@", "tea"),
             "valid", "content_word"),
            ("no_pair", "female.n.02 order.v.01 Agent -1 Theme +1 tea.n.01", "valid", "content_word"),
            ("invalid", "", "invalid", "content_word"),
            ("review", GOLD, "valid", "forwarding"),
        ]
        gold = [dict(example_id=k, item_id=k, variant="rsr_base", repair_type=kind,
                     sbn_repair=GOLD, reparandum_synset="coffee.n.01", repair_synset="tea.n.01")
                for k, _, _, kind in examples]
        basic = [dict(example_id=k, condition="C3", pred_status=status, pred_sbn=sbn)
                 for k, sbn, status, _ in examples]
        items = list(check_rows(gold, basic))
        self.assertEqual([r["repair"]["status"] for r in items],
                         ["pass", "fail", "fail", "fail", "incomplete", "review"])
        self.assertIn("scope", items[1]["repair"]["reason"])
        self.assertEqual([r["cc_present"] for r in items], [True, True, True, False, None, True])
        report = summarise(items, scored=True)
        self.assertEqual(report["cc_detection"]["rate"], 4 / 5)
        self.assertEqual(report["correct_among_detections"]["rate"], 1 / 3)
        self.assertEqual(report["end_to_end"]["rate"], 1 / 5)
        with self.assertRaisesRegex(ValueError, "missing 1"):
            list(check_rows(gold, basic[:-1]))
