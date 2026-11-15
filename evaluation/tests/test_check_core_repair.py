"""Core C3: distinguish C–C detection from exact withdrawal matching."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_core_repair import check_rows, same_withdrawal, summarise, withdrawal_graph
from formal_graph import Graph

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
            ("forwarding", GOLD, "valid", "forwarding"),
        ]
        gold = [dict(example_id=k, item_id=k, variant="rsr_base", repair_type=kind,
                     sbn_repair=GOLD, reparandum_synset="coffee.n.01", repair_synset="tea.n.01")
                for k, _, _, kind in examples]
        basic = [dict(example_id=k, condition="C3", pred_status=status, pred_sbn=sbn)
                 for k, sbn, status, _ in examples]
        items = list(check_rows(gold, basic))
        self.assertEqual([r["withdrawal"]["status"] for r in items],
                         ["pass", "fail", "fail", "fail", "incomplete", "pass"])
        self.assertEqual([r["cc_present"] for r in items], [True, True, True, False, None, True])
        report = summarise(items, scored=True)
        self.assertEqual(report["cc_detection"]["rate"], 4 / 5)
        self.assertEqual(report["withdrawal_match_among_detections"]["rate"], 2 / 4)
        self.assertEqual(report["withdrawal_match_end_to_end"]["rate"], 2 / 6)
        with self.assertRaisesRegex(ValueError, "missing 1"):
            list(check_rows(gold, basic[:-1]))

    def test_graph_comparison(self):
        def view(sbn):
            graph = Graph(sbn)
            return withdrawal_graph(graph, next(graph.pairs()))

        with_lunch = GOLD + " lunch.n.01 TimeOf -3"
        renumbered = ("order.v.01 Agent +1 female.n.02 CORRECTION <1 coffee.n.01 ThemeOf -2 "
                      "CONJUNCTION <2 tea.n.01 ThemeOf -3")
        lunch_withdrawn = GOLD.replace(" CONJUNCTION", " lunch.n.01 TimeOf -2 CONJUNCTION").replace(
            "tea.n.01 ThemeOf -2", "tea.n.01 ThemeOf -3")
        self.assertTrue(same_withdrawal(view(with_lunch), view(renumbered)))
        self.assertFalse(same_withdrawal(view(with_lunch), view(lunch_withdrawn)))
        self.assertFalse(same_withdrawal(view(GOLD), view(GOLD.replace("order.v.01", "serve.v.01"))))
        self.assertFalse(same_withdrawal(view(GOLD), view(GOLD.replace("coffee.n.01 ThemeOf -1",
                                                                      "coffee.n.01"))))

        cases = (
            ("order.v.01 CORRECTION <1 time.n.08 DayOfWeek monday TimeOf -1 "
             "CONJUNCTION <2 time.n.08 DayOfWeek tuesday TimeOf -2", "monday", "friday"),
            ("entity.n.01 CORRECTION <1 male.n.02 Name John EQU -1 "
             "CONJUNCTION <2 male.n.02 Name Mark EQU -2", "John", "James"),
            ("go.v.01 country.n.02 Name UK CORRECTION <1 entity.n.01 EQU -2 Source -1 "
             "CONJUNCTION <2 entity.n.01 EQU -3 Destination -2", "Source -1", "Destination -1"),
        )
        for gold, old, new in cases:
            with self.subTest(changed=old):
                self.assertFalse(same_withdrawal(view(gold), view(gold.replace(old, new))))
