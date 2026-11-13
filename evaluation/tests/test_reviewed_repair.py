"""One graph matcher for hand-selected requirements across repair types."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from formal_gold import annotated_pattern
from formal_graph import Graph
from formal_local import check_repair


class ReviewedRepairTests(unittest.TestCase):
    def test_optional_lunch_must_not_be_withdrawn(self):
        prefix = "female.n.02 order.v.01 Agent -1 CORRECTION <1 coffee.n.01 ThemeOf -1"
        suffix = " CONJUNCTION <2 tea.n.01 ThemeOf -2"
        gold = prefix + suffix + " lunch.n.01 TimeOf -3"
        pattern = annotated_pattern(Graph(gold), {
            "concept_indices": [0, 1, 2, 3],
            "relations": [[1, "Agent", 0], [1, "Theme", 2], [1, "Theme", 3]],
            "protected_relations": [[1, "Time", 4]],
        })
        self.assertEqual(check_repair(Graph(prefix + suffix), pattern)[0].status, "pass")
        shifted_suffix = suffix.replace("ThemeOf -2", "ThemeOf -3")
        other_event = prefix + " lunch.n.01" + shifted_suffix
        self.assertEqual(check_repair(Graph(other_event), pattern)[0].status, "pass")
        moved = prefix + " lunch.n.01 TimeOf -2" + shifted_suffix
        result, _ = check_repair(Graph(moved), pattern)
        self.assertEqual(result.reason, "protected relation is inside the withdrawn scope")

    def test_same_matcher_for_name_tense_and_role(self):
        cases = [
            ("entity.n.01 CORRECTION <1 male.n.02 Name John EQU -1 "
             "CONJUNCTION <2 male.n.02 Name Mark EQU -2 graduate.v.01 Agent -3",
             [[1, "EQU", 0], [2, "EQU", 0], [3, "Agent", 0]], "Agent -3", "Theme -3"),
            ("person.n.01 CORRECTION <1 be.v.02 Theme -1 Time +1 time.n.08 EQU now "
             "CONJUNCTION <2 be.v.02 Theme -3 Time +1 time.n.08 TPR now",
             [[1, "Theme", 0], [3, "Theme", 0], [1, "Time", 2], [3, "Time", 4]],
             "TPR now", "EQU now"),
            ("go.v.01 country.n.02 Name UK CORRECTION <1 entity.n.01 EQU -2 Source -1 "
             "CONJUNCTION <2 entity.n.01 EQU -3 Destination -2",
             [[2, "EQU", 0], [3, "EQU", 0], [2, "Source", 1], [3, "Destination", 1]],
             "Destination -2", "Source -2"),
        ]
        for gold, edges, old, new in cases:
            with self.subTest(changed=old):
                pattern = annotated_pattern(Graph(gold), {
                    "concept_indices": [1], "relations": edges,
                })
                self.assertEqual(check_repair(Graph(gold), pattern)[0].status, "pass")
                self.assertEqual(check_repair(Graph(gold.replace(old, new)), pattern)[0].status, "fail")
