"""Worked acceptance cards from the Research Log."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from formal_graph import Graph, full_score, make_scorer
from formal_local import Pattern, Reference, check_reference, check_repair, joint, value
from score_formal import evaluate, summarise

ORDER = ("female.n.02 time.n.08 TPR now order.v.01 Agent -2 Time -1 "
         "CORRECTION <1 tea.n.01 ThemeOf -1 CONJUNCTION <2 coffee.n.01 ThemeOf -2")


def pattern(event, left, right, roles, values=(), scope=()):
    return Pattern(dict(e=event, w=left, a=right),
                   (("e", "parent"), ("w", "withdrawn"), ("a", "accepted")), scope,
                   (("e", roles[0], "w", "withdrawn"),
                    ("e", roles[1], "a", "accepted")) + values)


class AcceptanceCards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = make_scorer()

    def test_hard_negative(self):
        negation = "female.n.02 NEGATION <1 nurse.n.01 EQU -1 CONTINUATION <2 doctor.n.01 EQU -2"
        omitted = "female.n.02 doctor.n.01 EQU -1"
        false = negation.replace("NEGATION", "CORRECTION").replace("CONTINUATION", "CONJUNCTION")
        targets = [{"example_id": str(i), "sbn_repair": negation} for i in range(3)]
        preds = [{"example_id": str(i), "condition": "C3", "pred_sbn": sbn}
                 for i, sbn in enumerate(("female.n.02 EQU -10", omitted, false))]
        rows = list(evaluate({"hard_negative": targets}, preds, ("C3",)))
        summary, = summarise(rows)
        self.assertEqual(summary["validity"]["rate"], 2 / 3)
        self.assertEqual(summary["false_repair"]["rate"], 1 / 2)
        self.assertEqual(rows[0]["f1"], 0)
        self.assertIsNone(rows[0]["relation_recall"])
        for row in rows[1:]:
            self.assertLess(row["f1"], 1)
            self.assertLess(row["relation_recall"], 1)

    def test_parallel_roles_survive_full_scoring(self):
        text = "person.n.01 see.v.01 Agent -1 Theme -1"
        gold = Graph(text)
        pred = Graph(text.replace("Agent -1 ", ""))
        self.assertLess(full_score(self.scorer, gold, pred)["f1"], 1)

    def test_gold_and_prediction_errors_are_separate(self):
        rows = [{"example_id": "x", "sbn_repair": "female.n.02 EQU -10"},
                {"example_id": "y", "sbn_repair": "female.n.02"}]
        pred = [{"example_id": "x", "condition": "C3", "pred_sbn": "female.n.02 EQU -10"}]
        x, y = evaluate({"hard_negative": rows}, pred, ("C3",))
        self.assertEqual(x["checks"]["validity"]["status"], "fail")
        self.assertEqual(x["checks"]["full"]["status"], "review")
        self.assertTrue(x["checks"]["full"]["reason"].startswith("gold error:"))
        self.assertIsNone(x["f1"])
        self.assertEqual(y["checks"]["validity"]["status"], "not_run")
        self.assertEqual(y["f1"], 0)

    def test_order(self):
        p = pattern("order.v.01", "tea.n.01", "coffee.n.01", ("Theme", "Theme"))
        moved = ("order.v.01 Agent +1 Time +2 female.n.02 time.n.08 TPR now "
                 "CORRECTION <1 tea.n.01 ThemeOf -3 CONJUNCTION <2 coffee.n.01 ThemeOf -4")
        forward_accepted = ORDER.replace("Time -1", "Time -1 Theme +2").replace(
            "coffee.n.01 ThemeOf -2", "coffee.n.01")
        for sbn, reason in (
            (moved, ""),
            (forward_accepted, ""),
            (ORDER + " person.n.01 EQU speaker mean.v.01 Agent -1", ""),
            ("female.n.02 CORRECTION <1 tea.n.01 CONJUNCTION <2 coffee.n.01",
             "required repair object absent"),
            (ORDER.replace("tea.n.01 ThemeOf -1", "tea.n.01"),
             "required relation or value absent in its expected scope"),
        ):
            self.assertEqual(check_repair(Graph(sbn), p)[0].reason, reason)
        gold = Graph(ORDER)
        self.assertEqual(full_score(self.scorer, gold, Graph(moved))["f1"], 1)
        self.assertEqual(full_score(self.scorer, gold, Graph(forward_accepted))["f1"], 1)
        self.assertLess(full_score(self.scorer, gold, Graph(ORDER + " mean.v.01"))["f1"], 1)
        scoped = ORDER.replace("order.v.01", "NEGATION <1 order.v.01")
        p_neg = pattern("order.v.01", "tea.n.01", "coffee.n.01", ("Theme", "Theme"),
                        scope=(("NEGATION", None, None),))
        self.assertEqual(check_repair(Graph(scoped), p_neg)[0].status, "pass")
        self.assertEqual(check_repair(gold, p_neg)[0].status, "fail")
        # Correct endpoints cannot substitute for the required condition scope.
        misplaced = ORDER.replace("Time -1", "Time -1 Theme +1").replace("tea.n.01 ThemeOf -1", "tea.n.01")
        self.assertEqual(check_repair(Graph(misplaced), p)[0].status, "fail")

    def test_values_and_roles(self):
        days = ("class.n.01 time.n.08 EQU now be.v.01 Theme -2 Time -1 "
                "CORRECTION <1 time.n.08 DayOfWeek monday TimeOf -1 "
                "CONJUNCTION <2 time.n.08 DayOfWeek tuesday TimeOf -2")
        p = pattern("be.v.01", "time.n.08", "time.n.08", ("Time", "Time"),
                    (("w", "DayOfWeek", value("monday"), "withdrawn"),
                     ("a", "DayOfWeek", value("tuesday"), "accepted")))
        self.assertEqual(check_repair(Graph(days), p)[0].status, "pass")
        self.assertEqual(check_repair(Graph(days.replace("monday", "tuesday")), p)[0].status, "fail")
        running = ("person.n.01 EQU speaker time.n.08 TPR now run.v.01 Theme -2 Time -1 school.n.02 "
                   "CORRECTION <1 entity.n.01 EQU -2 Destination -1 "
                   "CONJUNCTION <2 entity.n.01 EQU -3 Source -2")
        p = Pattern(dict(e="run.v.01", s="school.n.02", w="entity.n.01", a="entity.n.01"),
                    (("e", "parent"), ("s", "parent"), ("w", "withdrawn"), ("a", "accepted")), (),
                    (("w", "EQU", "e", "withdrawn"), ("a", "EQU", "e", "accepted"),
                     ("w", "Destination", "s", "withdrawn"), ("a", "Source", "s", "accepted")))
        self.assertEqual(check_repair(Graph(running), p)[0].status, "pass")
        swapped = running.replace("Destination", "@").replace("Source", "Destination").replace("@", "Source")
        self.assertEqual(check_repair(Graph(swapped), p)[0].status, "fail")

    def test_later_reference(self):
        p = pattern("borrow.v.01", "coat.n.01", "jacket.n.01", ("Theme", "Theme"))
        spec = Reference("entity.n.01", {"w": ("coat.n.01", ()), "a": ("jacket.n.01", ())},
                         (True, "Attribute", "warm.a.01"))
        good = ORDER.replace("order.v.01", "borrow.v.01").replace("tea.n.01", "coat.n.01").replace(
            "coffee.n.01", "jacket.n.01") + " CONTINUATION <1 entity.n.01 Sub -2 Sub -1 Quantity 2 warm.a.01 AttributeOf -1"
        both_asserted = good.replace("CORRECTION <1 ", "").replace("CONJUNCTION <2 ", "")
        extra = good.replace("entity.n.01 Sub -2", "entity.n.01 Sub -3 Sub -2")
        for sbn, members, both in ((good, "pass", "pass"), (both_asserted, "pass", "fail"),
                                  (good.replace("Sub -2 ", ""), "fail", "fail"),
                                  (extra, "fail", "fail")):
            g = Graph(sbn)
            m = check_reference(g, spec)
            self.assertEqual(m[0].status, members)
            self.assertEqual(joint(check_repair(g, p), m).status, both)


if __name__ == "__main__":
    unittest.main()
