"""Boundary cases encountered during the first development machine review."""
import unittest

from repair_data_gen.shared_pipeline.analyse_split import exact_surface, pair_key, quote_diagnostic


class MachineAnalysisTest(unittest.TestCase):
    def test_tense_can_point_outside_repair_box(self):
        row = dict(repair_type="tense", site_index="2", reparandum_value="TPR now",
                   sbn_clean="person.n.01 EQU speaker time.n.08 TSU now wash.v.03 Agent -2 Time -1 Source +1 dish.n.01",
                   sbn_repair="person.n.01 EQU speaker time.n.08 TSU now CORRECTION <1 time.n.08 TPR now wash.v.03 Agent -3 Time -1 CONJUNCTION <2 wash.v.03 Agent -4 Time -3 Source +1 dish.n.01")
        self.assertEqual(pair_key(row)[2:], ("TPR now", "TSU now"))

    def test_tense_ignores_unchanged_second_time_edge(self):
        row = dict(repair_type="tense", site_index="3", reparandum_value="EQU now",
                   sbn_clean="person.n.01 EQU speaker time.n.08 TPR now tooth.n.01 pull.v.15 Source -3 Time -2 Theme -1 Time +1 time.n.08 TPR now",
                   sbn_repair="person.n.01 EQU speaker time.n.08 TPR now tooth.n.01 CORRECTION <1 pull.v.15 Source -3 Time -2 Theme -1 Time +1 time.n.08 EQU now CONJUNCTION <2 pull.v.15 Source -5 Time -4 Theme -3 Time +1 time.n.08 TPR now")
        self.assertEqual(pair_key(row)[2:], ("EQU now", "TPR now"))

    def test_quote_diagnostic_is_read_only_and_does_not_rescue_direction_errors(self):
        row = dict(repair_type="constant_name", nl_clean='The movie "Ole Yeller" was made.',
                   nl_repair="The movie 'Bambi', the movie 'Ole Yeller' was made.",
                   repair_onset=str(len("The movie 'Bambi', ")), reparandum_surface="Bambi",
                   interregnum="none",
                   sbn_repair='entity.n.01 CORRECTION <1 movie.n.01 Name "Bambi" EQU -1 CONJUNCTION <2 movie.n.01 Name "Ole Yeller" EQU -2')
        original = dict(row)
        self.assertTrue(quote_diagnostic(row))
        self.assertEqual(row, original)
        row["nl_clean"] = 'The movie "Bambi" was made.'
        self.assertFalse(quote_diagnostic(row))

    def test_exact_surface_does_not_hide_added_punctuation(self):
        row = dict(nl_clean="A man is walking", nl_repair="A boy, a man is walking", repair_onset="7")
        self.assertTrue(exact_surface(row))
        row["nl_repair"] += "."
        self.assertFalse(exact_surface(row))


if __name__ == "__main__":
    unittest.main()
