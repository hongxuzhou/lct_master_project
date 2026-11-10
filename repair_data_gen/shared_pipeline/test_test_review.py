"""Do not damage valid surface text when legacy gold alignment fails."""
import unittest

from repair_data_gen.shared_pipeline.analyse_test_review import normalise_candidate


class TestSurfaceFallback(unittest.TestCase):
    def test_word_final_apostrophe_is_not_a_quote(self):
        text = "Fuckin' tremor, earthquake was terrible!"
        row = dict(request_id="test-apostrophe", repair_type="content_word",
                   nl_clean="Fuckin' earthquake was terrible!", nl_repair=text,
                   repair_onset=str(text.index("earthquake")), sbn_repair="unchanged")
        result, aligned = normalise_candidate(row)
        self.assertFalse(aligned)
        self.assertEqual(result["nl_repair"], text)
        self.assertEqual(result["repair_onset"], row["repair_onset"])

    def test_missing_citation_does_not_remove_final_period(self):
        text = "She died, he died in 2014."
        row = dict(request_id="test-citation", repair_type="forwarding",
                   nl_clean="He died in 2014. [4]", nl_repair=text,
                   repair_onset=str(text.index(", ") + 2), sbn_repair="unchanged")
        result, aligned = normalise_candidate(row)
        self.assertFalse(aligned)
        self.assertEqual(result["nl_repair"], text)
        self.assertEqual(result["sbn_repair"], row["sbn_repair"])


if __name__ == "__main__":
    unittest.main()
