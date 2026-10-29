import unittest

from repair_data_gen.shared_pipeline.prepare_dev_review import surface_eligible


def row(gold, left, right, kind="tense"):
    return dict(nl_clean=gold, nl_repair=left + ", " + right,
                repair_onset=str(len(left) + 2), repair_type=kind)


class SurfacePolicyTest(unittest.TestCase):
    def test_quote_and_punctuation_changes_are_not_rejections(self):
        self.assertTrue(surface_eligible(row('The movie "Ole Yeller" was made.',
                                            "The movie 'Bambi'", "the movie 'Ole Yeller' was made", "constant_name"))[0])

    def test_contracted_copula_remains_eligible(self):
        self.assertTrue(surface_eligible(row("I'm hungry.", "I was", "am hungry."))[0])

    def test_auxiliary_realisation_can_go_to_human_review(self):
        self.assertTrue(surface_eligible(row("I've caught a cold.", "I catch", "caught a cold."))[0])

    def test_reversed_number_is_rejected(self):
        self.assertFalse(surface_eligible(row("Mix with two eggs.", "Mix with two", "three eggs.", "constant_number"))[0])

    def test_reversed_finite_tense_is_rejected(self):
        self.assertFalse(surface_eligible(row("She was a novelist.", "She was", "is a novelist."))[0])

    def test_negation_cannot_be_dropped(self):
        self.assertFalse(surface_eligible(row("I don't want milk.", "I wanted", "want milk."))[0])

    def test_editorial_citation_is_not_semantic_content(self):
        self.assertTrue(surface_eligible(row("Tom (born July 19) [1] is an actor.",
                                            "Tom (born June", "July 19) is an actor.", "constant_temporal"))[0])


if __name__ == "__main__":
    unittest.main()
