"""Check that the PMB readers attach edges to the intended nodes."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbn_env import ensure_on_path

ensure_on_path()

from sbn2penman import SBNGraph as PenmanReader
from sbn_smatch import SBNGraph
from sbn_smatch_fine_grained import SBNGraph as FineGrainedReader
from sbn_spec import SBN_NODE_TYPE, SBNError


class IndexTests(unittest.TestCase):
    def test_full_concept_offsets_in_all_readers(self):
        # Four Co-Theme +11 edges in PMB p23/d1469 used to hit the next copula.
        tokens = ["be.v.01 Co-Theme +11", "time.n.08"] + ["entity.n.01"] * 9
        tokens += ["noble_gas.n.01 ANA -11 Time -1"]
        for reader in (SBNGraph, FineGrainedReader, PenmanReader):
            with self.subTest(reader=reader.__module__):
                # Use separate documents: combining the opposite edges creates a cycle.
                g = reader().from_string("\n".join([*tokens[:-1], "noble_gas.n.01"]))
                self.assertEqual(g.edges[(SBN_NODE_TYPE.SYNSET, 0),
                                         (SBN_NODE_TYPE.SYNSET, 11)]["token"], "Co-Theme")
                g = reader().from_string("\n".join(["be.v.01", *tokens[1:]]))
                for role, target in (("ANA", 0), ("Time", 10)):
                    self.assertEqual(g.edges[(SBN_NODE_TYPE.SYNSET, 11),
                                             (SBN_NODE_TYPE.SYNSET, target)]["token"], role)
                self.assertTrue(g.is_dag)

    def test_constant_prefix_and_out_of_range_index(self):
        g = SBNGraph().from_string("entity.n.01\nentity.n.01 EQU -1abc ANA -12")
        targets = {edge["token"]: (node[0], g.nodes[node]["token"])
                   for _, node, edge in g.edges(data=True) if edge["token"] in {"EQU", "ANA"}}
        self.assertEqual(targets, {"EQU": (SBN_NODE_TYPE.CONSTANT, "-1abc"),
                                   "ANA": (SBN_NODE_TYPE.CONSTANT, "-12")})
        self.assertTrue(g.is_possibly_ill_formed)

    def test_box_offsets(self):
        sbn = "entity.n.01 Proposition >10\n" + "NEGATION <1\nentity.n.01\n" * 9
        sbn += "CONTRAST <10\nentity.n.01"
        for reader in (SBNGraph, FineGrainedReader):
            with self.subTest(reader=reader.__module__):
                g = reader().from_string(sbn)
                final_box = (SBN_NODE_TYPE.BOX, 10)
                self.assertEqual(g.edges[(SBN_NODE_TYPE.BOX, 0), final_box]["token"], "CONTRAST")
                self.assertEqual(g.edges[(SBN_NODE_TYPE.SYNSET, 0), final_box]["token"], "Proposition")
                with self.assertRaisesRegex(SBNError, "Invalid box index"):
                    reader().from_string("entity.n.01\nNEGATION <1abc\nentity.n.01")


if __name__ == "__main__":
    unittest.main()
