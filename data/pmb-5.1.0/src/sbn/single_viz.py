"""
This script is used to generate small-scale DAGs from SBN sequences
"""

from sbn_smatch import SBNGraph

# Manually paste the SBN sequence here, or later upgrade to argparse
sbn = "NEGATION <1 farmer.n.01 own.v.01 Pivot -1 Theme +1 donkey.n.01 NEGATION <1 CORRECTION <1 beat.v.01 Agent -3 Theme +2 CONJUNCTION <2 feed.v.01 Agent -4 Theme +1 entity.n.01 EQU -3"   

try: 
    g = SBNGraph().from_string(sbn, is_single_line=True)
    g.to_png("output.png")
    print("Printing successfully, check output.png")
except Exception as e:
    print(f"Parsing wrong: {e}")