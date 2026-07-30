"""
This script is used to generate small-scale DAGs from SBN sequences
"""

from sbn_smatch import SBNGraph

# Manually paste the SBN sequence here, or later upgrade to argparse
sbn = "female.n.02  CORRECTION <1  time.n.08 TSU now  go.v.01 Theme -2 Time -1                   CONJUNCTION <2                       time.n.08 TPR now                          go.v.01 Theme -4 Time -1 Destination +1    church.n.01                                "   

try: 
    g = SBNGraph().from_string(sbn, is_single_line=True)
    g.to_png("output.png")
    print("Printing successfully, check output.png")
except Exception as e:
    print(f"Parsing wrong: {e}")