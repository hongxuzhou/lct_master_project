"""
This script is used to generate small-scale DAGs from SBN sequences
"""

from sbn_smatch import SBNGraph

# Manually paste the SBN sequence here, or later upgrade to argparse
sbn = "person.n.01 EQU speaker time.n.01 TPR now NEGATION <1 order.v.01 Agent -2 Time -1 Theme +1 CORRECTION <1 banana_bread.n.01 CONJUNCTION <2 cherry_pie.n.01 ThemeOf -2"

try: 
    g = SBNGraph().from_string(sbn, is_single_line=True)
    g.to_png("output.png")
    print("Printing successfully, check output.png")
except Exception as e:
    print(f"Parsing wrong: {e}")