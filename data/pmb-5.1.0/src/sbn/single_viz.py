"""
This script is used to generate small-scale DAGs from SBN sequences
"""

from sbn_smatch import SBNGraph

# Manually paste the SBN sequence here, or later upgrade to argparse
sbn = "person.n.01 Name \"Joe\" time.n.01 TPR now buy.v.01 Agent -2 Time -1 Theme +1 Manner +3 CORRECTION <1  car.n.01 Name \"Ferrari\" CONJUNCTION <2 car.n.01 Name \"Jaguar\" ThemeOf -2  gift.n.01 PartOf \"birthday\" MannerOf -3  CORRECTION <1  male.n.01 EQU -6  buy.v.01 Agent -1 Theme -3 Manner +1 gift.n.01 PartOf \"wedding\" "

try: 
    g = SBNGraph().from_string(sbn, is_single_line=True)
    g.to_png("output.png")
    print("Printing successfully, check output.png")
except Exception as e:
    print(f"Parsing wrong: {e}")