"""
This script cleans the extra escape characters "\" in the dataset.
""" 
import pandas as pd

df = pd.read_csv("your_data_cleaned.tsv", sep="\t", dtype=str)

# convert all the escaped quotation marks to normal ones 
df["mr"] = df["mr"].str.replace(r'\"', '"', regex=False)   # \" → "
df["mr"] = df["mr"].str.replace(r'\\"', '"', regex=False)   # \" → "
df["mr"] = df["mr"].str.replace(r'\\\"', '"', regex=False)  # \" → " multi-layer

# verify Kinshasa 
print(df.loc[df["id"] == "p27/d2907", "mr"].values[0])

df.to_csv("pilot_dataset_cleaned.tsv", sep="\t", index=False)