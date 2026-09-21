"""
This script merges synthesised repair data to the TSV dataset
"""

import pandas as pd
import json
import csv

def load_jsonl_to_df(filepath, target_col_name):
    """
    Read jsonl file, and extract id and repair_nl
    rename repair_nl to column name
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f: 
            if line.strip():
                item = json.loads(line)
                # Only keep the useful fields to save ram 
                data.append(
                    {
                        "id": item["id"],
                        target_col_name: item["repair_nl"]
                    }
                )
    return pd.DataFrame(data)

def main(): 
    # Read the TSV file 
    tsv_path = "colloquium_prep/pmb_pilot_data.tsv"
    df_tsv = pd.read_csv(tsv_path, sep="\t", dtype=str) # Force all the data to be string to avoid format issue

    # Uniform primary key name
    id_col = "<built-in function id>"
    if id_col in df_tsv.columns:
        df_tsv = df_tsv.rename(columns={id_col: "id"})
    
    # Delete old empty columns 
    cols_to_drop = ["repair_head_nl", "repair_mid_nl", "repair_tail_nl", "repair_interrug_nl"]
    df_tsv = df_tsv.drop(columns=cols_to_drop, errors='ignore')
    
    # Read the repair synthesis data in jsonl 
    df_head = load_jsonl_to_df("colloquium_prep/head.jsonl", "repair_head_nl")
    df_mid = load_jsonl_to_df("colloquium_prep/mid.jsonl", "repair_mid_nl")
    df_tail = load_jsonl_to_df("colloquium_prep/tail.jsonl", "repair_tail_nl")

    # Do left merge one by one 
    df_merged = df_tsv.merge(df_head, on='id', how='left')
    df_merged = df_merged.merge(df_mid, on='id', how='left')
    df_merged = df_merged.merge(df_tail, on='id', how='left')

    # Replace NaN with empty string 
    df_merged = df_merged.fillna('')

    # Output as new TSV 
    output_path = "3_repair_pmb.tsv"
    df_merged.to_csv(
        output_path, 
        sep='\t', 
        index=False, 
        quoting=csv.QUOTE_NONE, 
        escapechar='\\'        
    )
    print(f"data inserted, saved to {output_path}")

if __name__== "__main__":
    main()