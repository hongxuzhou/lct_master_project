"""
Based on the synthesised middle repair samples, insert the interregnum "I mean," between the reparandum and the repair.
"""

import pandas as pd
import difflib
import csv

def insert_interregnum(original, repaired, interregnum="I mean,"):
    """
    Compare the original natural language sampels vs. the middle-repair samples. For each reparandum, immediately insert an interregnum "I mean," after it. 
    """
    # Fault tolerance: if a sample does not have a reparandum-repair structure, return empty string
    if pd.isna(original) or pd.isna(repaired) or not str(repaired).strip():
        return ""
    
    tokens_orig = str(original).split()
    tokens_rep = str(repaired).split()
    
    matcher = difflib.SequenceMatcher(None, tokens_orig, tokens_rep)
    result_tokens = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            result_tokens.extend(tokens_rep[j1:j2])
        elif tag == 'insert':
            reparandum_tokens = tokens_rep[j1:j2]
            
            # Find the token with a comma in the insert chunk
            comma_idx = -1
            for idx, token in enumerate(reparandum_tokens):
                if ',' in token:
                    comma_idx = idx
                    break
        
            # Once find the comma, insert the "I mean," after the token
            if comma_idx != -1:
                result_tokens.extend(reparandum_tokens[:comma_idx + 1])
                result_tokens.append(interregnum)
                result_tokens.extend(reparandum_tokens[comma_idx + 1:])
            else:
                # If cannot find a comma, attach it to the end. 
                result_tokens.extend(reparandum_tokens)
                result_tokens.append(interregnum)
        elif tag == 'replace':
            # For unexpected replacement, keep the original and warning
            # print(f"Warning: Replace found '{tokens_orig[i1:i2]}' -> '{tokens_rep[j1:j2]}'")
            result_tokens.extend(tokens_rep[j1:j2])
        elif tag == 'delete':
            pass # ignore deletion
            
    return " ".join(result_tokens)

def main():
    # 1. Read dataset 
    input_path = "/Users/hongxuzhou/Documents/GitHub/lct_master_project/3_repair_pmb.tsv"
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, sep='\t', dtype=str)
    
    # For possible NaN, convert to empty string 
    df = df.fillna('')
    

    # Generate new column "repair_interrug_nl"
    # Use apply function
    print("Processing sequence alignment and inserting interregnum...")
    df['repair_interrug_nl'] = df.apply(
        lambda row: insert_interregnum(row['nl'], row['repair_mid_nl']), 
        axis=1
    )
    
    # 3. Output as new TSV dataset 
    output_path = "/Users/hongxuzhou/Documents/GitHub/lct_master_project/4_repair_interrug_pmb.tsv"
    df.to_csv(
        output_path, 
        sep='\t', 
        index=False, 
        quoting=csv.QUOTE_NONE, 
        escapechar='\\'
    )
    print(f"Processing complete! Data saved to {output_path}")

if __name__ == "__main__":
    main()