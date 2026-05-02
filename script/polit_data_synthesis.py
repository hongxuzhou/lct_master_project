"""
generate_repair.py

Synthesise self-repair variants from PMB gold data using Qwen3-32B-Instruct.

Usage examples
--------------
# Head repair (reads nl column from TSV)
python generate_repair.py \
    --prompt_type head \
    --input_tsv data.tsv \
    --output_jsonl output/head.jsonl \
    --model_path /path/to/Qwen3-32B-Instruct \
    --checkpoint output/head.ckpt

# Interregnum (reads repair_nl from mid JSONL)
python generate_repair.py \
    --prompt_type interregnum \
    --interregnum_input output/mid.jsonl \
    --output_jsonl output/interregnum.jsonl \
    --model_path /path/to/Qwen3-32B-Instruct \
    --checkpoint output/interregnum.ckpt

Output JSONL schema (one record per line)
-----------------------------------------
{"id": "p17/d0758", "original_nl": "...", "repair_nl": "..."}
"""

import argparse
import csv
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPTS = {
    "head": """\
You are an expert computational linguist and text editor. Your task is to modify clean English sentences to include a specific type of self-repair structure for an NLP parser evaluation experiment.

[DEFINITIONS]
- Self-repair: A phenomenon where the speaker extracts part of the information and replaces it with a new piece of information.
- Reparandum: The retracted/abandoned information.
- Repair: The newly added, correct information.
Example: In "I need a train, a flight, to Boston", the reparandum is "a train", and the repair is "a flight".

[TASK]
Thoroughly comprehend the input sentence. Inject a self-repair structure exactly at the beginning of the sentence. 
The injected structure must consist of a reparandum immediately followed by the repair, separated ONLY by a comma.

[STRICT RULES: DO's]
1. SEMANTIC ANCHOR: The *repair* MUST be the original information from the input sentence. The *reparandum* must be the newly invented (incorrect) information. This ensures the final semantic meaning of the sentence remains identical to the input.
   - Input: "Blueberry is my favourite milkshake flavour."
   - Correct Output: "Strawberry, blueberry is my favourite milkshake flavour." (reparandum: strawberry, repair: blueberry)
   - Wrong Output: "Blueberry, strawberry is my favourite milkshake..." (Changes the core meaning).
2. INFORMATION DENSITY: The reparandum and repair must contain roughly the same amount of information and belong to the same syntactic category.
   - Correct: "When travelling in Rome, Seville, Joe has the best orange juice he ever had."
   - Wrong: "When travelling in Rome with his husband as part of their honeymoon vacation, Seville, Joe has the best orange juice he ever had." (Reparandum is too heavy).

[STRICT RULES: DON'Ts]
1. NO INTERREGNUMS: You must NEVER add metalinguistic expressions (e.g., "actually", "I mean", "no", "hold on", "uh", "um") between the reparandum and the repair. It must be a direct replacement.
   - Correct: "Ali, Joe bought a bike."
   - Wrong: "Ali, no, Joe bought a bike."
   - Correct: "Banana, apple is good for your health"
   - Wrong: "Banana, I mean, apple is good for your health"
2. NO TENSE/ASPECT CHANGES: Never adjust the grammar, tense, or aspect of the original input.
   - Correct: "Here, there are many roses."
   - Wrong: "There were, are many roses."
   - Correct: "Standing, sitting on the bench, John ate an apple."
   - Wrong: "Stood, sitting on the bench, John ate an apple."

[OUTPUT FORMAT]
You must output ONLY the modified sentence. Do not include any explanations, greetings, or quotation marks. 

Input Sentence: "{sentence}"
Output:""",

    "mid": """\
You are an expert computational linguist and text editor. Your task is to modify clean English sentences to include a specific type of self-repair structure for an NLP parser evaluation experiment.

[DEFINITIONS]
- Self-repair: A phenomenon where the speaker extracts part of the information and replaces it with a new piece of information.
- Reparandum: The retracted/abandoned information.
- Repair: The newly added, correct information.
Example: In "I need a train, a flight, to Boston", the reparandum is "a train", and the repair is "a flight".

[TASK]
Thoroughly comprehend the input sentence. Inject a self-repair structure exactly at the middle of the sentence. 
The injected structure must consist of a reparandum immediately followed by the repair, separated ONLY by a comma.

[STRICT RULES: DO's]
1. SEMANTIC ANCHOR: The *repair* MUST be the original information from the input sentence. The *reparandum* must be the newly invented (incorrect) information. This ensures the final semantic meaning of the sentence remains identical to the input.
   - Input: "I bought four apples from Aldi yesterday."
   - Correct Output: "I bought three, four apples from Aldi yesterday." (reparandum: three, repair: four)
   - Wrong Output: "I bought four, three apples from Aldi yesterday." (Changes the core meaning).
2. INFORMATION DENSITY: The reparandum and repair must contain roughly the same amount of information and belong to the same syntactic category.
   - Correct: "When travelling in Seville, Joe has the best apple, orange juice he ever had."
   - Wrong: "When travelling in Seville, Joe has the best organic and environment-friendly apple, orange juice he ever had." (Reparandum is too heavy).

[STRICT RULES: DON'Ts]
1. NO INTERREGNUMS: You must NEVER add metalinguistic expressions (e.g., "actually", "I mean", "no", "hold on", "uh", "um") between the reparandum and the repair. It must be a direct replacement.
   - Correct: "Ali bought a bike, a scooter for his son."
   - Wrong: "Ali, bought a bike, no, a scooter for his son."
   - Correct: "Banana is bad, good for your health"
   - Wrong: "Banana is bad, I mean, good for your health"
2. NO TENSE/ASPECT CHANGES: Never adjust the grammar, tense, or aspect of the original input.
   - Correct: "Here are a few, many roses."
   - Wrong: "Here were, are many roses."
   - Correct: "Linda stole, robbed the kiosk"
   - Wrong: "Linda has stolen, stole the kiosk"

[OUTPUT FORMAT]
You must output ONLY the modified sentence. Do not include any explanations, greetings, or quotation marks. 

Input Sentence: "{sentence}"
Output:""",

    "tail": """\
You are an expert computational linguist and text editor. Your task is to modify clean English sentences to include a specific type of self-repair structure for an NLP parser evaluation experiment.

[DEFINITIONS]
- Self-repair: A phenomenon where the speaker extracts part of the information and replaces it with a new piece of information.
- Reparandum: The retracted/abandoned information.
- Repair: The newly added, correct information.
Example: In "I need a train, a flight, to Boston", the reparandum is "a train", and the repair is "a flight".

[TASK]
Thoroughly comprehend the input sentence. Inject a self-repair structure exactly at the ending part of the sentence. 
The injected structure must consist of a reparandum immediately followed by the repair, separated ONLY by a comma.

[STRICT RULES: DO's]
1. SEMANTIC ANCHOR: The *repair* MUST be the original information from the input sentence. The *reparandum* must be the newly invented (incorrect) information. This ensures the final semantic meaning of the sentence remains identical to the input.
   - Input: "I bought four apples from Aldi yesterday."
   - Correct Output: "I bought four apples from Lidl, Aldi yesterday." (reparandum: Lidl, repair: Aldi)
   - Wrong Output: "I bought four  apples from Aldi, Lidl yesterday." (Changes the core meaning).
2. INFORMATION DENSITY: The reparandum and repair must contain roughly the same amount of information and belong to the same syntactic category.
   - Correct: "When travelling in Seville, Joe has the best orange juice she, he ever had."
   - Wrong: "When travelling in Seville, Joe has the best orange juice she and her girlfriend along with their families, he ever had." (Reparandum is too heavy).

[STRICT RULES: DON'Ts]
1. NO INTERREGNUMS: You must NEVER add metalinguistic expressions (e.g., "actually", "I mean", "no", "hold on", "uh", "um") between the reparandum and the repair. It must be a direct replacement.
   - Correct: "Ali bought a bike for his son's birthday, school return day."
   - Wrong: "Ali bought a bike for his son's birthday, actually, school return day."
   - Correct: "Banana is good for your body shape, health."
   - Wrong: "Banana is good for your body shape, I mean, health."
2. NO TENSE/ASPECT CHANGES: Never adjust the grammar, tense, or aspect of the original input.
   - Correct: "The kiosk around the corner was robbed by Mary, Linda."
   - Wrong: "The kiosk around the corner was robbed, is being robbed by Linda."

[OUTPUT FORMAT]
You must output ONLY the modified sentence. Do not include any explanations, greetings, or quotation marks. 

Input Sentence: "{sentence}"
Output:""",

    "interregnum": """\
You are an expert computational linguist and text editor. Your task is to process English sentences that ALREADY contain a self-repair structure by injecting a specific interregnum, for an NLP parser evaluation experiment.

[DEFINITIONS]

    - Self-repair: A structure containing a reparandum (abandoned info) followed by a repair (new info).

    - Interregnum: A metalinguistic editing phrase that signals the speaker is making a correction. For this experiment, the ONLY allowed interregnum is "I mean".

[TASK]
Thoroughly comprehend the input sentence. The input sentence already contains a self-repair exactly at the middle part, separated by a comma (e.g., "... reparandum, repair ...").
Your task is to locate the exact boundary between the reparandum and the repair, and insert the interregnum "I mean" at that boundary.
The final structure MUST be formatted exactly as: [reparandum], I mean, [repair]

[STRICT RULES: DO's]

1. EXACT PRESERVATION: You must keep every single word, grammar, tense, and punctuation mark of the input sentence exactly the same, except for the insertion of ", I mean,". You are NOT allowed to invent new reparandums or repairs.

    - Input: "I bought three, four apples from Aldi yesterday."

    - Correct Output: "I bought three, I mean, four apples from Aldi yesterday."

    - Wrong Output: "I bought three, I mean, five apples from Aldi yesterday." (Changed the repair information).

2. BOUNDARY ACCURACY: Only insert the interregnum at the self-repair boundary. Do not insert it at regular syntactic commas.

    - Input: "When travelling in Seville, Joe has the best apple, orange juice he ever had."

    - Correct Output: "When travelling in Seville, Joe has the best apple, I mean, orange juice he ever had."

    - Wrong Output: "When travelling in Seville, I mean, Joe has the best apple, orange juice he ever had." (Inserted at the wrong comma).

[STRICT RULES: DON'Ts]

1. NO OTHER INTERREGNUMS: You must NEVER use any other metalinguistic expressions (e.g., "actually", "no", "hold on", "uh", "um"). ONLY use "I mean".

    - Correct: "Banana is bad, I mean, good for your health."

    - Wrong: "Banana is bad, uh, good for your health."

[OUTPUT FORMAT]
You must output ONLY the modified sentence. Do not include any explanations, greetings, or quotation marks.


Input Sentence: "{sentence}"
Output:""",
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> set:
    """Return the set of IDs already processed."""
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def save_checkpoint(path: str, sample_id: str) -> None:
    """Append one completed ID to the checkpoint file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(sample_id + "\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tsv_samples(tsv_path: str) -> list:
    """
    Read the PMB TSV and return [{id, nl}, ...].
    Expects tab-separated with a header row containing at least 'id' and 'nl'.
    """
    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            samples.append({"id": row["id"], "nl": row["nl"]})
    return samples


def load_mid_jsonl(jsonl_path: str) -> list:
    """
    Read the mid repair JSONL and return samples for interregnum generation.
    Each output record carries:
      - 'nl'          : the mid repair sentence (becomes the prompt input)
      - 'original_nl' : the original PMB sentence (preserved in output)
    """
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append({
                "id":          record["id"],
                "nl":          record["repair_nl"],
                "original_nl": record["original_nl"],
            })
    return samples


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_prompt(prompt_type: str, sentence: str) -> str:
    return PROMPTS[prompt_type].format(sentence=sentence)


def generate_one(model, tokenizer, prompt_type, sentence,
                 max_new_tokens, temperature, top_p, top_k) -> str:
    messages = [{"role": "user", "content": build_prompt(prompt_type, sentence)}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            #temperature=temperature,
            #top_p=top_p,
            #top_k=top_k,
            do_sample=False, # Greedy decoding
        )

    new_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate self-repair variants from PMB gold data."
    )
    parser.add_argument(
        "--prompt_type",
        required=True,
        choices=["head", "mid", "tail", "interregnum"],
        help="Which repair type to generate.",
    )
    parser.add_argument(
        "--input_tsv",
        default=None,
        help="Path to the PMB TSV file. Required for head / mid / tail.",
    )
    parser.add_argument(
        "--interregnum_input",
        default=None,
        help="Path to the mid repair JSONL. Required for interregnum.",
    )
    parser.add_argument(
        "--output_jsonl",
        required=True,
        help="Path to write output JSONL.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Local path or HF hub name for Qwen3-32B-Instruct.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint file (records completed IDs for resume).",
    )
    parser.add_argument("--max_new_tokens", type=int,   default=128)
    #parser.add_argument("--temperature",    type=float, default=0.7)
    #parser.add_argument("--top_p",          type=float, default=0.8)
    #parser.add_argument("--top_k",          type=int,   default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate argument combinations
    if args.prompt_type in ("head", "mid", "tail"):
        if not args.input_tsv:
            sys.exit("ERROR: --input_tsv is required for head / mid / tail.")
    else:  # interregnum
        if not args.interregnum_input:
            sys.exit("ERROR: --interregnum_input is required for interregnum.")

    # Load samples
    if args.prompt_type == "interregnum":
        samples = load_mid_jsonl(args.interregnum_input)
    else:
        samples = load_tsv_samples(args.input_tsv)

    print(f"Total samples: {len(samples)}", flush=True)

    # Resume: skip already-completed IDs
    done_ids  = load_checkpoint(args.checkpoint)
    remaining = [s for s in samples if s["id"] not in done_ids]
    print(f"Done: {len(done_ids)}  |  Remaining: {len(remaining)}", flush=True)

    if not remaining:
        print("Nothing left to generate. Exiting.")
        return

    # Load model
    print(f"Loading model from {args.model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    print("Model ready.", flush=True)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
    os.makedirs(out_dir, exist_ok=True)

    # Generate and write incrementally
    with open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        for i, sample in enumerate(remaining):
            sample_id   = sample["id"]
            input_sent  = sample["nl"]
            original_nl = sample.get("original_nl", input_sent)

            try:
                repair_nl = generate_one(
                    model, tokenizer,
                    args.prompt_type, input_sent,
                    args.max_new_tokens, #args.temperature,
                    #args.top_p, args.top_k,
                )
            except Exception as e:
                print(f"[{i+1}/{len(remaining)}] ERROR {sample_id}: {e}", flush=True)
                continue

            record = {
                "id":          sample_id,
                "original_nl": original_nl,
                "repair_nl":   repair_nl,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            save_checkpoint(args.checkpoint, sample_id)
            print(
                f"[{i+1}/{len(remaining)}] {sample_id} -> {repr(repair_nl[:80])}",
                flush=True,
            )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()