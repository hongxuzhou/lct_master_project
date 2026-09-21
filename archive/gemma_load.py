"""
This script loads models for my master's study to the common dir in the LST HPC at UdS.

The models include Gemma 2 9b instruct and t5 gemma 9b-2b prefixlm instruct.
"""

import torch
import gc
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


CACHE_DIR = "/scratch/common_models/HuggingFace"


def verify_gemma2():
    print("\n=== 1. download and verify gemma 2 9b instruct ===")
    gemma2 = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        cache_dir=CACHE_DIR,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", cache_dir=CACHE_DIR)

    message = ["Language modeling is "]
    inputs = tokenizer(message, return_tensors="pt", return_token_type_ids=False)
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    response = gemma2.generate(**inputs, max_new_tokens=20, do_sample=True, top_k=0, temperature=1.0, top_p=0.7)
    print("Gemma 2 output verification:", tokenizer.batch_decode(response, skip_special_tokens=True))

    del gemma2, tokenizer, inputs, response
    torch.cuda.empty_cache()
    gc.collect()
    print("Gemma 2 RAM released")


def verify_t5gemma():
    print("\n=== 2. download and verify T5Gemma 9b-2b PrefixLM Instruct ===")
    t5gemma = AutoModelForSeq2SeqLM.from_pretrained(
        "google/t5gemma-9b-2b-prefixlm-it",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        cache_dir=CACHE_DIR,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-9b-2b-prefixlm-it", cache_dir=CACHE_DIR)

    message = ["Language modeling is "]
    input_ids = tokenizer(message, return_tensors="pt")["input_ids"].to("cuda:0")

    response = t5gemma.generate(input_ids, max_new_tokens=20, do_sample=True, temperature=0.1)
    print("T5Gemma verification output:", tokenizer.decode(response[0], skip_special_tokens=True))

    del t5gemma, tokenizer, input_ids, response
    torch.cuda.empty_cache()
    gc.collect()
    print("T5Gemma RAM released")


if __name__ == "__main__":
    verify_gemma2()
    verify_t5gemma()
    print("\nAll models downloaded and verified")