"""
This scrips loads models for my master's study to the common dir in the LST HPC at UdS. 

The models include Olmo 3 7b base, Bolmo 3 7b, and Xiao's byT5-parser.
"""

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration

# load the models to the common directory 
CACHE_DIR = "/scratch/common_models/HuggingFace"
device = "cuda"

def verify_olmo():
    print("\n=== 1. download and verify Olmo 7b ===")
    # use float 16 
    olmo = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1025-7B", cache_dir=CACHE_DIR,).to(device)
    tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B", cache_dir=CACHE_DIR)
    
    message = ["Language modeling is "]
    inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    response = olmo.generate(**inputs, max_new_tokens=20, do_sample=True, top_k=0, temperature=1.0, top_p=0.7)
    print("Olmo output verification:", tokenizer.batch_decode(response, skip_special_tokens=True))
    
    # Verification done, release GPU and RAM 
    del olmo, tokenizer, inputs, response
    torch.cuda.empty_cache()
    gc.collect()
    print("Olmo RAM released")

def verify_bolmo():
    print("\n=== 2. download and verify Bolmo 7B ===")
    bolmo = AutoModelForCausalLM.from_pretrained("allenai/Bolmo-7B", trust_remote_code=True, cache_dir=CACHE_DIR,).to(device)
    tokenizer = AutoTokenizer.from_pretrained("allenai/Bolmo-7B", trust_remote_code=True, cache_dir=CACHE_DIR)

    message = ["Language modeling is "]
    input_ids = tokenizer(message, return_tensors="pt")["input_ids"].to(device)

    response = bolmo.generate(input_ids, max_new_tokens=20, do_sample=True, temperature=0.1)
    print("Bolmo verification output:", tokenizer.decode(response[0], skip_special_tokens=True))

    del bolmo, tokenizer, input_ids, response
    torch.cuda.empty_cache()
    gc.collect()
    print("Bolmo RAM realsed")

def verify_byt5():
    print("\n=== 3. download and verify byT5 DRS parser ===")
    tokenizer = AutoTokenizer.from_pretrained('XiaoZhang98/byT5-DRS', max_length=512, cache_dir=CACHE_DIR)
    model = T5ForConditionalGeneration.from_pretrained("XiaoZhang98/byT5-DRS", cache_dir=CACHE_DIR).to(device)

    example = "I am a student."
    x = tokenizer(example, return_tensors='pt', padding=True, truncation=True, max_length=512)['input_ids'].to(device)

    output = model.generate(x)
    pred_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("byT5 verification output:", pred_text)

    del model, tokenizer, x, output
    torch.cuda.empty_cache()
    gc.collect()
    print("byT5 RAM released")

if __name__ == "__main__":
    verify_olmo()
    verify_bolmo()
    verify_byt5()
    print("\nAll models downloaded and verified")