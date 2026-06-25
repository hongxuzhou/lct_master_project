#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 1 LoRA SFT: Gemma-2-9B-it on PMB 5.1.0 gold (sentence -> SBN).

Design notes (read before changing anything):

1. Hardware. Run on ONE GPU only (cluster constraint). Gemma-2-9B in bf16 is
   ~18GB of weights alone; with LoRA optimizer states + activations you want an
   80GB card (hopper-* H100, or tony-3 A100-80G). On a 40GB card (tony-4) enable
   USE_4BIT below. V100 (tony-1/2) is NOT recommended: 32GB is too tight and
   V100 has no bf16 (fp16 + Gemma-2 logit-softcap is numerically fragile).

2. LoRA config follows "LoRA Without Regret" (Schulman & Thinking Machines, 2025):
   - Apply LoRA to ALL linear layers (attention + MLP), not attention-only.
   - Optimal LoRA LR ~= 10x FullFT LR  -> we use 2e-4.
   - Keep effective batch size small (<=32); LoRA pays a loss penalty at large
     batch independent of rank. We use 16.
   - rank is not very sensitive over 4-512; r=32 is ample for ~9.5k examples.

3. Zhang & Bos (2025) exact LoRA hyperparameters are NOT published in the paper.
   Items marked TODO(Xiao) must be confirmed with Xiao before claiming alignment.
"""

import os
import json
import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# ----------------------------------------------------------------------------- #
# Paths / environment (cluster-specific)
# ----------------------------------------------------------------------------- #
# Local snapshot already on /scratch; point HF cache to scratch so nothing large
os.environ.setdefault("HF_HOME", "/scratch/hongxuczhou/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/scratch/hongxuzhou/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # model is already local
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The local model directory on the shared cluster cache.
DEFAULT_MODEL_PATH = "/scratch/common_models/HuggingFace/models--google--gemma-2-9b-it"
DEFAULT_DATA_DIR   = "/nethome/honzhou/thesis_zone/pmb-5.1.0"


def resolve_model_path(p):
    """HF cache dirs store the real files under snapshots/<hash>/.
    Accept either a plain dir or an HF-cache dir and return a loadable path."""
    snap = os.path.join(p, "snapshots")
    if os.path.isdir(snap):
        subs = [os.path.join(snap, d) for d in os.listdir(snap)]
        subs = [d for d in subs if os.path.isdir(d)]
        if subs:
            return sorted(subs)[-1]
    return p


# ----------------------------------------------------------------------------- #
# Args
# ----------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--train_file", default="train_gold.json")
    ap.add_argument("--dev_file", default="dev_standard.json")
    ap.add_argument("--output_dir", default="/scratch/hongxuzhou/thesis/pilot/runs/sbn-gemma2-9b-lora-pmb")

    # LoRA Without Regret defaults
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)      # alpha = 2*r convention
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--learning_rate", type=float, default=2e-4)   # ~10x FullFT

    # Effective batch = per_device * grad_accum. Keep <= 32 (paper). Here 16.
    ap.add_argument("--per_device_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)

    ap.add_argument("--epochs", type=float, default=3.0)       # TODO(Xiao): confirm
    ap.add_argument("--max_seq_length", type=int, default=1024) # gold max ~66 SBN + prompt; 512 is safe -- I changed to 1024 for future repair resolution -- Hongxu Zhou 24/Jun/2026
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--lr_scheduler", default="cosine")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_4bit", action="store_true",
                    help="Enable QLoRA 4-bit. Use ONLY on a 40GB card (tony-4).")
    ap.add_argument("--use_wandb", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if not args.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.setdefault("WANDB_PROJECT", "sbn-lora")

    model_path = resolve_model_path(args.model_path)
    print(f"[info] loading model from: {model_path}")
    print(f"[info] bf16 supported: {torch.cuda.is_bf16_supported()}")

    # ------------------------------------------------------------------------- #
    # Tokenizer
    # ------------------------------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Gemma-2 has a pad token; if missing, fall back to eos.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # For causal LM SFT we right-pad during training.
    tokenizer.padding_side = "right"

    # --------------------------------------------
    #  Rewrite Gemma 2 chat template, inserting the generation tag required by TRL - Hongxu Zhou 24/jun
    tokenizer.chat_template = (
        "{{ bos_token }}"
        "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}"
        "{% for message in messages %}"
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
        "{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}"
        "{{ '<start_of_turn>' + role + '\n' }}"
        "{% if message['role'] == 'assistant' %}"
            "{% generation %}{{ message['content'] | trim }}{% endgeneration %}"
        "{% else %}"
            "{{ message['content'] | trim }}"
        "{% endif %}"
        "{{ '<end_of_turn>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
    )

    # ------------------------------------------------------------------------- #
    # Quantization (optional, 40GB card only)
    # ------------------------------------------------------------------------- #
    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # ------------------------------------------------------------------------- #
    # Model
    # ------------------------------------------------------------------------- #
    # attn_implementation: Gemma-2 uses sliding-window attention + logit softcap.
    # "eager" is the safe/correct choice for Gemma-2 (FA2 historically dropped the
    # softcap). On H100 you may try "flash_attention_2" if installed AND you have
    # verified loss parity, but eager is the conservative default.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16, # no torch_dtype anymore 
        quantization_config=quant_config,
        attn_implementation="eager",
        device_map={"": 0},          # single GPU, no sharding
    )
    model.config.use_cache = False   # required with gradient checkpointing
    model.config.pretraining_tp = 1

    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    # ------------------------------------------------------------------------- #
    # LoRA: ALL linear layers (attention + MLP) per LoRA Without Regret
    # ------------------------------------------------------------------------- #
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",      # MLP
    ]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------------- #
    # Data
    # ------------------------------------------------------------------------- #
    data_files = {
        "train": os.path.join(args.data_dir, args.train_file),
        "validation": os.path.join(args.data_dir, args.dev_file),
    }
    ds = load_dataset("json", data_files=data_files)
    print(f"[info] train={len(ds['train'])}  dev={len(ds['validation'])}")

    # Records already carry a `messages` field (user prompt -> assistant SBN).
    # SFTTrainer will apply Gemma-2's chat template to `messages` automatically.
    # We strip the extra columns so the trainer only sees `messages`.
    keep = "messages"
    remove_cols = [c for c in ds["train"].column_names if c != keep]
    ds = ds.remove_columns(remove_cols)

    # ------------------------------------------------------------------------- #
    # SFT config
    # ------------------------------------------------------------------------- #
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_torch",
        max_grad_norm=1.0,

        bf16=True,
        fp16=False,

        max_length=args.max_seq_length, # trl 0.24 changed the parameter name
        packing=False,                  # short, heterogeneous targets -> no packing
        dataset_kwargs={"skip_prepare_dataset": False},

        # Train only on the assistant's SBN, not the prompt. This is the single
        # most important correctness knob for instruction-format SFT.
        assistant_only_loss=True, 

        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        seed=args.seed,
        report_to=("wandb" if args.use_wandb else "none"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,       
    )

    # ------------------------------------------------------------------------- #
    # Train
    # ------------------------------------------------------------------------- #
    trainer.train()

    # Save adapter ONLY (~150-400MB). Per the plan this is what gets
    # pushed to HF (Shrikes/sbn-gemma2-9b-lora-pmb).
    final_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[done] adapter saved to {final_dir}")


if __name__ == "__main__":
    main()