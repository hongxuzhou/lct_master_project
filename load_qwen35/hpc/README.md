# Qwen3.5 model inspection jobs

The cluster copy of this directory is expected at:

```text
/nethome/honzhou/thesis_zone/load_qwen35/
```

Model snapshots use the shared cache at
`/scratch/common_models/HuggingFace`. Small architecture and runtime reports
are written to `architecture_reports/` beside the script. Each architecture
report preserves the complete checkpoint config, then separately enumerates
the text backbone that will be relevant to the later sentence-to-SBN task.

## One-time preparation

~~Create a dedicated environment at `/nethome/honzhou/venvs/qwen35`. It must
contain a CUDA-compatible PyTorch build plus current versions of
`transformers`, `accelerate`, `huggingface_hub`, and `safetensors`.
`transformers` must expose `Qwen3_5ForCausalLM`; the script fails early if it
does not.~~

Use the conda env `thesis`. I updated all the required packages. 

Before submitting, create the log directory and make the wrappers executable:

```bash
mkdir -p /scratch/hongxuzhou/logs/qwen35_inspection
chmod +x /nethome/honzhou/thesis_zone/load_qwen35/hpc/*.sh
```

## Submit

Run the CPU stage first:

```bash
cd /nethome/honzhou/thesis_zone/load_qwen35
condor_submit hpc/prepare.sub
```

After it succeeds, submit the two GPU stages:

```bash
condor_submit hpc/verify_small.sub
condor_submit hpc/verify_27b.sub
```

Alternatively, let DAGMan enforce the dependency:

```bash
condor_submit_dag hpc/pipeline.dag
```

`verify_small` requests an A100 40 GB and runs 2B then 9B in separate Python
processes. `verify_27b` requests one 80 GB A100/H100. All verification is
offline and fails instead of silently offloading weights to CPU or disk.
