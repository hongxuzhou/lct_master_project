# Qwen3.5 LoRA smoke tests

This directory has one cluster-neutral implementation and two thin cluster
adapters.

- `qwen35_lora_smoke_test.py` owns dataset selection, tokenization, LoRA,
  training, save/reload, and validation.
- `preflight_tokenization.py` checks the same tokenization and loss masking on
  CPU before a GPU is requested.
- `revisions.json` is the single source of pinned model and dataset commits.
- `habrok/` contains the Slurm adapter.
- `lst/` contains the UdS LST HTCondor adapter.

Do not fork the Python implementation per cluster. Cluster paths, caches,
authentication, offline mode, scheduler resources, and output staging belong
in the adapter directories.

Changing a revision is an explicit experiment-input change. Update the
manifest deliberately and rerun preparation plus CPU preflight on both
clusters before requesting GPUs.
