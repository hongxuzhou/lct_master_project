"""
Push the static pilot dataset to the Hugging Face Hub via the native `datasets` API.

Idiomatic HF flow (this is the part worth learning):
  - load the local parquet into a `datasets.Dataset` object
  - inspect `.features` -- the typed schema the Hub stores and the viewer reads
  - `push_to_hub(...)` -- the library re-encodes to parquet shards, uploads them,
    and wires up the dataset viewer automatically. No manual file upload needed.

Auth: relies on the already-stored HF token (`huggingface-cli login` done).
The target repo Shrikes/self_repair_parsing_pilot_data must already exist.
"""

from pathlib import Path

from datasets import Dataset

REPO = Path("/Users/hongxuzhou/Documents/GitHub/lct_master_project")
PARQUET = REPO / "colloquium_prep" / "pilot_dataset.parquet"
HF_REPO_ID = "Shrikes/self_repair_parsing_pilot_data"


def main():
    print(f"Loading {PARQUET.name} into a datasets.Dataset ...")
    ds = Dataset.from_parquet(str(PARQUET))

    print(f"\nrows: {ds.num_rows}")
    print("features (typed schema stored on the Hub):")
    for name, feat in ds.features.items():
        print(f"  {name}: {feat}")

    print(f"\nFirst example id = {ds[0]['id']}")

    print(f"\nPushing to https://huggingface.co/datasets/{HF_REPO_ID} ...")
    ds.push_to_hub(HF_REPO_ID)
    print("Done. Pull on the HPC with:")
    print(f'  load_dataset("{HF_REPO_ID}")')


if __name__ == "__main__":
    main()