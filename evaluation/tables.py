#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Table IO shared by the evaluation scripts. Format is inferred from the extension."""

from pathlib import Path

import pandas as pd

__all__ = ["read_table", "write_table"]


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=sep, dtype=str)


def write_table(df: pd.DataFrame, path) -> None:
    path = Path(path)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        sep = "\t" if path.suffix in (".tsv", ".txt") else ","
        df.to_csv(path, sep=sep, index=False)
