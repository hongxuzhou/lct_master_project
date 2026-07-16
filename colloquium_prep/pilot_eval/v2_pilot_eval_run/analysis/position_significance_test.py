"""
Test whether repair position (head/mid/tail) significantly affects
(a) success/fail outcome, and (b) f1 quality among successes,
separately for the vanilla-repair group and the interregnum group.

Data is repeated-measures: same doc id appears in all 7 conditions,
so we use paired/repeated-measures tests, not independent-sample tests.
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import chi2, friedmanchisquare, wilcoxon

DF_PATH = "v2_pilot_eval_run/preds_long_scored.parquet"

VANILLA = ["repair_head", "repair_mid", "repair_tail"]
INTERREG = ["repair_head_interrug", "repair_mid_interrug", "repair_tail_interrug"]


def cochrans_q(mat: np.ndarray):
    """mat: N x k binary matrix (1=success). Returns Q, df, p."""
    N, k = mat.shape
    col_sums = mat.sum(axis=0)
    row_sums = mat.sum(axis=1)
    Cbar = col_sums.mean()
    numerator = k * (k - 1) * np.sum((col_sums - Cbar) ** 2)
    denominator = k * row_sums.sum() - np.sum(row_sums ** 2)
    Q = numerator / denominator
    df = k - 1
    p = chi2.sf(Q, df)
    return Q, df, p


def mcnemar(a: np.ndarray, b: np.ndarray, correction=True):
    """Paired binary vectors a, b (1=success). Returns stat, p (asymptotic chi2, df=1)."""
    n01 = np.sum((a == 0) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0))
    n = n01 + n10
    if n == 0:
        return np.nan, 1.0, n01, n10
    if correction:
        stat = (abs(n01 - n10) - 1) ** 2 / n
    else:
        stat = (n01 - n10) ** 2 / n
    p = chi2.sf(stat, 1)
    return stat, p, n01, n10


def holm_correct(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj_p = (m - rank) * pvals[idx]
        running_max = max(running_max, adj_p)
        adj[idx] = min(running_max, 1.0)
    return adj


def analyze_group(df: pd.DataFrame, conditions: list[str], label: str):
    print(f"\n{'='*70}\nGROUP: {label}  conditions={conditions}\n{'='*70}")

    sub = df[df["condition"].isin(conditions)].copy()
    sub["success"] = (sub["status"] == "success").astype(int)

    # wide success matrix: doc id x condition
    wide_succ = sub.pivot(index="id", columns="condition", values="success")
    wide_succ = wide_succ.dropna()  # keep only docs present in all 3 conditions
    wide_succ = wide_succ[conditions]  # enforce column order = position order
    n_docs = len(wide_succ)
    print(f"\n[Success/Fail] complete triplets: N={n_docs} docs")
    print("success rate per condition:")
    print(wide_succ.mean().round(4))

    Q, df_, p = cochrans_q(wide_succ.values)
    print(f"\nCochran's Q = {Q:.3f}, df={df_}, p={p:.6f}")

    # pairwise McNemar with Holm correction
    pairs = list(itertools.combinations(conditions, 2))
    raw_p = []
    details = []
    for c1, c2 in pairs:
        stat, pv, n01, n10 = mcnemar(wide_succ[c1].values, wide_succ[c2].values)
        raw_p.append(pv)
        details.append((c1, c2, stat, pv, n01, n10))
    adj_p = holm_correct(np.array(raw_p))
    print("\nPairwise McNemar (Holm-corrected):")
    for (c1, c2, stat, pv, n01, n10), apv in zip(details, adj_p):
        print(f"  {c1:>22} vs {c2:<22}  chi2={stat:6.3f}  p_raw={pv:.6f}  p_holm={apv:.6f}  (n01={n01}, n10={n10})")

    # --- F1 among successes, complete-case triplets ---
    wide_f1 = sub.pivot(index="id", columns="condition", values="f1")
    wide_f1 = wide_f1[conditions].dropna()  # only docs that succeeded in ALL 3
    n_f1 = len(wide_f1)
    print(f"\n[F1 quality] complete-case triplets (success in all 3): N={n_f1} docs")
    print("mean f1 per condition:")
    print(wide_f1.mean().round(4))

    if n_f1 >= 10:
        stat, p_fr = friedmanchisquare(*[wide_f1[c].values for c in conditions])
        print(f"\nFriedman chi2 = {stat:.3f}, p={p_fr:.6f}")

        raw_p2 = []
        details2 = []
        for c1, c2 in pairs:
            stat_w, pv_w = wilcoxon(wide_f1[c1].values, wide_f1[c2].values)
            raw_p2.append(pv_w)
            details2.append((c1, c2, stat_w, pv_w))
        adj_p2 = holm_correct(np.array(raw_p2))
        print("\nPairwise Wilcoxon signed-rank (Holm-corrected):")
        for (c1, c2, stat_w, pv_w), apv in zip(details2, adj_p2):
            print(f"  {c1:>22} vs {c2:<22}  W={stat_w:8.1f}  p_raw={pv_w:.6f}  p_holm={apv:.6f}")
    else:
        print("Too few complete cases for Friedman test.")


def main():
    df = pd.read_parquet(DF_PATH)

    # exclude the 3 docs whose gold itself errors (data defect, not model behavior)
    gold_error_ids = df.loc[df["status"] == "gold_error", "id"].unique()
    print(f"Excluding {len(gold_error_ids)} docs with gold_error: {list(gold_error_ids)}")
    df = df[~df["id"].isin(gold_error_ids)].copy()

    analyze_group(df, VANILLA, "vanilla (repair_head/mid/tail)")
    analyze_group(df, INTERREG, "interregnum (repair_*_interrug)")


if __name__ == "__main__":
    main()
