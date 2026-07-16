"""
Build a stratified qualitative sample for reading, informed by:
  - RQ1 (why does the model succeed despite reparandum noise?)
  - RQ2 (what causes failures?)
  - the position-confound hypothesis: head/mid/tail is an LLM-free-text
    instruction target, not a controlled syntactic variable, so the
    statistical position effect may actually be driven by which
    constituent (subject / verb / object / adverbial / subordinate-clause
    opener) gets interrupted -- not by linear position per se.

Three strata, each tagged in `sample_group`:
  A. longitudinal  -- 6-7 doc ids traced through all 7 conditions
                       (typical / resilient / fragile, picked using
                       repair_mid_interrug as the hardest condition)
  B. failure_survey -- parse_error + ill_formed rows, weighted toward
                       repair_mid_interrug / repair_tail_interrug where
                       parse_error is concentrated
  C. manual_repairs -- the 7 doc ids with known-defective LLM
                       augmentations (MANUAL_REPAIRS in build_pilot_dataset.py),
                       all 7 conditions each, to check whether known-bad
                       rows leaked into the "success" counts

Output: one Excel file with all sampled rows + gold nl/mr for reference +
blank columns for manual syntactic-category annotation.
"""
import numpy as np
import pandas as pd

SCORED_PATH = "v2_pilot_eval_run/preds_long_scored.parquet"
GOLD_PATH = "../pilot_dataset.parquet"  # relative to analysis/ working dir when run from pilot_eval/
OUT_PATH = "v2_pilot_eval_run/analysis/qualitative_sample.xlsx"

VANILLA = ["repair_head", "repair_mid", "repair_tail"]
INTERREG = ["repair_head_interrug", "repair_mid_interrug", "repair_tail_interrug"]
ALL_CONDS = ["gold"] + VANILLA + INTERREG

# id -> overridden plain condition (from MANUAL_REPAIRS in build_pilot_dataset.py).
# Only this condition + its interregnum counterpart actually touched the known
# defect; the other conditions for these ids are ordinary rows and add no signal.
MANUAL_REPAIR_CONDITIONS = {
    "p90/d2399": "repair_head",
    "p67/d2005": "repair_head",
    "p16/d1602": "repair_head",
    "p72/d2539": "repair_head",
    "p36/d3158": "repair_head",
    "p01/d1880": "repair_mid",
    "p41/d3041": "repair_mid",
}

RNG_SEED = 42


def load_data():
    scored = pd.read_parquet(SCORED_PATH)
    gold_ref = pd.read_parquet(GOLD_PATH)[["id", "nl", "mr"]].rename(
        columns={"nl": "gold_nl", "mr": "gold_mr"}
    )
    return scored, gold_ref


def stratum_longitudinal(df: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    """Pick doc ids for full 7-condition tracing: typical / resilient / fragile."""
    hardest = df[df["condition"] == "repair_mid_interrug"].set_index("id")
    head = df[df["condition"] == "repair_head"].set_index("id")

    # fragile: already fails (non-success) in the *easiest* vanilla condition
    fragile_pool = head[head["status"] != "success"].index.tolist()
    fragile = list(rng.choice(fragile_pool, size=min(2, len(fragile_pool)), replace=False))

    # resilient: succeeds in the hardest condition with f1 in the top quartile
    succ_hard = hardest[hardest["status"] == "success"]
    q75 = succ_hard["f1"].quantile(0.75)
    resilient_pool = succ_hard[succ_hard["f1"] >= q75].index.tolist()
    resilient = list(rng.choice(resilient_pool, size=min(2, len(resilient_pool)), replace=False))

    # typical: succeeds in the hardest condition with f1 near the median
    median = succ_hard["f1"].median()
    succ_hard = succ_hard.assign(dist=(succ_hard["f1"] - median).abs())
    typical_pool = succ_hard.sort_values("dist").index.tolist()
    typical_pool = [i for i in typical_pool if i not in fragile and i not in resilient]
    typical = list(rng.choice(typical_pool[:20], size=min(3, len(typical_pool)), replace=False))

    return fragile + resilient + typical


def stratum_failure_survey(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for cond in ALL_CONDS:
        sub = df[df["condition"] == cond]
        pe = sub[sub["status"] == "parse_error"]
        ill = sub[sub["status"] == "ill_formed"]

        if cond in ("repair_mid_interrug", "repair_tail_interrug"):
            n_pe = min(7, len(pe))
        elif cond == "gold":
            n_pe = min(1, len(pe))
        else:
            n_pe = min(2, len(pe))
        n_ill = min(2, len(ill))

        if n_pe:
            rows.append(pe.sample(n=n_pe, random_state=rng.integers(1e6)))
        if n_ill:
            rows.append(ill.sample(n=n_ill, random_state=rng.integers(1e6)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    scored, gold_ref = load_data()
    rng = np.random.default_rng(RNG_SEED)

    gold_error_ids = scored.loc[scored["status"] == "gold_error", "id"].unique().tolist()
    clean = scored[~scored["id"].isin(gold_error_ids)].copy()

    # --- A. longitudinal ---
    long_ids = stratum_longitudinal(clean, rng)
    long_rows = clean[clean["id"].isin(long_ids)].copy()
    long_rows["sample_group"] = "A_longitudinal"

    # --- B. failure survey ---
    fail_rows = stratum_failure_survey(clean, rng)
    fail_rows["sample_group"] = "B_failure_survey"

    # --- C. manual_repairs (known-defective augmentation ids) ---
    # only the overridden condition + its interregnum counterpart + gold (for reference)
    manual_frames = []
    for rid, plain_cond in MANUAL_REPAIR_CONDITIONS.items():
        interrug_cond = plain_cond + "_interrug"
        wanted_conds = ["gold", plain_cond, interrug_cond]
        manual_frames.append(
            scored[(scored["id"] == rid) & (scored["condition"].isin(wanted_conds))]
        )
    manual_rows = pd.concat(manual_frames, ignore_index=True)
    manual_rows["sample_group"] = "C_manual_repairs_known_defect"

    combined = pd.concat([long_rows, fail_rows, manual_rows], ignore_index=True)

    # collapse duplicate (id, condition) rows that landed in >1 stratum, keep all group tags
    combined["dup_key"] = combined["id"] + "||" + combined["condition"]
    group_tags = combined.groupby("dup_key")["sample_group"].apply(lambda s: "+".join(sorted(set(s))))
    combined = combined.drop_duplicates("dup_key").drop(columns=["sample_group"]).merge(
        group_tags.rename("sample_group"), left_on="dup_key", right_index=True
    ).drop(columns=["dup_key"])

    combined = combined.merge(gold_ref, on="id", how="left")

    # order columns, add blank annotation columns
    combined["syntactic_category"] = ""   # subject / verb-predicate / object-complement / adverbial / subclause-opener / other
    combined["category_match"] = ""       # yes / no / unclear -- does reparandum match repair's syntactic category?
    combined["notes"] = ""

    cols = [
        "sample_group", "id", "condition", "status", "f1",
        "input_nl", "pred_sbn", "gold_nl", "gold_mr",
        "syntactic_category", "category_match", "notes",
    ]
    combined = combined[cols].sort_values(
        ["sample_group", "id", "condition"], key=lambda s: s.map(
            {c: i for i, c in enumerate(ALL_CONDS)}
        ) if s.name == "condition" else s
    )

    combined.to_excel(OUT_PATH, index=False)
    print(f"Wrote {len(combined)} rows to {OUT_PATH}")
    print(combined["sample_group"].value_counts())
    print(f"\nlongitudinal doc ids: {long_ids}")


if __name__ == "__main__":
    main()
