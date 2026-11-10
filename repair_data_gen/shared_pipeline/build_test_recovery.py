"""Build separately traced test recovery candidates from approved surfaces.

Run in repairsbn. This does not create human labels or a final test split.
"""
from __future__ import annotations

from collections import Counter
import contextlib
import io
import json
from pathlib import Path

from .analyse_split import ROOT, read, write, sha
from .prepare_dev_review import decision
import analyse_run as checks
import check_repair_part
import check_untouched
import check_withdraw
from check_place_r5 import pin as place_pin
from check_temporal_extra import pin as temporal_pin
from repair_transform import build_repair, Strategy
from sbn_lin import parse_sbn_line, serialise
from validate_with_pmb_parser import check


TEST = ROOT / "repair_data_gen/splits/test"
OUT = TEST / "recovery_v1"
# Explicit repair-side suffixes preserve unambiguous insertion offsets even
# when dates, thousands separators or quoted dialogue contain other commas.
TAIL = {
    "T001": "on August 8.", "T002": "22nd February 1732.",
    "T003": "in the cathedral.", "E001": "Swift died in 1745.",
    "E002": "180 people.", "E003": "8586 meters high.",
    "E005": "two cups of coffee.", "E006": "10 dollars.",
    "E007": "about 50 dollars.", "E008": "10,000 yen on his winning.",
    "E009": "stop between Shibuya and Naka-Meguro.",
    "E010": "Eskimos live in Greenland?", "E011": "stand for?",
    "E012": "slipping?", "E013": "candy?", "E014": "his error.",
    "E016": "die?", "E018": "didn't have any privacy.",
    "E019": 'studied karate." "I\'m glad you did, too."',
    "E020": 'on the third floor."', "E021": "1 November 1945.",
    "E022": "about 2:30.",
    "E023": "September, the firm built its first “E112” wind power plant near Magdeburg.",
}
LEXICAL = {
    "E009": ("decelerate.v.01", "slow down"),
    "E010": ("canadian.n.01", "Canadians"),
    "E011": ("rhyme.v.02", "rhyme with"),
    "E012": ("stumble.v.02", "stumbling"),
    "E013": ("cake.n.03", "cake"),
    "E014": ("joke.n.01", "his joke"),
    "E016": ("retire.v.01", "retire"),
}
# Constant replacement: source site, role, withdrawn value and surface.
CONSTANT = {
    "T001": (3, "MonthOfYear", "7", "July 8"),
    "T002": (3, "DayOfMonth", "10", "10th"),
    "E001": (0, "Name", '"Defoe"', "Defoe"),
    "E003": (2, "EQU", "8200", "8200"),
    "E005": (3, "EQU", "5", "five"),
    "E006": (3, "EQU", "25", "25"),
    "E021": (4, "DayOfMonth", "10", "10"),
    "E022": (4, "ClockTime", "15:00", "about 3:00"),
    "E023": (0, "MonthOfYear", "8", "August"),
}


def splice(row, site, synset, constant=None):
    """Use the existing operator; require shared-content attachment too."""
    gold = parse_sbn_line(row["sbn_clean"])
    attempts = []
    for strategy in (Strategy.INPLACE, Strategy.ANCHOR, Strategy.HOIST):
        result = build_repair(gold, site, synset, strategy=strategy)
        doc = parse_sbn_line(result.sbn)
        corr = next(s.new_box for s in doc.separators if s.name == "CORRECTION")
        rep = next(c for c in doc.concepts if c.box == corr)
        if constant:
            role, value = constant
            rep.roles = [(r, value if r == role and not v.startswith(("+", "-")) else v)
                         for r, v in rep.roles]
        if synset == "retire.v.01":
            # PMB gold retire.v.01 examples use Agent. The accepted die.v.01
            # retains its gold Patient role; do not copy that role to retire.
            rep.roles = [("Agent" if r == "Patient" else r, v) for r, v in rep.roles]
        sbn = serialise(doc.concepts, doc.separators)
        trial = dict(row, sbn_repair=sbn)
        structural = check(sbn)
        try:
            probs = check_repair_part.problems(trial)
            missing, added = check_withdraw.diff(trial)
        except (ValueError, IndexError):
            continue
        attempts.append((sbn, result))
        if (all(structural[k] == "ok" for k in checks.CHECKS)
                and not missing and not added
                and not any(p.startswith("R5:") for p in probs)):
            return sbn, result.strategy.value, ";".join(d.value for d in result.devices)
        # Index overflow is handled in a separate chat, never repaired here.
        if structural["magnitude"] != "ok":
            break
    if not attempts:
        raise ValueError(f"No construction for {row['request_id']}")
    sbn, result = attempts[0]
    return sbn, result.strategy.value, ";".join(d.value for d in result.devices)


def validate(row, sample, policy):
    result = check(row["sbn_repair"])
    row.update({k: result[k] or "" for k in checks.CHECKS})
    row["check_detail"] = result["detail"]
    row["reply"] = "ok"  # record decoding, not a claim of a new model reply
    row["wn_ok"], row["wn_missing"] = checks.new_concepts_exist(row["sbn_clean"], row["sbn_repair"])
    row["wn_relation"] = checks.wn_relation(row)
    row["place_inside"] = checks.place_inside(row)
    row["site_ok"] = checks.site_ok(row)
    row["ambiguous"] = checks.ambiguous(row)
    row["untouched"] = "ok" if not check_untouched.missing_edges(row) else "lost gold edges"
    probs = check_repair_part.problems(row)
    row["repair_part"] = ";".join(probs) or "ok"
    missing, added = check_withdraw.diff(row)
    row["withdraw"] = "ok" if not missing and not added else str((missing, added))
    gold = parse_sbn_line(row["sbn_clean"])
    site = int(row["site_index"])
    row["declared_site"] = "ok" if gold.concepts[site].synset == row["repair_synset"] else "mismatch"
    row["pin"] = ((temporal_pin if row["repair_type"] == "constant_temporal" else place_pin)(row, sample)
                  if sample.get("pinned_site") else "ok")
    row["machine_pass"] = str(not any(row.get(k) != "ok" for k in
        checks.CHECKS + ("wn_ok", "untouched", "repair_part", "withdraw", "site_ok", "pin", "declared_site"))
        and row["wn_relation"] != "same_synset" and row["place_inside"] != "yes")
    row["policy_eligible"], row["policy_reason"] = decision(row, policy)
    return row


def main():
    inventory_path = OUT / "inventory.json"
    decisions_path = OUT / "surface_decisions.json"
    inv = json.loads(inventory_path.read_text())
    for path, digest in inv["input_sha256"].items():
        if sha(TEST / path) != digest:
            raise ValueError(f"Inventory source changed: {path}")
    decisions = {r["request_id"]: r for r in json.loads(decisions_path.read_text())["decisions"]}
    samples = {r["request_id"]: r for path in sorted((TEST / "samples").glob("*.tsv")) for r in read(path)}
    policy = json.loads((TEST / "review_policy.json").read_text())
    recovered, dispositions = [], []
    for item in inv["items"]:
        key, request = item["review_id"], item["request_id"]
        approved = decisions[request]
        if not approved["decision"].startswith("surface_approved"):
            dispositions.append({"review_id": key, "request_id": request,
                                 "status": approved["decision"]})
            continue
        row = dict(item["source_record"])
        row.update(nl_repair=approved["nl_repair"], status="ok", interregnum="none",
                   review_id=key, recovery_id=f"recovery_v1::{request}",
                   recovery_origin="local_reconstruction_after_user_surface_review",
                   original_category=item["category"], original_source=item["source_table"],
                   original_model=row.get("model", ""), final_acceptance="False")
        row["model"] = ""  # do not attribute reconstructed targets to the cluster model
        if key in TAIL:
            tail = TAIL[key]
            if not row["nl_repair"].endswith(tail):
                raise ValueError(f"Repair suffix mismatch: {key}")
            row["repair_onset"] = str(len(row["nl_repair"]) - len(tail))
        onset = int(row["repair_onset"])
        if row["nl_repair"][onset - 2:onset] != ", ":
            raise ValueError(f"Invalid boundary: {key}")
        if key in CONSTANT:
            site, role, value, surface = CONSTANT[key]
            row.update(site_index=str(site), repair_synset=parse_sbn_line(row["sbn_clean"]).concepts[site].synset,
                       reparandum_value=value.strip('"'), reparandum_surface=surface)
            row["reparandum_synset"] = row["repair_synset"]
            row["sbn_repair"], row["strategy"], row["devices"] = splice(row, site, row["repair_synset"], (role, value))
        elif key in LEXICAL:
            synset, surface = LEXICAL[key]
            row.update(reparandum_synset=synset, reparandum_surface=surface)
            row["sbn_repair"], row["strategy"], row["devices"] = splice(row, int(row["site_index"]), synset)
        elif key == "T003":
            row.update(site_index="3", repair_synset="take_place.v.01", reparandum_synset="entity.n.01",
                       reparandum_value="STO", reparandum_surface="outside the cathedral", strategy="dummy_place", devices="d3_dummy")
            row["sbn_repair"] = ('funeral.n.01 service.n.03 Theme -1 time.n.08 TSU now '
                'take_place.v.01 Theme -2 Time -1 EQU +1 entity.n.01 cathedral.n.01 '
                'CORRECTION <1 entity.n.01 EQU -3 Location +1 location.n.01 STO -2 '
                'CONJUNCTION <2 entity.n.01 EQU -5 Location -3')
        elif key == "E007":
            row["site_index"] = "3"
        elif key == "E018":
            row["site_index"] = "2"
        elif key == "E019":
            row["site_index"] = "4"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                validate(row, samples[request], policy)
        except Exception as error:
            row.update(policy_eligible=False, policy_reason=f"validation error: {type(error).__name__}: {error}")
        recovered.append(row)
        dispositions.append({"review_id": key, "request_id": request,
                             "status": "eligible_for_full_review" if row["policy_eligible"] else "unresolved",
                             "reason": row["policy_reason"]})
    write(OUT / "reconstructed.checked.tsv", recovered)
    (OUT / "dispositions.json").write_text(json.dumps(dispositions, ensure_ascii=False, indent=2) + "\n")
    summary = {"status": "reconstructed candidates; no final human acceptance",
               "counts": dict(Counter(r["status"] for r in dispositions)),
               "input_sha256": {str(p): sha(p) for p in (inventory_path, decisions_path, TEST / "review_policy.json",
                    *sorted((TEST / "samples").glob("*.tsv")))},
               "code_sha256": {str(Path(__file__)): sha(Path(__file__))},
               "output_sha256": {str(OUT / "reconstructed.checked.tsv"): sha(OUT / "reconstructed.checked.tsv")}}
    (OUT / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["counts"]))
    for rec in dispositions:
        if rec["status"] == "unresolved":
            print(rec["review_id"], rec["reason"])


if __name__ == "__main__":
    main()
