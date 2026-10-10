from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "shared" / "src"))

from moltoken_common import ensure_dirs


PHASE01 = ROOT.parent / "data" / "lipophilicity"
PHASE04 = ROOT / "predictors"
PROCESSED = PHASE01 / "processed" / "lipophilicity_clean.csv"
INDEPENDENT = PHASE01 / "splits" / "independent_100.csv"
SPLITS = PHASE04 / "splits"
REPORTS = PHASE04 / "reports"

SEED = 20260528
TARGET_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}
REGIONS = ["low", "in_range", "high"]


def score_assignment(
    counts: dict[str, dict[str, int]], totals: dict[str, int], target: dict[str, int]
) -> float:
    score = 0.0
    for split, ratio in TARGET_RATIOS.items():
        desired_total = sum(target.values()) * ratio
        score += abs(sum(counts[split].values()) - desired_total) / max(1.0, desired_total)
        for region in REGIONS:
            desired = target[region] * ratio
            score += 2.0 * abs(counts[split][region] - desired) / max(1.0, desired)
    return score


def choose_split_for_group(
    group_counts: dict[str, int],
    counts: dict[str, dict[str, int]],
    target: dict[str, int],
) -> str:
    best_split = "train"
    best_score = float("inf")
    for split in TARGET_RATIOS:
        trial = {key: value.copy() for key, value in counts.items()}
        for region, value in group_counts.items():
            trial[split][region] += value
        score = score_assignment(trial, {}, target)
        if score < best_score:
            best_score = score
            best_split = split
    return best_split


def write_chemprop(df: pd.DataFrame, path: Path) -> None:
    df[["canonical_smiles", "exp"]].rename(columns={"canonical_smiles": "smiles"}).to_csv(path, index=False)


def main() -> None:
    ensure_dirs(SPLITS, REPORTS)
    clean = pd.read_csv(PROCESSED)
    independent = pd.read_csv(INDEPENDENT)
    independent_ids = set(independent["mol_id"])
    df = clean[~clean["mol_id"].isin(independent_ids)].copy()

    target_counts = {region: int((df["target_region"] == region).sum()) for region in REGIONS}
    scaffold_groups = []
    for scaffold, group in df.groupby("murcko_scaffold", dropna=False):
        vc = group["target_region"].value_counts().to_dict()
        scaffold_groups.append(
            {
                "scaffold": scaffold if isinstance(scaffold, str) else "",
                "mol_ids": group["mol_id"].tolist(),
                "size": int(len(group)),
                "counts": {region: int(vc.get(region, 0)) for region in REGIONS},
            }
        )

    rng = np.random.default_rng(SEED)
    for item in scaffold_groups:
        item["tie_break"] = float(rng.random())
        item["dominant_region_count"] = max(item["counts"].values())
    scaffold_groups.sort(
        key=lambda item: (-item["size"], -item["dominant_region_count"], item["tie_break"])
    )

    split_counts = {split: {region: 0 for region in REGIONS} for split in TARGET_RATIOS}
    membership_rows: list[dict[str, str]] = []
    id_to_split: dict[str, str] = {}
    for item in scaffold_groups:
        split = choose_split_for_group(item["counts"], split_counts, target_counts)
        for region, value in item["counts"].items():
            split_counts[split][region] += value
        for mol_id in item["mol_ids"]:
            id_to_split[mol_id] = split
            membership_rows.append(
                {"mol_id": mol_id, "murcko_scaffold": item["scaffold"], "split": split}
            )

    df["scaffold_split"] = df["mol_id"].map(id_to_split)
    split_frames = {split: df[df["scaffold_split"] == split].copy() for split in TARGET_RATIOS}
    keep_cols = [
        "mol_id",
        "CMPD_CHEMBLID",
        "smiles",
        "canonical_smiles",
        "exp",
        "target_region",
        "target_distance_exp",
        "murcko_scaffold",
    ]
    for split, frame in split_frames.items():
        frame[keep_cols].to_csv(SPLITS / f"scaffold_{split}.csv", index=False)
        write_chemprop(frame, SPLITS / f"chemprop_scaffold_{split}.csv")

    membership = pd.DataFrame(membership_rows)
    membership.to_csv(SPLITS / "scaffold_membership.csv", index=False)

    summary = {
        "seed": SEED,
        "independent_excluded": int(len(independent)),
        "remaining_rows": int(len(df)),
        "unique_scaffolds": int(df["murcko_scaffold"].nunique()),
        "split_counts": {split: int(len(frame)) for split, frame in split_frames.items()},
        "target_region_counts": {
            split: frame["target_region"].value_counts().to_dict()
            for split, frame in split_frames.items()
        },
        "scaffold_overlap": {
            f"{a}_intersects_{b}": int(
                len(
                    set(split_frames[a]["murcko_scaffold"]).intersection(
                        set(split_frames[b]["murcko_scaffold"])
                    )
                )
            )
            for a in TARGET_RATIOS
            for b in TARGET_RATIOS
            if a < b
        },
    }
    (REPORTS / "scaffold_split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# 04 Stratified Scaffold Split Report",
        "",
        f"- independent_100 excluded: `{len(independent)}`",
        f"- remaining rows: `{len(df)}`",
        f"- unique scaffolds: `{summary['unique_scaffolds']}`",
        "",
        "| split | count | low | in_range | high |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, frame in split_frames.items():
        vc = frame["target_region"].value_counts()
        lines.append(
            f"| {split} | {len(frame)} | {int(vc.get('low', 0))} | {int(vc.get('in_range', 0))} | {int(vc.get('high', 0))} |"
        )
    lines.extend(["", "## Scaffold Overlap", ""])
    for key, value in summary["scaffold_overlap"].items():
        lines.append(f"- {key}: `{value}`")
    (REPORTS / "scaffold_split_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
