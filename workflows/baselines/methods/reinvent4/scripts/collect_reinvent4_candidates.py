#!/usr/bin/env python3
"""Collect REINVENT4 stage CSVs into a parent-linked, budget-audited table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


STATE_LABELS = {0: "invalid", 1: "valid", 2: "duplicate_within_batch"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="summary_*.csv")
    parser.add_argument("--parent-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-parents", type=int, default=88)
    parser.add_argument("--expected-steps", type=int, default=50)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()

    files = sorted(args.summary_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No {args.pattern!r} files in {args.summary_dir}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source_summary_file"] = path.name
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    required = {
        "SMILES",
        "SMILES_state",
        "Input_SMILES",
        "Score",
        "main_logD_target_window",
        "main_logD_target_window (raw)",
        "step",
    }
    if missing := required.difference(raw.columns):
        raise ValueError(f"REINVENT summaries missing columns: {sorted(missing)}")

    parents = pd.read_csv(args.parent_table)
    parent_required = {"parent_id", "parent_smiles"}
    if missing := parent_required.difference(parents.columns):
        raise ValueError(f"Parent table missing columns: {sorted(missing)}")
    parents = parents.copy()
    parents["parent_key"] = parents["parent_smiles"].map(canonical)
    if parents["parent_key"].isna().any():
        bad = parents.loc[parents["parent_key"].isna(), "parent_id"].tolist()
        raise ValueError(f"Invalid parent SMILES for IDs: {bad}")
    if parents["parent_key"].duplicated().any():
        bad = parents.loc[parents["parent_key"].duplicated(False), "parent_id"].tolist()
        raise ValueError(f"Canonical parent SMILES are not unique: {bad}")

    raw["parent_key"] = raw["Input_SMILES"].map(canonical)
    linked = raw.merge(
        parents[["parent_id", "parent_smiles", "parent_key"]],
        on="parent_key",
        how="left",
        validate="many_to_one",
    )
    if linked["parent_id"].isna().any():
        missing_inputs = linked.loc[linked["parent_id"].isna(), "Input_SMILES"].unique()
        raise ValueError(f"Could not map Input_SMILES to locked parents: {missing_inputs.tolist()}")

    linked.insert(0, "raw_attempt_index", range(len(linked)))
    linked["smiles_state_label"] = linked["SMILES_state"].map(STATE_LABELS)
    if linked["smiles_state_label"].isna().any():
        values = sorted(linked.loc[linked["smiles_state_label"].isna(), "SMILES_state"].unique())
        raise ValueError(f"Unknown REINVENT SMILES_state values: {values}")
    linked["chemically_valid"] = linked["SMILES_state"].isin([1, 2])
    linked["reward_eligible"] = linked["SMILES_state"].eq(1)
    linked["duplicate_within_batch"] = linked["SMILES_state"].eq(2)
    linked = linked.rename(
        columns={
            "SMILES": "candidate_smiles",
            "Input_SMILES": "reinvent_input_smiles",
            "Score": "reinvent_total_reward",
            "main_logD_target_window": "logd_window_reward",
            "main_logD_target_window (raw)": "predicted_logd",
        }
    )

    parent_count = int(linked["parent_id"].nunique())
    expected_attempts = args.expected_parents * args.expected_steps
    per_parent = linked.groupby("parent_id").size()
    formal_ok = bool(
        parent_count == args.expected_parents
        and len(linked) == expected_attempts
        and set(linked["step"].astype(int)) == set(range(1, args.expected_steps + 1))
        and per_parent.eq(args.expected_steps).all()
    )
    if not args.allow_smoke and not formal_ok:
        raise ValueError(
            "Formal budget audit failed: "
            f"parents={parent_count}, attempts={len(linked)}, "
            f"steps={sorted(linked['step'].astype(int).unique().tolist())}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    linked.to_csv(args.output, index=False)
    manifest = {
        "budget_shape_passed": formal_ok,
        "formal_budget_passed": bool(formal_ok and not args.allow_smoke),
        "smoke_mode": bool(args.allow_smoke),
        "source_files": [str(path.resolve()) for path in files],
        "source_sha256": {path.name: sha256(path) for path in files},
        "parent_count": parent_count,
        "raw_attempts": int(len(linked)),
        "expected_raw_attempts": expected_attempts,
        "valid_or_duplicate": int(linked["chemically_valid"].sum()),
        "reward_eligible_valid": int(linked["reward_eligible"].sum()),
        "invalid": int(linked["SMILES_state"].eq(0).sum()),
        "duplicates_within_batch": int(linked["duplicate_within_batch"].sum()),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
