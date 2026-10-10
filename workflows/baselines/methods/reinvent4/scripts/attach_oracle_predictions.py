#!/usr/bin/env python3
"""Attach the locked Chemprop CLI ensemble prediction to each formal parent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direction(value: float, low: float, high: float) -> str:
    if value < low:
        return "increase_logd"
    if value > high:
        return "decrease_logd"
    return "already_in_window"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-table", type=Path, required=True)
    parser.add_argument("--prediction-table", type=Path, required=True)
    parser.add_argument("--prediction-column", default="exp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-low", type=float, default=1.0)
    parser.add_argument("--target-high", type=float, default=3.4)
    args = parser.parse_args()

    parents = pd.read_csv(args.parent_table)
    predictions = pd.read_csv(args.prediction_table)
    required_parent = {"parent_id", "parent_smiles"}
    required_prediction = {"parent_id", "parent_smiles", args.prediction_column}
    if missing := required_parent.difference(parents.columns):
        raise ValueError(f"Parent table missing columns: {sorted(missing)}")
    if missing := required_prediction.difference(predictions.columns):
        raise ValueError(f"Prediction table missing columns: {sorted(missing)}")
    if len(parents) != 90 or len(predictions) != 90:
        raise ValueError(f"Expected 90 rows in both tables; got {len(parents)} and {len(predictions)}")

    reference = predictions[["parent_id", "parent_smiles", args.prediction_column]].rename(
        columns={"parent_smiles": "predicted_parent_smiles"}
    )
    merged = parents.merge(reference, on="parent_id", how="left", validate="one_to_one")
    if merged[args.prediction_column].isna().any():
        bad = merged.loc[merged[args.prediction_column].isna(), "parent_id"].tolist()
        raise ValueError(f"Missing predictions for: {bad}")
    if not merged["parent_smiles"].eq(merged["predicted_parent_smiles"]).all():
        raise ValueError("Parent SMILES differ between locked input and Chemprop CLI output")
    merged = merged.drop(columns="predicted_parent_smiles")
    merged["original_main_pred_logd"] = merged[args.prediction_column].astype(float)
    if args.prediction_column != "original_main_pred_logd":
        merged = merged.drop(columns=args.prediction_column)
    merged["optimization_direction"] = merged["original_main_pred_logd"].map(
        lambda value: direction(value, args.target_low, args.target_high)
    )

    counts = merged["optimization_direction"].value_counts().to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    report = {
        "passed": True,
        "parent_count": int(len(merged)),
        "prediction_column": args.prediction_column,
        "target_window": [args.target_low, args.target_high],
        "direction_counts": counts,
        "parent_table_sha256": sha256(args.parent_table),
        "prediction_table_sha256": sha256(args.prediction_table),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    args.manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
