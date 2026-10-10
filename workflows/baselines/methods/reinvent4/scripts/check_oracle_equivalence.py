#!/usr/bin/env python3
"""Numerically verify that REINVENT scoring reproduces MolToken predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reinvent-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-column", default="parent_id")
    parser.add_argument("--reference-column", default="original_main_pred_logd")
    parser.add_argument("--reinvent-column", default="main_logD_target_window (raw)")
    parser.add_argument("--absolute-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    observed = pd.read_csv(args.reinvent_output)
    needed_reference = {args.id_column, args.reference_column}
    needed_observed = {args.id_column, args.reinvent_column}
    if missing := needed_reference.difference(reference.columns):
        raise ValueError(f"Reference missing columns: {sorted(missing)}")
    if missing := needed_observed.difference(observed.columns):
        raise ValueError(f"REINVENT output missing columns: {sorted(missing)}")

    paired = reference[[args.id_column, args.reference_column]].merge(
        observed[[args.id_column, args.reinvent_column]],
        on=args.id_column,
        how="inner",
        validate="one_to_one",
    )
    paired["absolute_difference"] = (
        paired[args.reference_column] - paired[args.reinvent_column]
    ).abs()
    max_error = float(paired["absolute_difference"].max()) if len(paired) else float("inf")
    passed = bool(len(paired) == len(reference) == len(observed) and max_error <= args.absolute_tolerance)
    report = {
        "passed": passed,
        "paired_rows": int(len(paired)),
        "reference_rows": int(len(reference)),
        "reinvent_rows": int(len(observed)),
        "absolute_tolerance": args.absolute_tolerance,
        "max_absolute_error": max_error,
        "mean_absolute_error": float(paired["absolute_difference"].mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
