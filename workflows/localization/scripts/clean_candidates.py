#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "localization/tables"
REPORTS = ROOT / "localization/reports"


def bool_value(x):
    return str(x).strip().lower() in {"true", "1", "yes", "y", "pass"}


def chemprop_safe_smiles(smi):
    if smi is None or pd.isna(smi):
        return False, "missing_smiles"

    smi = str(smi).strip()
    if not smi:
        return False, "empty_smiles"

    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception:
        return False, "rdkit_parse_exception"

    if mol is None:
        return False, "rdkit_parse_failed"

    if mol.GetNumAtoms() == 0:
        return False, "zero_atom_molecule"

    if len(Chem.GetMolFrags(mol)) != 1:
        return False, "multi_fragment_smiles"

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            return False, "contains_dummy_atom"
        if atom.GetAtomicNum() <= 0:
            return False, "invalid_atomic_number"

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False, "sanitize_failed"

    try:
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return False, "canonicalize_failed"

    if not can:
        return False, "canonical_empty"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(TABLES / "phase19b_common_budget2_candidates_all.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(TABLES / "phase19b_common_budget2_candidates_chemprop_safe.csv"),
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if not inp.exists():
        raise FileNotFoundError(f"Missing input: {inp}")

    df = pd.read_csv(inp, low_memory=False)

    safe_flags = []
    safe_reasons = []

    for smi in df.get("counterfactual_smiles", pd.Series([None] * len(df))):
        ok, reason = chemprop_safe_smiles(smi)
        safe_flags.append(ok)
        safe_reasons.append(reason)

    df["chemprop_safe_smiles"] = safe_flags
    df["chemprop_safe_reason"] = safe_reasons

    original_valid = df["edit_valid"].map(bool_value) if "edit_valid" in df.columns else pd.Series([False] * len(df))

    unsafe_valid = original_valid & (~df["chemprop_safe_smiles"])

    if "edit_reason" not in df.columns:
        df["edit_reason"] = ""

    df.loc[unsafe_valid, "edit_reason"] = (
        "chemprop_unsafe_" + df.loc[unsafe_valid, "chemprop_safe_reason"].astype(str)
    )
    df.loc[unsafe_valid, "edit_valid"] = False

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    report = {
        "phase": "19B2_chemprop_safe_candidate_cleaning",
        "status": "pass",
        "input": str(inp),
        "output": str(out),
        "rows": int(len(df)),
        "original_valid_edits": int(original_valid.sum()),
        "unsafe_valid_downgraded": int(unsafe_valid.sum()),
        "final_valid_edits": int(df["edit_valid"].map(bool_value).sum()),
        "chemprop_safe_reason_counts": df["chemprop_safe_reason"].value_counts(dropna=False).to_dict(),
        "unsafe_valid_by_reason": df.loc[unsafe_valid, "chemprop_safe_reason"].value_counts(dropna=False).to_dict(),
        "valid_by_method_group_action_after_cleaning": (
            df[df["edit_valid"].map(bool_value)]
            .groupby(["method_group", "edit_action"])
            .size()
            .reset_index(name="valid_edits")
            .to_dict(orient="records")
        ),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS / "phase19b_common_budget2_chemprop_safe_cleaning_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("[WROTE]", out)
    print("[WROTE]", report_path)


if __name__ == "__main__":
    main()
