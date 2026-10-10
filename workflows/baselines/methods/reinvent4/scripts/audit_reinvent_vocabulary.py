#!/usr/bin/env python3
"""Audit every parent SMILES against the selected REINVENT generator vocabulary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from reinvent.runmodes import create_adapter
from reinvent.utils import get_tokens_from_vocabulary
from reinvent.utils.config_parse import find_invalid_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--parent-table", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-parents", type=int, default=88)
    parser.add_argument("--allow-incompatible", action="store_true")
    args = parser.parse_args()

    parents = pd.read_csv(args.parent_table)
    required = {"parent_id", "parent_smiles"}
    missing = required.difference(parents.columns)
    if missing:
        raise ValueError("Missing parent-table columns: " + ", ".join(sorted(missing)))

    adapter, _, model_type = create_adapter(str(args.prior.resolve()), "inference", torch.device("cpu"))
    allowed, second = get_tokens_from_vocabulary(adapter.vocabulary)
    if model_type != "Mol2Mol" or second:
        raise ValueError(f"Expected a single-vocabulary Mol2Mol prior; got {model_type}")

    rows = []
    for row in parents.itertuples(index=False):
        smiles = str(row.parent_smiles)
        invalid = sorted(find_invalid_tokens(smiles, allowed))
        rows.append(
            {
                "parent_id": str(row.parent_id),
                "parent_smiles": smiles,
                "compatible": not invalid,
                "unsupported_tokens": ";".join(invalid),
            }
        )
    audit = pd.DataFrame(rows)
    compatible = audit.loc[audit["compatible"]].copy()
    incompatible = audit.loc[~audit["compatible"]].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / "phase24_reinvent_vocabulary_audit.csv"
    compatible_path = args.out_dir / "phase24_reinvent_compatible_seeds.smi"
    manifest_path = args.out_dir / "phase24_reinvent_vocabulary_manifest.json"
    audit.to_csv(audit_path, index=False)
    compatible_path.write_text(
        "\n".join(compatible["parent_smiles"].tolist()) + "\n", encoding="utf-8"
    )
    manifest = {
        "prior": str(args.prior.resolve()),
        "model_type": model_type,
        "total": int(len(audit)),
        "compatible": int(len(compatible)),
        "incompatible": int(len(incompatible)),
        "coverage": float(len(compatible) / len(audit)) if len(audit) else 0.0,
        "incompatible_parent_ids": incompatible["parent_id"].tolist(),
        "audit_table": str(audit_path.resolve()),
        "compatible_seed_file": str(compatible_path.resolve()),
        "expected_parents": args.expected_parents,
        "formal_full_set_compatible": bool(
            len(audit) == args.expected_parents and len(incompatible) == 0
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if len(incompatible) and not args.allow_incompatible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
