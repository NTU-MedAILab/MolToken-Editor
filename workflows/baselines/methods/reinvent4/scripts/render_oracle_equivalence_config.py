#!/usr/bin/env python3
"""Render the REINVENT scoring-only oracle equivalence configuration."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path


PLACEHOLDERS = {
    "__DEVICE__": "device",
    "__JSON_CONFIG__": "json_config",
    "__PARENT_TABLE__": "parent_table",
    "__OUTPUT_CSV__": "output_csv",
    "__CHEMPROP_ENSEMBLE_DIR__": "chemprop_ensemble_dir",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-config", type=Path, required=True)
    parser.add_argument("--parent-table", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--chemprop-ensemble-dir", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()

    import pandas as pd

    parents = pd.read_csv(args.parent_table)
    required = {"parent_id", "parent_smiles"}
    if missing := required.difference(parents.columns):
        raise ValueError(f"Parent table missing columns: {sorted(missing)}")
    if not args.allow_smoke and len(parents) != 90:
        raise ValueError(f"Formal equivalence requires 90 parents; found {len(parents)}")
    models = sorted(args.chemprop_ensemble_dir.glob("*.pt"))
    if len(models) != 10:
        raise ValueError(f"Oracle directory must contain 10 .pt files; found {len(models)}")

    values = {
        "device": args.device,
        "json_config": str(args.json_config.resolve()),
        "parent_table": str(args.parent_table.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "chemprop_ensemble_dir": str(args.chemprop_ensemble_dir.resolve()),
    }
    rendered = args.template.read_text(encoding="utf-8")
    for placeholder, key in PLACEHOLDERS.items():
        rendered = rendered.replace(placeholder, values[key].replace("\\", "/"))
    leftovers = [token for token in PLACEHOLDERS if token in rendered]
    if leftovers:
        raise ValueError("Unresolved placeholders: " + ", ".join(leftovers))
    parsed = tomllib.loads(rendered)
    components = parsed["scoring"]["component"]
    if len(components) != 1 or "MolTokenChemProp2" not in components[0]:
        raise ValueError("Exactly one MolTokenChemProp2 component is required")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.json_config.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "parent_count": int(len(parents)),
                "model_count": len(models),
                "formal": bool(len(parents) == 90 and not args.allow_smoke),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
