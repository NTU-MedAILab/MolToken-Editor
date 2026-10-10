#!/usr/bin/env python3
"""Render and validate the pre-registered REINVENT4 configuration."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path


PLACEHOLDERS = {
    "__DEVICE__": "device",
    "__TB_LOGDIR__": "tb_logdir",
    "__JSON_CONFIG__": "json_config",
    "__SUMMARY_PREFIX__": "summary_prefix",
    "__MOL2MOL_PRIOR__": "mol2mol_prior",
    "__SEED_SMILES_FILE__": "seed_smiles_file",
    "__CHECKPOINT_FILE__": "checkpoint_file",
    "__CHEMPROP_ENSEMBLE_DIR__": "chemprop_ensemble_dir",
}
FORBIDDEN_COMPONENTS = (
    "QED",
    "SAScore",
    "TanimotoSimilarity",
    "custom_alerts",
    "XGBoost",
    "RandomForest",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tb-logdir", type=Path, required=True)
    parser.add_argument("--json-config", type=Path, required=True)
    parser.add_argument("--summary-prefix", type=Path, required=True)
    parser.add_argument("--mol2mol-prior", type=Path, required=True)
    parser.add_argument("--seed-smiles-file", type=Path, required=True)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--chemprop-ensemble-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-missing-paths",
        action="store_true",
        help="Syntax-only smoke test; forbidden for a formal run.",
    )
    args = parser.parse_args()

    values = {
        "device": args.device,
        "tb_logdir": str(args.tb_logdir.resolve()),
        "json_config": str(args.json_config.resolve()),
        "summary_prefix": str(args.summary_prefix.resolve()),
        "mol2mol_prior": str(args.mol2mol_prior.resolve()),
        "seed_smiles_file": str(args.seed_smiles_file.resolve()),
        "checkpoint_file": str(args.checkpoint_file.resolve()),
        "chemprop_ensemble_dir": str(args.chemprop_ensemble_dir.resolve()),
    }
    if not args.allow_missing_paths:
        checks = {
            "Mol2Mol prior": args.mol2mol_prior.is_file(),
            "seed SMILES": args.seed_smiles_file.is_file(),
            "Chemprop ensemble": args.chemprop_ensemble_dir.is_dir(),
        }
        failed = [name for name, okay in checks.items() if not okay]
        if failed:
            raise FileNotFoundError("Missing required inputs: " + ", ".join(failed))
        pt_files = sorted(args.chemprop_ensemble_dir.glob("*.pt"))
        if len(pt_files) != 10:
            raise ValueError(
                f"Formal oracle directory must contain 10 .pt files; found {len(pt_files)}"
            )
        seed_count = sum(
            1
            for line in args.seed_smiles_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if seed_count != 88:
            raise ValueError(f"Formal seed file must contain 88 SMILES; found {seed_count}")

    rendered = args.template.read_text(encoding="utf-8")
    for placeholder, key in PLACEHOLDERS.items():
        rendered = rendered.replace(placeholder, values[key].replace("\\", "/"))
    leftovers = [token for token in PLACEHOLDERS if token in rendered]
    if leftovers:
        raise ValueError("Unresolved placeholders: " + ", ".join(leftovers))
    for component in FORBIDDEN_COMPONENTS:
        marker = f"[stage.scoring.component.{component}]"
        if marker in rendered:
            raise ValueError(f"Forbidden reward component detected: {component}")

    parsed = tomllib.loads(rendered)
    parameters = parsed["parameters"]
    stage = parsed["stage"][0]
    components = stage["scoring"]["component"]
    seed_path = Path(values["seed_smiles_file"])
    seed_count = (
        sum(1 for line in seed_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if seed_path.is_file()
        else 88
    )
    raw_budget = seed_count * parameters["batch_size"] * stage["max_steps"]
    if raw_budget > 4446:
        raise ValueError("REINVENT raw generation budget exceeds the locked MolToken budget")
    if stage["min_steps"] != stage["max_steps"]:
        raise ValueError("Formal experiment must use a fixed number of RL steps")
    if parameters["randomize_smiles"] is not False:
        raise ValueError("The MMP prior requires randomize_smiles=false")
    if "unique_sequences" in parameters:
        raise ValueError("REINVENT4 v4.8 RL rejects the unique_sequences field")
    if len(components) != 1 or "MolTokenChemProp2" not in components[0]:
        raise ValueError("Exactly one MolTokenChemProp2 scoring component is required")
    endpoint = components[0]["MolTokenChemProp2"]["endpoint"]
    if len(endpoint) != 1 or endpoint[0]["weight"] != 1.0:
        raise ValueError("The sole logD endpoint must have weight 1.0")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    summary = {
        "output": str(args.output.resolve()),
        "device": args.device,
        "parent_count": seed_count,
        "samples_per_parent_per_step": parameters["batch_size"],
        "steps": stage["max_steps"],
        "max_raw_sequences": raw_budget,
        "objective_component_count": len(components),
        "objective": endpoint[0]["name"],
        "formal_path_checks_skipped": bool(args.allow_missing_paths),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
