#!/usr/bin/env python3
"""Fail-fast preflight for the external single-logD baseline run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


LOCKED_PRIOR_SHA256 = "840dcd7c3ecf2dff999a93315524831e38ef0d0c802857a069d0d961f872f0b1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--vocabulary-audit", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--oracle-equivalence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-reinvent-import", action="store_true")
    args = parser.parse_args()

    with args.parent_manifest.open(encoding="utf-8") as handle:
        parents = json.load(handle)
    with args.oracle_manifest.open(encoding="utf-8") as handle:
        oracle = json.load(handle)
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    with args.vocabulary_audit.open(encoding="utf-8") as handle:
        vocabulary = json.load(handle)
    with args.oracle_equivalence.open(encoding="utf-8") as handle:
        equivalence = json.load(handle)

    errors = []
    if not parents.get("formal"):
        errors.append("parent manifest is marked as a smoke test")
    if parents.get("parent_count") != 88:
        errors.append(f"parent count is {parents.get('parent_count')}, expected 88")
    if parents.get("target_window") != [1.0, 3.4]:
        errors.append(f"parent target window is {parents.get('target_window')}, expected [1.0, 3.4]")
    if oracle.get("ensemble_size") != 10:
        errors.append(f"oracle ensemble size is {oracle.get('ensemble_size')}, expected 10")
    if oracle.get("chemprop_generation") != "v2":
        errors.append("oracle manifest is not marked as Chemprop v2")
    if oracle.get("molecule_featurizer") != "RDKit2D" or oracle.get(
        "molecule_feature_dimension"
    ) != 217:
        errors.append("oracle manifest does not lock raw RDKit2D(217) features")
    if vocabulary.get("total") != 88 or vocabulary.get("compatible") != 88:
        errors.append(
            "REINVENT vocabulary does not cover all 88 locked common-set parents"
        )
    if not equivalence.get("passed"):
        errors.append("REINVENT scorer did not reproduce the locked MolToken oracle predictions")
    if not args.prior.is_file():
        errors.append(f"Mol2Mol prior is missing: {args.prior}")
    elif args.prior.stat().st_size < 1_000_000:
        errors.append(
            "Mol2Mol prior is suspiciously small; it may be a Git-LFS pointer instead of weights"
        )
    elif sha256(args.prior) != LOCKED_PRIOR_SHA256:
        errors.append("Mol2Mol prior SHA256 differs from the pre-registered official prior")
    parameters = config["parameters"]
    stage = config["stage"][0]
    if parameters["batch_size"] != 1:
        errors.append("Mol2Mol batch_size must be 1 sample per parent per step")
    if parameters.get("randomize_smiles") is not False:
        errors.append("Mol2Mol randomize_smiles must be explicitly false")
    if "unique_sequences" in parameters:
        errors.append("unique_sequences is invalid in the REINVENT4 v4.8 RL schema")
    if stage["max_steps"] != 50 or stage.get("min_steps") != 50:
        errors.append("REINVENT must use exactly 50 fixed steps")
    if parameters["batch_size"] * parents.get("parent_count", 0) * stage["max_steps"] != 4400:
        errors.append("REINVENT raw-attempt budget is not exactly 4400")
    components = stage.get("scoring", {}).get("component", [])
    if len(components) != 1 or "MolTokenChemProp2" not in components[0]:
        errors.append("reward must contain exactly one MolTokenChemProp2 component")
    else:
        endpoints = components[0]["MolTokenChemProp2"].get("endpoint", [])
        if len(endpoints) != 1 or endpoints[0].get("weight") != 1.0:
            errors.append("the sole logD endpoint must have weight 1.0")
        elif endpoints[0].get("transform", {}).get("low") != 1.0 or endpoints[0].get(
            "transform", {}
        ).get("high") != 3.4:
            errors.append("logD reward window is not the locked [1.0, 3.4]")
    for forbidden_section in ("diversity_filter", "intrinsic_penalty", "inception"):
        if config.get(forbidden_section) is not None or stage.get(forbidden_section) is not None:
            errors.append(f"forbidden reward/learning section present: {forbidden_section}")

    reinvent_path = shutil.which("reinvent")
    if not args.skip_reinvent_import:
        sys.path.insert(0, str(args.plugin_dir.resolve()))
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import chemprop; from reinvent_plugins.components import comp_moltoken_chemprop2; "
                "assert chemprop.__version__ == '2.2.3'; print(chemprop.__version__)",
            ],
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "PYTHONPATH": str(args.plugin_dir.resolve())
                + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
            },
        )
        if probe.returncode != 0:
            errors.append("REINVENT4 Python import failed: " + probe.stderr.strip())
        if reinvent_path is None:
            errors.append("reinvent executable is not on PATH")

    report = {
        "ok": not errors,
        "errors": errors,
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "parent_manifest_sha256": sha256(args.parent_manifest),
        "oracle_manifest_sha256": sha256(args.oracle_manifest),
        "vocabulary_audit_sha256": sha256(args.vocabulary_audit),
        "oracle_equivalence_sha256": sha256(args.oracle_equivalence),
        "prior_sha256": sha256(args.prior) if args.prior.is_file() else None,
        "reinvent_executable": reinvent_path,
        "python": sys.executable,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
