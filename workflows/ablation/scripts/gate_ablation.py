#!/usr/bin/env python3
"""Two-by-two decision-gate ablation for frozen MolToken-Editor candidates.

The candidate set is held fixed.  The evidence/safety gate and auxiliary
verifier-consensus gate are removed one at a time and together.  Property,
safety, and consensus fields are never used to reselect candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


VARIANT_ORDER = [
    "full_all_gates",
    "without_verifier_consensus",
    "without_evidence_safety_gate",
    "property_window_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen-candidate MolToken decision-gate ablation."
    )
    parser.add_argument("repo", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--budget", type=int, default=5)
    return parser.parse_args()


def canonicalize(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_bootstrap(
    parent_tables: dict[str, pd.DataFrame],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    full = parent_tables["full_all_gates"].set_index("parent_key")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(full), size=(replicates, len(full)))
    rows: list[dict[str, object]] = []
    for variant in VARIANT_ORDER[1:]:
        other = parent_tables[variant].set_index("parent_key").loc[full.index]
        for metric in ["parent_any_accept", "candidate_accept_rate_fixed_budget"]:
            difference = (
                other[metric].to_numpy(dtype=float)
                - full[metric].to_numpy(dtype=float)
            )
            sampled = difference[indices].mean(axis=1)
            rows.append(
                {
                    "variant": variant,
                    "reference": "full_all_gates",
                    "metric": metric,
                    "difference": difference.mean(),
                    "ci95_low": np.quantile(sampled, 0.025),
                    "ci95_high": np.quantile(sampled, 0.975),
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_path = (
        repo
        / "workflows/baselines/inputs/"
        "budget_lock/moltoken_editor/moltoken_b5_selected_scored.csv"
    )
    exact88_path = (
        repo
        / "workflows/baselines/inputs/"
        "common_exact88/phase26_locked_exact88.csv"
    )
    for path in [selected_path, exact88_path]:
        if not path.is_file():
            raise SystemExit(f"[FAIL] missing input: {path}")

    frame = pd.read_csv(selected_path, low_memory=False).copy()
    exact88 = pd.read_csv(exact88_path, low_memory=False).copy()
    required = {
        "canonical_smiles",
        "target_window_success",
        "evidence_ready",
        "consensus_success",
        "strict_success",
        "relaxed_guardrail_pass",
        "ood_risk",
        "edited_interference_alert",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"[FAIL] selected candidates missing fields: {missing}")

    frame["parent_key"] = frame["canonical_smiles"].map(canonicalize)
    exact88["parent_key"] = exact88["parent_smiles"].map(canonicalize)
    parent_order = exact88["parent_key"].tolist()
    if len(frame) != 440 or frame["parent_key"].nunique() != 88:
        raise SystemExit("[FAIL] selected candidate table is not 88 parents x 5")
    if set(frame["parent_key"]) != set(parent_order):
        raise SystemExit("[FAIL] selected candidate parents do not equal Exact88")

    property_hit = as_bool(frame["target_window_success"])
    evidence_gate = as_bool(frame["evidence_ready"])
    verifier_gate = as_bool(frame["consensus_success"])
    strict_frozen = as_bool(frame["strict_success"])

    decisions = {
        "full_all_gates": property_hit & evidence_gate & verifier_gate,
        "without_verifier_consensus": property_hit & evidence_gate,
        "without_evidence_safety_gate": property_hit & verifier_gate,
        "property_window_only": property_hit,
    }
    if not decisions["full_all_gates"].equals(strict_frozen):
        disagreement = int((decisions["full_all_gates"] != strict_frozen).sum())
        raise SystemExit(
            f"[FAIL] reconstructed full decision differs from strict_success "
            f"for {disagreement} candidates"
        )

    expected_counts = {
        "full_all_gates": (131, 51),
        "without_verifier_consensus": (178, 70),
        "without_evidence_safety_gate": (253, 66),
        "property_window_only": (340, 87),
    }

    candidate_long_parts: list[pd.DataFrame] = []
    parent_tables: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, object]] = []
    total_slots = len(parent_order) * args.budget
    full_decision = decisions["full_all_gates"]

    diagnostic_fields = {
        "evidence_ready_rate_among_accepted": evidence_gate,
        "verifier_consensus_rate_among_accepted": verifier_gate,
        "guardrail_pass_rate_among_accepted": as_bool(
            frame["relaxed_guardrail_pass"]
        ),
        "alert_free_rate_among_accepted": ~as_bool(
            frame["edited_interference_alert"]
        ),
        "ood_free_rate_among_accepted": ~as_bool(frame["ood_risk"]),
    }

    for variant in VARIANT_ORDER:
        accepted = decisions[variant]
        expected_candidate_count, expected_parent_count = expected_counts[variant]
        parent = (
            pd.DataFrame(
                {
                    "parent_key": frame["parent_key"],
                    "accepted": accepted.astype(int),
                }
            )
            .groupby("parent_key", sort=False)
            .agg(
                accepted_candidates=("accepted", "sum"),
                parent_any_accept=("accepted", "max"),
            )
            .reindex(parent_order)
            .reset_index()
        )
        parent["candidate_accept_rate_fixed_budget"] = (
            parent["accepted_candidates"] / args.budget
        )
        parent["variant"] = variant
        parent_tables[variant] = parent

        candidate_count = int(accepted.sum())
        parent_count = int(parent["parent_any_accept"].sum())
        if (candidate_count, parent_count) != (
            expected_candidate_count,
            expected_parent_count,
        ):
            raise SystemExit(
                f"[FAIL] {variant} expected "
                f"{expected_candidate_count}/{expected_parent_count}, observed "
                f"{candidate_count}/{parent_count}"
            )

        row: dict[str, object] = {
            "variant": variant,
            "accepted_candidates": candidate_count,
            "candidate_accept_rate_fixed_budget": candidate_count / total_slots,
            "parents_with_any_accept": parent_count,
            "parent_accept_rate": parent_count / len(parent_order),
            "additional_candidates_vs_full": int((accepted & ~full_decision).sum()),
            "additional_parents_vs_full": int(
                parent["parent_any_accept"].sum()
                - parent_tables["full_all_gates"]["parent_any_accept"].sum()
            ),
        }
        for label, diagnostic in diagnostic_fields.items():
            row[label] = float(diagnostic[accepted].mean()) if accepted.any() else np.nan
        summaries.append(row)

        candidate_part = frame[
            [
                column
                for column in [
                    "mol_id_selection",
                    "mol_id",
                    "canonical_smiles",
                    "edited_smiles",
                    "edit_action",
                    "target_window_success",
                    "evidence_ready",
                    "consensus_success",
                    "strict_success",
                    "relaxed_guardrail_pass",
                    "ood_risk",
                    "edited_interference_alert",
                    "parent_tanimoto",
                ]
                if column in frame.columns
            ]
        ].copy()
        candidate_part.insert(0, "variant", variant)
        candidate_part["accepted"] = accepted.to_numpy(dtype=bool)
        candidate_part["additional_vs_full"] = (
            accepted & ~full_decision
        ).to_numpy(dtype=bool)
        candidate_long_parts.append(candidate_part)

    summary = pd.DataFrame(summaries)
    parent_long = pd.concat(parent_tables.values(), ignore_index=True)
    candidate_long = pd.concat(candidate_long_parts, ignore_index=True)
    effects = paired_bootstrap(parent_tables, args.bootstrap, args.seed)

    outputs = {
        "gate_ablation_summary.csv": summary,
        "gate_ablation_effects_vs_full.csv": effects,
        "gate_ablation_parent_level.csv": parent_long,
        "gate_ablation_candidate_level.csv": candidate_long,
    }
    for name, data in outputs.items():
        data.to_csv(output_dir / name, index=False)

    manifest = {
        "status": "pass",
        "design": "2x2 removal of evidence/safety and verifier-consensus gates",
        "population": "Exact88",
        "candidate_set": "fixed frozen 88 x 5 MolToken-Editor candidates",
        "property_endpoint": "target_window_success",
        "variants": {
            "full_all_gates": "property AND evidence/safety AND verifier consensus",
            "without_verifier_consensus": "property AND evidence/safety",
            "without_evidence_safety_gate": "property AND verifier consensus",
            "property_window_only": "property only",
        },
        "interpretation": (
            "Increases after removing gates are not efficacy gains; they quantify "
            "candidates admitted without the removed reliability constraint."
        ),
        "bootstrap": {
            "unit": "parent molecule",
            "paired": True,
            "replicates": args.bootstrap,
            "seed": args.seed,
        },
        "inputs": {
            "selected_scored": {
                "path": str(selected_path),
                "sha256": sha256(selected_path),
            },
            "exact88": {"path": str(exact88_path), "sha256": sha256(exact88_path)},
        },
        "validated_counts": expected_counts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    package_path = output_dir / "moltoken_gate_ablation_results.tar.gz"
    package_names = list(outputs) + ["manifest.json"]
    with tarfile.open(package_path, "w:gz") as archive:
        for name in package_names:
            archive.add(output_dir / name, arcname=name)

    print(summary.to_string(index=False))
    print()
    print("[PASS] full gate logic exactly reproduces frozen strict_success")
    print("[PASS] validated all four candidate and parent acceptance counts")
    print("[OUTPUT]", output_dir)
    print("[PACKAGE]", package_path)
    print("[SHA256]", sha256(package_path))


if __name__ == "__main__":
    main()
