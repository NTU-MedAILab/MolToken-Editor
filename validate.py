#!/usr/bin/env python3
"""Validate repository structure and manuscript-facing numerical evidence."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
ERRORS: list[str] = []


def require(relative: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        ERRORS.append(f"missing required file: {relative}")
    return path


def close(actual: object, expected: float, label: str, tolerance: float = 1e-12) -> None:
    try:
        passed = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        passed = False
    if not passed:
        ERRORS.append(f"{label}: expected {expected}, observed {actual}")


def check_layout() -> None:
    allowed = {
        ".git", ".gitattributes", ".gitignore", "__pycache__", "CITATION.cff", "LICENSE_NOTICE.md",
        "README.md", "data", "environment", "reproduce.py", "results",
        "validate.py", "workflows",
    }
    unexpected = sorted(path.name for path in ROOT.iterdir() if path.name not in allowed)
    if unexpected:
        ERRORS.append(f"unexpected top-level entries: {unexpected}")
    numbered_prefixes = {f"{index:02d}_" for index in range(100)}
    for path in ROOT.rglob("*"):
        if path.is_dir() and path.name[:3] in numbered_prefixes:
            ERRORS.append(f"number-prefixed directory retained: {path.relative_to(ROOT)}")
        if path.is_dir() and path.name.lower() == "docs":
            ERRORS.append(f"docs directory retained: {path.relative_to(ROOT)}")


def check_sources() -> None:
    for path in ROOT.rglob("*.py"):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except Exception as exc:
            ERRORS.append(f"invalid Python {path.relative_to(ROOT)}: {exc}")
    for path in ROOT.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            ERRORS.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
    suffixes = {".py", ".json", ".sh", ".toml", ".yml", ".md"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes or path == ROOT / "validate.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "/home/" in text or "/Users/" in text or "04_运行代码" in text:
            ERRORS.append(f"machine-specific path retained: {path.relative_to(ROOT)}")



def check_reproduction_dependencies() -> None:
    """Check internal files and imports required by the reproduction entry points."""

    required_files = [
        "workflows/moltoken/scripts/contribution_quality.py",
        "workflows/moltoken/configs/quality_scoring.json",
        "workflows/moltoken/scripts/localize_moltokens.py",
        "workflows/moltoken/scripts/generate_edits.py",
        "workflows/moltoken/scripts/score_additional_edits.py",
        "workflows/localization/scripts/base_scoring.py",
        "workflows/localization/scripts/build_targets.py",
        "workflows/localization/scripts/normalize_targets.py",
        "workflows/localization/scripts/generate_candidates.py",
        "workflows/localization/scripts/score_candidates.py",
        (
            "workflows/attribution_control/scripts/"
            "recalculate_locked_evidence.py"
        ),
        (
            "workflows/attribution_control/scripts/"
            "recalculate_locked_evidence.py"
        ),
    ]

    for relative in required_files:
        require(relative)

    a05_script = (
        ROOT
        / "workflows"
        / "attribution_control"
        / "scripts"
        / "recalculate_locked_evidence.py"
    )
    if a05_script.is_file():
        completed = subprocess.run(
            [sys.executable, str(a05_script)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            ERRORS.append(
                "A05 locked-evidence recalculation failed: "
                + detail[-1000:]
            )
        else:
            try:
                report = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                ERRORS.append(
                    f"A05 recalculation returned invalid JSON: {exc}"
                )
            else:
                if report.get("status") != "PASS":
                    ERRORS.append(
                        "A05 locked-evidence recalculation did not PASS"
                    )
                if report.get("mode") != "locked-evidence recalculation":
                    ERRORS.append(
                        "A05 reproduction mode differs from the lock"
                    )

    stale_markers = [
        "05_moltoken_validation",
        "run_phase10d_edit_generation_scoring.py",
        "run_phase19c_baseline_scoring.py",
        "phasemoltoken_config.json",
        "results/01_internal_validation",
    ]

    workflows = ROOT / "workflows"

    for source in workflows.rglob("*"):
        if not source.is_file():
            continue
        if source.suffix.lower() not in {
            ".py",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
        }:
            continue

        content = source.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        for marker in stale_markers:
            if marker in content:
                ERRORS.append(
                    "stale reproduction reference "
                    f"{marker!r} in {source.relative_to(ROOT)}"
                )

    import_smoke_scripts = [
        "workflows/moltoken/scripts/contribution_quality.py",
        "workflows/moltoken/scripts/localize_moltokens.py",
        "workflows/moltoken/scripts/generate_edits.py",
        "workflows/moltoken/scripts/score_additional_edits.py",
        "workflows/localization/scripts/base_scoring.py",
        "workflows/localization/scripts/generate_candidates.py",
        "workflows/localization/scripts/score_candidates.py",
    ]

    for relative in import_smoke_scripts:
        source = ROOT / relative

        if not source.is_file():
            continue

        command = [
            sys.executable,
            "-c",
            (
                "import runpy; "
                f"runpy.run_path({str(source)!r}, "
                "run_name='reproducibility_import_audit')"
            ),
        ]

        try:
            result = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            ERRORS.append(
                f"import smoke test timed out: {relative}"
            )
            continue

        if result.returncode != 0:
            details = (
                result.stderr.strip()
                or result.stdout.strip()
                or f"exit code {result.returncode}"
            )
            last_line = details.splitlines()[-1]

            ERRORS.append(
                f"import smoke test failed: "
                f"{relative}: {last_line}"
            )

def check_data() -> None:
    raw = pd.read_csv(require("data/lipophilicity/raw/lipophilicity.csv"))
    clean = pd.read_csv(require("data/lipophilicity/processed/lipophilicity_clean.csv"))
    membership = pd.read_csv(require("data/external/library_membership.csv"))
    if len(raw) != 4200:
        ERRORS.append(f"raw Lipophilicity row count: {len(raw)}")
    if len(clean) != 4200:
        ERRORS.append(f"clean Lipophilicity row count: {len(clean)}")
    if len(membership) != 3000 or membership["library_id"].nunique() != 100:
        ERRORS.append("external-library membership is not 3,000 rows across 100 libraries")


def check_results() -> None:
    predictor = pd.read_csv(require("results/manuscript_tables/predictor_performance.csv"))
    if len(predictor) != 6:
        ERRORS.append(f"predictor table row count: {len(predictor)}")
    xgb_random = predictor[(predictor["model"] == "XGBoost") & (predictor["split"] == "random")].iloc[0]
    rf_scaffold = predictor[(predictor["model"] == "Random Forest") & (predictor["split"] == "scaffold")].iloc[0]
    close(xgb_random["r2_mean"], 0.6703531868732713, "XGBoost random R2")
    close(rf_scaffold["r2_mean"], 0.4510451723291585, "Random Forest scaffold R2")

    stages = pd.read_csv(require("results/manuscript_tables/moltoken_stage_counts.csv"))
    stage_map = dict(zip(stages["metric"], stages["value"]))
    for metric, expected in {
        "candidate_regions": 4711,
        "valid_counterfactuals": 4476,
        "active_moltokens": 234,
        "selected_parents": 120,
    }.items():
        close(stage_map.get(metric), expected, f"stage count {metric}")

    localization = pd.read_csv(
        require("results/manuscript_tables/localization_comparison.csv")
    ).set_index("method")

    expected_localization = {
        "MolToken-Editor": (
            "moltoken_editor", 4.89, 1.282, 89.6, 72.8, 0.804
        ),
        "BRICS": (
            "brics_edit", 4.91, 1.201, 82.2, 67.9, 0.766
        ),
        "RECAP": (
            "recap_edit", 4.80, 1.209, 84.6, 68.7, 0.776
        ),
        "Bemis–Murcko side-chain": (
            "rgroup_enumeration", 4.89, 1.228, 77.3, 64.6, 0.754
        ),
        "Action-matched random": (
            "random_action_matched", 4.72, 1.121, 76.4, 62.7, 0.734
        ),
    }

    if set(localization.index) != set(expected_localization):
        ERRORS.append(
            "common-81 localization method set differs from the manuscript"
        )

    for method, expected in expected_localization.items():
        if method not in localization.index:
            continue

        row = localization.loc[method]
        group, retained, delta, target, strict, score = expected

        if row["method_group"] != group:
            ERRORS.append(
                f"{method} method_group differs from the common-81 lock"
            )

        close(
            row["common_evaluable_starting_molecules"],
            81,
            f"{method} common-81 cohort",
        )
        close(
            row["mean_retained_candidates"],
            retained,
            f"{method} retained candidates",
        )
        close(
            row["mean_delta_distance"],
            delta,
            f"{method} delta distance",
        )
        close(
            row["target_window_success_percent"],
            target,
            f"{method} target-window success",
        )
        close(
            row["strict_success_percent"],
            strict,
            f"{method} strict success",
        )
        close(
            row["composite_edit_score"],
            score,
            f"{method} composite score",
        )

    common81_parent = pd.read_csv(
        require(
            "results/figure_source_data/"
            "localization_common81_parent_level.csv"
        )
    )

    if len(common81_parent) != 405:
        ERRORS.append(
            "common-81 parent-level source must contain 405 rows"
        )

    expected_groups = {
        values[0]: values[1:]
        for values in expected_localization.values()
    }

    if set(common81_parent["method_group"]) != set(expected_groups):
        ERRORS.append(
            "common-81 parent-level method groups differ from the lock"
        )

    for group, expected in expected_groups.items():
        subset = common81_parent[
            common81_parent["method_group"] == group
        ]

        if len(subset) != 81 or subset["mol_id"].nunique() != 81:
            ERRORS.append(
                f"{group} does not contain exactly 81 unique parents"
            )
            continue

        retained, delta, target, strict, score = expected
        close(
            round(float(subset["retained_candidates"].mean()), 2),
            retained,
            f"{group} parent-level retained candidates",
        )
        close(
            round(float(subset["mean_delta_distance"].mean()), 3),
            delta,
            f"{group} parent-level delta distance",
        )
        close(
            round(float(subset["target_window_success"].mean() * 100), 1),
            target,
            f"{group} parent-level target-window success",
        )
        close(
            round(float(subset["strict_success"].mean() * 100), 1),
            strict,
            f"{group} parent-level strict success",
        )
        close(
            round(float(subset["composite_edit_score"].mean()), 3),
            score,
            f"{group} parent-level composite score",
        )

    require(
        "results/figure_source_data/"
        "localization_common81_paired_bootstrap.csv"
    )
    require(
        "results/provenance/a01/"
        "table2_common81_ci_resolution.csv"
    )
    require(
        "results/provenance/a01/"
        "table2_common81_qualitative_ci_check.csv"
    )
    require(
        "results/provenance/pre_a01/"
        "localization_comparison_non_common81.csv"
    )

    for relative_path in (
        "results/provenance/a02/A02_FINAL_AUDIT.json",
        "results/provenance/a02/A02_MANUSCRIPT_CORRECTION_REGISTER.csv",
        "results/provenance/a02/A02_MMP_RESOLUTION.md",
        "results/provenance/a03/A03_FINAL_AUDIT.json",
        "results/provenance/a03/A03_set_flow.csv",
        "results/provenance/a03/A03_set_relations.json",
        "results/provenance/a03/A03_independent100_independence.json",
        "results/provenance/a03/A03_exact_structure_overlap.csv",
        "results/provenance/a03/A03_scaffold_overlap.csv",
        "results/provenance/a04/A04_FINAL_AUDIT.json",
        "results/provenance/a04/A04_budget_protocol_decision.json",
        "results/provenance/a04/A04_final_external88_budget_ledger.csv",
        "results/provenance/a04/A04_method_level_budget_summary.csv",
        "results/provenance/a04/A04_MANUSCRIPT_CORRECTION_REGISTER.csv",
    ):
        require(relative_path)

    budget = pd.read_csv(
        require("results/manuscript_tables/equal_budget_comparison.csv")
    ).set_index("method_label")

    expected_budget = {
        "MolToken-Editor": (
            0.7954545454545454,
            0.5795454545454546,
            1,
            0.0,
        ),
        "REINVENT4": (
            0.5265151515151515,
            0.4242424242424242,
            3,
            0.6666666666666666,
        ),
        "MMP-Transformer": (
            0.662878787878788,
            0.5151515151515151,
            3,
            0.0,
        ),
    }

    if set(budget.index) != set(expected_budget):
        ERRORS.append(
            "Exact88 B=5 method set differs from the manuscript lock"
        )

    for method, expected_values in expected_budget.items():
        if method not in budget.index:
            continue

        target, strict, seed_runs, missing_slots = expected_values
        row = budget.loc[method]

        if row["protocol"] != "oracle_independent_B5_any_hit":
            ERRORS.append(
                f"{method} protocol differs from the frozen B=5 protocol"
            )

        close(row["k"], 5, f"{method} requested evaluation slots")
        close(row["n_seed_runs"], seed_runs, f"{method} seed runs")
        close(
            row["parent_target_hit_rate_mean"],
            target,
            f"{method} parent target",
        )
        close(
            row["parent_strict_hit_rate_mean"],
            strict,
            f"{method} parent strict",
        )
        close(
            row["total_missing_generation_slots_mean"],
            missing_slots,
            f"{method} mean missing generation slots",
        )

    if "MMP-Transformer" in budget.index:
        mmp = budget.loc["MMP-Transformer"]
        close(
            round(float(mmp["parent_target_hit_rate_mean"]) * 100, 2),
            66.29,
            "MMP overall target percent",
        )
        close(
            round(float(mmp["parent_target_hit_rate_sd"]) * 100, 2),
            4.59,
            "MMP overall target SD percent",
        )

    direction = pd.read_csv(
        require("results/manuscript_tables/equal_budget_by_direction.csv")
    )
    direction = direction[
        direction["method_label"] == "MMP-Transformer"
    ].set_index("direction")

    expected_direction = {
        "increase_logd": (66, 0.7121212121212123, 6.60),
        "decrease_logd": (22, 0.5151515151515151, 5.25),
    }

    if set(direction.index) != set(expected_direction):
        ERRORS.append(
            "MMP direction set differs from the A02 manuscript lock"
        )

    for name, expected_values in expected_direction.items():
        if name not in direction.index:
            continue

        parents, target_rate, displayed_sd = expected_values
        row = direction.loc[name]

        close(row["parents"], parents, f"MMP {name} parent count")
        close(
            row["parent_target_hit_rate_mean"],
            target_rate,
            f"MMP {name} target rate",
        )
        close(
            round(float(row["parent_target_hit_rate_sd"]) * 100, 2),
            displayed_sd,
            f"MMP {name} target SD percent",
        )

    effects = pd.read_csv(require("results/figure_source_data/localization_ablation_effects.csv"))
    random_parent = effects[(effects["variant"] == "random_localization") & (effects["metric"] == "parent_target_window_rate")].iloc[0]
    close(random_parent["difference"], -0.32954545454545453, "random localization parent effect")

    gates = pd.read_csv(require("results/figure_source_data/gate_ablation_summary.csv")).set_index("variant")
    for variant, expected_count in {
        "full_all_gates": 131,
        "without_verifier_consensus": 178,
        "without_evidence_safety_gate": 253,
        "property_window_only": 340,
    }.items():
        close(gates.loc[variant, "accepted_candidates"], expected_count, f"{variant} accepted candidates")


def check_release_manifest() -> None:
    manifest_relative = "results/RELEASE_MANIFEST.sha256"
    manifest = require(manifest_relative)
    if not manifest.is_file():
        return

    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError:
            ERRORS.append(
                f"invalid release manifest line {line_number}: {line!r}"
            )
            continue
        if relative in entries:
            ERRORS.append(f"duplicate release manifest entry: {relative}")
            continue
        entries[relative] = digest

    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )

    if completed.returncode == 0:
        expected_paths = {
            raw.decode("utf-8")
            for raw in completed.stdout.split(b"\0")
            if raw
        }
    else:
        expected_paths = {
            str(candidate.relative_to(ROOT))
            for candidate in ROOT.rglob("*")
            if candidate.is_file()
            and ".git" not in candidate.relative_to(ROOT).parts
            and "__pycache__" not in candidate.relative_to(ROOT).parts
            and candidate.suffix != ".pyc"
        }

    expected_paths.discard(manifest_relative)

    missing_entries = sorted(expected_paths - set(entries))
    unexpected_entries = sorted(set(entries) - expected_paths)
    if missing_entries:
        ERRORS.append(
            "release manifest missing files: " + repr(missing_entries)
        )
    if unexpected_entries:
        ERRORS.append(
            "release manifest has unexpected files: "
            + repr(unexpected_entries)
        )

    for relative, expected_digest in entries.items():
        candidate = ROOT / relative
        if not candidate.is_file():
            ERRORS.append(f"manifest target is missing: {relative}")
            continue
        observed_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if observed_digest != expected_digest:
            ERRORS.append(
                f"release manifest hash mismatch: {relative}"
            )


def main() -> None:
    check_layout()
    check_sources()
    check_reproduction_dependencies()
    check_data()
    check_results()
    check_release_manifest()
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(ROOT).parts
        and "__pycache__" not in path.relative_to(ROOT).parts
    ]
    result = {
        "status": "PASS" if not ERRORS else "FAIL",
        "file_count": len(files),
        "errors": ERRORS,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if ERRORS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
