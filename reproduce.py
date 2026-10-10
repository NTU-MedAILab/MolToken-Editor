#!/usr/bin/env python3
"""Unified entry point for the experiments reported in the manuscript."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parent
WORKFLOWS = REPO / "workflows"


@dataclass(frozen=True)
class Step:
    label: str
    script: str
    args: tuple[str, ...] = ()

    def command(self, batch_size: int) -> list[str]:
        values = tuple(
            value.format(repo=REPO, workflows=WORKFLOWS, batch_size=batch_size)
            for value in self.args
        )
        return [sys.executable, str(WORKFLOWS / self.script), *values]


PREDICTORS = (
    Step("descriptor baselines", "predictors/scripts/train_descriptor_models.py"),
    Step("scaffold split", "predictors/scripts/build_scaffold_split.py"),
    Step(
        "D-MPNN ensembles",
        "predictors/scripts/run_chemprop_experiments.py",
        ("--variants", "dmpnn_rdkit2d", "--split-types", "random", "scaffold", "--epochs", "50", "--batch-size", "{batch_size}"),
    ),
    Step(
        "predictor metrics",
        "predictors/scripts/collect_chemprop_results.py",
        ("--variants", "dmpnn_rdkit2d", "--split-types", "random", "scaffold", "--batch-size", "{batch_size}"),
    ),
    Step("auxiliary models", "predictors/scripts/train_verifier_models.py"),
    Step("predictor checks", "predictors/scripts/validate_outputs.py"),
)


MOLTOKEN = (
    Step("external-library inventory", "moltoken/scripts/inventory_libraries.py"),
    Step("parent screening", "moltoken/scripts/screen_parents.py", ("--force-predict", "--batch-size", "{batch_size}")),
    Step("counterfactual attribution", "moltoken/scripts/localize_moltokens.py", ("--force-predict",)),
    Step("definition audit", "moltoken/scripts/audit_definition.py"),
    Step("boundary refinement", "moltoken/scripts/refine_boundaries.py"),
    Step("edit generation", "moltoken/scripts/generate_edits.py", ("--force-predict",)),
    Step("auxiliary-model gating", "moltoken/scripts/apply_verifier_consensus.py"),
    Step(
        "evidence completion",
        "moltoken/scripts/score_additional_edits.py",
        ("--force-predict",),
    ),
    Step("evidence tables", "moltoken/scripts/build_evidence_tables.py"),
    Step("validation selection", "moltoken/scripts/select_validation_list.py"),
    Step("candidate evaluation", "moltoken/scripts/evaluate_candidates.py"),
)


LOCALIZATION = (
    Step("localization targets", "localization/scripts/build_targets.py", ("--project-root", "{workflows}")),
    Step("normalize targets", "localization/scripts/normalize_targets.py"),
    Step("lock common parents", "localization/scripts/lock_common_parents.py"),
    Step("generate comparison candidates", "localization/scripts/generate_candidates.py", ("--comparison-name", "common_budget2")),
    Step("clean candidates", "localization/scripts/clean_candidates.py"),
    Step("score candidates", "localization/scripts/score_candidates.py", ("--comparison-name", "common_budget2_safe", "--force-predict")),
    Step("bootstrap localization comparison", "localization/scripts/bootstrap_statistics.py"),
    Step("format localization tables", "localization/scripts/build_tables.py"),
)


def ablation_commands() -> list[tuple[str, list[str]]]:
    output = REPO / "runs" / "ablation"
    return [
        (
            "component ablations",
            [
                sys.executable,
                str(WORKFLOWS / "ablation/scripts/reproduce.py"),
                "--output-dir",
                str(output),
            ],
        )
    ]


def equal_budget_commands() -> list[tuple[str, list[str]]]:
    root = WORKFLOWS / "baselines"
    commands: list[tuple[str, list[str]]] = []
    for seed in (20260703, 20260704, 20260705):
        commands.append(
            (
                f"equal-budget comparison seed {seed}",
                [
                    sys.executable,
                    str(root / "scripts/compare_three_methods.py"),
                    "--p26-root",
                    str(root),
                    "--seed",
                    str(seed),
                ],
            )
        )
    commands.append(
        (
            "aggregate three seeds",
            [
                sys.executable,
                str(root / "scripts/aggregate_three_seeds.py"),
                "--p26-root",
                str(root),
            ],
        )
    )
    return commands


def attribution_control_commands() -> list[tuple[str, list[str]]]:
    return [
        (
            "A05 matched-control locked-evidence recalculation",
            [
                sys.executable,
                str(
                    WORKFLOWS
                    / "attribution_control"
                    / "scripts"
                    / "recalculate_locked_evidence.py"
                ),
            ],
        )
    ]


def execute(commands: list[tuple[str, list[str]]], dry_run: bool) -> None:
    for index, (label, command) in enumerate(commands, start=1):
        print(f"[{index:02d}/{len(commands):02d}] {label}\n  {shlex.join(command)}")
        if not Path(command[1]).is_file():
            raise FileNotFoundError(command[1])
        if not dry_run:
            subprocess.run(command, cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "validate",
            "predictors",
            "moltoken",
            "localization",
            "ablation",
            "equal-budget",
            "attribution-control",
            "all",
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands: list[tuple[str, list[str]]] = []
    if args.stage in {"validate", "all"}:
        commands.append(("repository validation", [sys.executable, str(REPO / "validate.py")]))
    if args.stage in {"predictors", "all"}:
        commands.extend((step.label, step.command(args.batch_size)) for step in PREDICTORS)
    if args.stage in {"moltoken", "all"}:
        commands.extend((step.label, step.command(args.batch_size)) for step in MOLTOKEN)
    if args.stage in {"localization", "all"}:
        commands.extend((step.label, step.command(args.batch_size)) for step in LOCALIZATION)
    if args.stage in {"ablation", "all"}:
        commands.extend(ablation_commands())
    if args.stage in {"equal-budget", "all"}:
        commands.extend(equal_budget_commands())
    if args.stage == "attribution-control":
        commands.extend(attribution_control_commands())
    execute(commands, args.dry_run)


if __name__ == "__main__":
    main()
