#!/usr/bin/env python3
"""Reproduce the two component-ablation result packages from public inputs."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WORKFLOW = SCRIPT_DIR.parent
REPO = SCRIPT_DIR.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--budget", type=int, default=5)
    return parser.parse_args()


def expand(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rb") as input_handle, target.open("wb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, 1024 * 1024)


def run(command: list[str]) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="moltoken_ablation_") as temporary:
        archive = Path(temporary)
        table_dir = archive / "localization_frozen_all/tables"
        expand(
            WORKFLOW / "inputs/phase19b_common_budget2_candidates_all.csv.gz",
            table_dir / "phase19b_common_budget2_candidates_all.csv",
        )
        expand(
            WORKFLOW / "inputs/phase19c_common_budget2_safe_scored_all.csv.gz",
            table_dir / "phase19c_common_budget2_safe_scored_all.csv",
        )
        run([
            sys.executable,
            str(SCRIPT_DIR / "localization_ablation.py"),
            str(REPO),
            "--archive-root", str(archive),
            "--output-dir", str(output / "localization"),
            "--bootstrap", str(args.bootstrap),
            "--seed", str(args.seed),
            "--budget", str(args.budget),
        ])
        run([
            sys.executable,
            str(SCRIPT_DIR / "gate_ablation.py"),
            str(REPO),
            "--output-dir", str(output / "decision_gates"),
            "--bootstrap", str(args.bootstrap),
            "--seed", str(args.seed),
            "--budget", str(args.budget),
        ])


if __name__ == "__main__":
    main()
