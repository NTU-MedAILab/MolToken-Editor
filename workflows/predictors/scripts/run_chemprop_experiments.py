from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
PHASE01 = ROOT.parent / "data" / "lipophilicity"
PHASE04 = ROOT / "predictors"
CHEMPROP_CLI = PHASE04 / "scripts" / "chemprop_stable_cli.py"
MODELS = PHASE04 / "models" / "chemprop_main"
REPORTS = PHASE04 / "reports"
LOGS = REPORTS / "chemprop_run_logs"

DEFAULT_SEEDS = [20260528, 20260529, 20260530, 20260531, 20260601]
VARIANTS = {
    "dmpnn_rdkit2d": ["--molecule-featurizers", "rdkit_2d"],
}


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def split_paths(split_type: str) -> tuple[Path, Path, Path]:
    if split_type == "random":
        base = PHASE01 / "splits"
        return base / "chemprop_train.csv", base / "chemprop_valid.csv", base / "chemprop_test.csv"
    if split_type == "scaffold":
        base = PHASE04 / "splits"
        return (
            base / "chemprop_scaffold_train.csv",
            base / "chemprop_scaffold_valid.csv",
            base / "chemprop_scaffold_test.csv",
        )
    raise ValueError(f"unknown split_type: {split_type}")


def build_command(
    *,
    variant: str,
    split_type: str,
    seed: int,
    epochs: int,
    warmup_epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[list[str], Path]:
    train_path, valid_path, test_path = split_paths(split_type)
    output_dir = MODELS / variant / split_type / f"seed_{seed}"
    cmd = [
        str(PYTHON),
        str(CHEMPROP_CLI),
        "train",
        "-i",
        str(train_path),
        str(valid_path),
        str(test_path),
        "-o",
        str(output_dir),
        "-s",
        "smiles",
        "--target-columns",
        "exp",
        "--task-type",
        "regression",
        "--metrics",
        "rmse",
        "mae",
        "r2",
        "--tracking-metric",
        "rmse",
        "--epochs",
        str(epochs),
        "--warmup-epochs",
        str(warmup_epochs),
        "--patience",
        str(patience),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        "0",
        "--accelerator",
        "cpu",
        "--devices",
        "1",
        "--data-seed",
        str(seed),
        "--pytorch-seed",
        str(seed),
        "--depth",
        "3",
        "--message-hidden-dim",
        "300",
        "--ffn-hidden-dim",
        "300",
        "--ffn-num-layers",
        "2",
        "--dropout",
        "0.10",
        *VARIANTS[variant],
    ]
    return cmd, output_dir


def run_one(cmd: list[str], log_path: Path) -> dict[str, object]:
    started = datetime.now().isoformat(timespec="seconds")
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    finished = datetime.now().isoformat(timespec="seconds")
    log_path.write_text(
        "\n".join(
            [
                f"started: {started}",
                f"finished: {finished}",
                f"returncode: {result.returncode}",
                "",
                "command:",
                " ".join(cmd),
                "",
                "stdout:",
                result.stdout,
                "",
                "stderr:",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )
    return {
        "started": started,
        "finished": finished,
        "returncode": result.returncode,
        "log_path": str(log_path.relative_to(ROOT)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-types", nargs="+", default=["random", "scaffold"], choices=["random", "scaffold"])
    parser.add_argument("--variants", nargs="+", default=["dmpnn_rdkit2d"], choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(MODELS, REPORTS, LOGS)
    rows: list[dict[str, object]] = []

    for variant in args.variants:
        for split_type in args.split_types:
            for seed in args.seeds:
                cmd, output_dir = build_command(
                    variant=variant,
                    split_type=split_type,
                    seed=seed,
                    epochs=args.epochs,
                    warmup_epochs=args.warmup_epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                )
                best_model = output_dir / "model_0" / "best.pt"
                row = {
                    "variant": variant,
                    "split_type": split_type,
                    "seed": seed,
                    "epochs": args.epochs,
                    "output_dir": str(output_dir.relative_to(ROOT)),
                    "best_model": str(best_model.relative_to(ROOT)),
                    "command": cmd,
                }
                if args.skip_existing and best_model.exists():
                    row["status"] = "skipped_existing"
                    row["returncode"] = 0
                elif args.dry_run:
                    row["status"] = "dry_run"
                    row["returncode"] = None
                else:
                    ensure_dirs(output_dir)
                    log_path = LOGS / f"{variant}_{split_type}_seed_{seed}_epochs_{args.epochs}.log"
                    result = run_one(cmd, log_path)
                    row.update(result)
                    row["status"] = "completed" if result["returncode"] == 0 else "failed"
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "command"}, ensure_ascii=False))
                if row["status"] == "failed":
                    break

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split_types": args.split_types,
        "variants": args.variants,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "runs": rows,
    }
    serializable = []
    for row in rows:
        item = row.copy()
        item["command"] = " ".join(item["command"])
        serializable.append(item)
    summary["runs"] = serializable
    (REPORTS / "chemprop_run_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    failed = [row for row in rows if row.get("status") == "failed"]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
