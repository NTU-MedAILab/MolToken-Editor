from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PHASE04 = ROOT / "predictors"
REPORTS = PHASE04 / "reports"
PREDICTIONS = PHASE04 / "predictions"

EXPECTED_SEEDS = {20260528, 20260529, 20260530, 20260531, 20260601}
EXPECTED_SPLITS = {"random", "scaffold"}
EXPECTED_VARIANTS = {"dmpnn_rdkit2d"}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def add_status(rows: list[dict[str, str]], item: str, status: str, note: str) -> None:
    rows.append({"item": item, "status": status, "note": note})


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    failures: list[str] = []

    scaffold_summary = load_json(REPORTS / "scaffold_split_summary.json")
    if scaffold_summary:
        overlap = scaffold_summary.get("scaffold_overlap", {})
        if all(value == 0 for value in overlap.values()):
            add_status(rows, "scaffold split", "pass", "train/valid/test scaffold overlap is zero")
        else:
            failures.append("scaffold split has scaffold overlap")
            add_status(rows, "scaffold split", "fail", json.dumps(overlap, ensure_ascii=False))
    else:
        failures.append("missing scaffold split summary")
        add_status(rows, "scaffold split", "fail", "missing scaffold_split_summary.json")

    verifier_metrics_path = REPORTS / "verifier_model_metrics.csv"
    if verifier_metrics_path.exists():
        verifier_metrics = pd.read_csv(verifier_metrics_path)
        if len(verifier_metrics) >= 60:
            add_status(rows, "verifier models", "pass", f"{len(verifier_metrics)} metric rows")
        else:
            warnings.append("verifier metrics exist but fewer than expected rows")
            add_status(rows, "verifier models", "warn", f"{len(verifier_metrics)} metric rows")
    else:
        failures.append("missing verifier model metrics")
        add_status(rows, "verifier models", "fail", "missing verifier_model_metrics.csv")

    run_manifest = load_json(REPORTS / "chemprop_run_manifest.json")
    completed_runs = [row for row in run_manifest.get("runs", []) if row.get("status") in {"completed", "skipped_existing"}]
    failed_runs = [row for row in run_manifest.get("runs", []) if row.get("status") == "failed"]
    if completed_runs:
        add_status(rows, "Chemprop training", "pass", f"{len(completed_runs)} completed or existing runs")
    else:
        warnings.append("Chemprop formal runs have not been completed yet")
        add_status(rows, "Chemprop training", "warn", "no completed Chemprop run in manifest")
    if failed_runs:
        failures.append("one or more Chemprop runs failed")
        add_status(rows, "Chemprop failed runs", "fail", f"{len(failed_runs)} failed runs")

    completed_matrix = {
        (row.get("variant"), row.get("split_type"), int(row.get("seed")))
        for row in completed_runs
        if row.get("variant") and row.get("split_type") and row.get("seed") is not None
    }
    expected_matrix = {
        (variant, split_type, seed)
        for variant in EXPECTED_VARIANTS
        for split_type in EXPECTED_SPLITS
        for seed in EXPECTED_SEEDS
    }
    missing = sorted(expected_matrix - completed_matrix)
    if missing:
        warnings.append("full 5-seed Chemprop matrix is not complete")
        add_status(rows, "5-seed Chemprop matrix", "warn", f"missing {len(missing)} of {len(expected_matrix)} runs")
    else:
        add_status(rows, "5-seed Chemprop matrix", "pass", "all expected runs completed")

    chemprop_metrics_path = REPORTS / "chemprop_main_metrics.csv"
    if chemprop_metrics_path.exists():
        chemprop_metrics = pd.read_csv(chemprop_metrics_path)
        if chemprop_metrics.empty:
            warnings.append("Chemprop metrics file is empty")
            add_status(rows, "Chemprop metrics", "warn", "empty metrics table")
        else:
            valid = chemprop_metrics[chemprop_metrics["split"] == "valid"].copy()
            test = chemprop_metrics[chemprop_metrics["split"] == "test"].copy()
            best_valid_rmse = float(valid["rmse"].min()) if not valid.empty else float("nan")
            best_test_rmse = float(test["rmse"].min()) if not test.empty else float("nan")
            add_status(rows, "Chemprop metrics", "pass", f"best valid RMSE={best_valid_rmse:.4f}, best test RMSE={best_test_rmse:.4f}")
            if best_valid_rmse > 0.85:
                warnings.append("Chemprop validation RMSE is still pilot-level and should be improved before manuscript experiments")
    else:
        warnings.append("Chemprop metrics have not been collected")
        add_status(rows, "Chemprop metrics", "warn", "missing chemprop_main_metrics.csv")

    consensus_path = PREDICTIONS / "independent_100_predictor_consensus.csv"
    if consensus_path.exists():
        consensus = pd.read_csv(consensus_path)
        if len(consensus) > 0:
            add_status(rows, "independent_100 consensus", "pass", f"{len(consensus)} ensemble rows")
        else:
            warnings.append("independent_100 consensus table is empty")
            add_status(rows, "independent_100 consensus", "warn", "empty table")
    else:
        warnings.append("independent_100 consensus table has not been generated")
        add_status(rows, "independent_100 consensus", "warn", "missing independent_100_predictor_consensus.csv")

    verdict = "pass_with_warnings"
    if failures:
        verdict = "fail"
    elif not warnings:
        verdict = "pass"

    summary = {
        "verdict": verdict,
        "warnings": warnings,
        "failures": failures,
        "checks": rows,
        "missing_chemprop_runs": [
            {"variant": variant, "split_type": split_type, "seed": seed}
            for variant, split_type, seed in missing
        ],
    }
    (REPORTS / "phase04_self_check.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# 04 阶段自检报告",
        "",
        f"结论：`{verdict}`",
        "",
        "## 检查项",
        "",
        "| 工作 | 状态 | 说明 |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['item']} | {row['status']} | {row['note']} |")

    lines.extend(["", "## 需要注意的问题", ""])
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- 暂无。")

    lines.extend(["", "## 失败项", ""])
    if failures:
        lines.extend([f"- {item}" for item in failures])
    else:
        lines.append("- 暂无。")

    lines.extend(
        [
            "",
            "## 审查意见",
            "",
            "04 阶段已经具备期刊实验框架的骨架：scaffold 切分、传统复核器、多 seed 主预测器流程、独立 100 分子复核表均已纳入。",
            "若当前 Chemprop 只完成少量 seed 或少量 epoch，则只能作为流程可行性结果；用于正文实验前，必须补齐 5 seed、random/scaffold 下的 D-MPNN+RDKit2D 正式设定，并以统一指标重新汇总。",
        ]
    )
    (REPORTS / "phase04_self_check.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
