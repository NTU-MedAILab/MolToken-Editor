#!/usr/bin/env python3
"""Recalculate Table S1 from frozen holdout ensemble predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

LABELS = ("low", "in_range", "high")
METRIC_NAMES = ("rmse", "mae", "r2", "spearman", "region_balanced_accuracy", "region_macro_f1")
MODEL_IDS = ("dmpnn_rdkit2d", "xgboost", "random_forest")

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def label(value: float) -> str:
    if value < 1.0:
        return "low"
    if value <= 3.4:
        return "in_range"
    return "high"

def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for pos in range(start, end):
            result[order[pos]] = average_rank
        start = end
    return result

def pearson(x: list[float], y: list[float]) -> float:
    xmean, ymean = statistics.mean(x), statistics.mean(y)
    xx = sum((v - xmean) ** 2 for v in x)
    yy = sum((v - ymean) ** 2 for v in y)
    return sum((a - xmean) * (b - ymean) for a, b in zip(x, y)) / math.sqrt(xx * yy)

def metrics(rows: list[dict[str, str]], prediction_column: str) -> dict[str, object]:
    y = [float(r["exp"]) for r in rows]
    p = [float(r[prediction_column]) for r in rows]
    errors = [b - a for a, b in zip(y, p)]
    ymean = statistics.mean(y)
    ss_res = sum(e * e for e in errors)
    ss_tot = sum((v - ymean) ** 2 for v in y)
    actual = [label(v) for v in y]
    predicted = [label(v) for v in p]
    recalls, f1s = [], []
    per_class = {}
    for cls in LABELS:
        indexes = [i for i, x in enumerate(actual) if x == cls]
        tp = sum(actual[i] == cls and predicted[i] == cls for i in range(len(rows)))
        fp = sum(actual[i] != cls and predicted[i] == cls for i in range(len(rows)))
        fn = len(indexes) - tp
        if indexes:
            recalls.append(tp / len(indexes))
        if 2 * tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
        per_class[cls] = {
            "n": len(indexes),
            "mae": sum(abs(errors[i]) for i in indexes) / len(indexes) if indexes else None,
            "mean_error_pred_minus_true": sum(errors[i] for i in indexes) / len(indexes) if indexes else None,
            "recall": tp / len(indexes) if indexes else None,
        }
    return {
        "n": len(rows),
        "rmse": math.sqrt(ss_res / len(rows)),
        "mae": sum(abs(e) for e in errors) / len(rows),
        "r2": 1 - ss_res / ss_tot,
        "spearman": pearson(ranks(y), ranks(p)),
        "region_balanced_accuracy": statistics.mean(recalls),
        "region_macro_f1": statistics.mean(f1s),
        "per_class_posthoc": per_class,
    }

def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(repo, data):
    provenance = json.loads((data / "provenance.json").read_text())
    for name, digest in provenance["sha256"].items():
        check(hashlib.sha256((data / name).read_bytes()).hexdigest() == digest,
              "Frozen file hash differs: " + name)
    truth_rows = read_csv(repo / "data/lipophilicity/splits/independent_100.csv")
    truth = {row["mol_id"]: row for row in truth_rows}
    check(len(truth_rows) == len(truth) == 100, "Expected 100 distinct holdout molecules")
    ensembles = {}
    seed_values = defaultdict(dict)
    seeds = set(map(str, provenance["training_seeds"]))
    families = (("chemprop", "dmpnn_rdkit2d"), ("verifier", None))
    for family, default_model in families:
        for suffix in ("ensemble", "seed_predictions"):
            name = f"independent_100_{family}_{suffix}.csv"
            records = read_csv(data / name)
            for row in records:
                model = default_model or row["model"]
                check(model in MODEL_IDS, "Unexpected model ID")
                check(row["split_type"] in ("random", "scaffold"), "Unexpected split label")
                parent = truth.get(row["mol_id"])
                check(parent is not None, "Unknown holdout molecule")
                check(row["canonical_smiles"] == parent["canonical_smiles"], "Holdout structure differs")
                check(float(row["exp"]) == float(parent["exp"]), "Measured logD differs")
                key = (row["split_type"], model, row["mol_id"])
                if suffix == "ensemble":
                    check(key not in ensembles, "Duplicate ensemble prediction")
                    check(math.isfinite(float(row["pred_mean"])), "Nonfinite ensemble prediction")
                    ensembles[key] = row
                else:
                    check(row["seed"] in seeds, "Unexpected training seed")
                    check(row["seed"] not in seed_values[key], "Duplicate seed prediction")
                    check(math.isfinite(float(row["pred"])), "Nonfinite seed prediction")
                    seed_values[key][row["seed"]] = float(row["pred"])
    expected_keys = {(split, model, mol_id) for split in ("random", "scaffold")
                     for model in MODEL_IDS for mol_id in truth}
    check(set(ensembles) == set(seed_values) == expected_keys, "Incomplete holdout predictions")
    max_error = 0.0
    for key, row in ensembles.items():
        check(set(seed_values[key]) == seeds, "Incomplete five-model ensemble")
        difference = abs(float(row["pred_mean"]) - statistics.mean(seed_values[key].values()))
        max_error = max(max_error, difference)
        check(difference <= 1e-10, "Ensemble differs from the five seed predictions")
    output_metrics, output_recall = [], []
    for split in ("random", "scaffold"):
        for model in MODEL_IDS:
            group = [ensembles[(split, model, mol_id)] for mol_id in sorted(truth)]
            values = metrics(group, "pred_mean")
            output_metrics.append({"split": split, "model": model,
                                   **{name: values[name] for name in ("n", *METRIC_NAMES)}})
            for cls in LABELS:
                rows = [row for row in group if label(float(row["exp"])) == cls]
                tp = sum(label(float(row["pred_mean"])) == cls for row in rows)
                output_recall.append({"split": split, "model": model, "class": cls,
                                      "n": len(rows), "true_positive": tp, "recall": tp / len(rows)})
    for name, observed, keys in (("metrics.csv", output_metrics, ("split", "model")),
                                 ("class_recall.csv", output_recall, ("split", "model", "class"))):
        expected = {tuple(row[key] for key in keys): row for row in read_csv(data / name)}
        check(len(expected) == len(observed), "Frozen summary size differs: " + name)
        for row in observed:
            reference = expected[tuple(row[key] for key in keys)]
            for field, value in row.items():
                if field not in keys:
                    check(math.isclose(value, float(reference[field]), rel_tol=0, abs_tol=1e-10),
                          "Recalculated value differs: " + name + " / " + field)
    return output_metrics, output_recall, max_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    data = args.data_dir or repo / "results/holdout100"
    rows, recalls, max_error = audit(repo, data)
    if not args.check_only:
        output = args.output_dir or repo / "runs/holdout100"
        output.mkdir(parents=True, exist_ok=True)
        for name, records in (("metrics.csv", rows), ("class_recall.csv", recalls)):
            with (output / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        print("OUTPUT_DIRECTORY:", output)
    print("HOLDOUT100_RECALCULATION_PASS")
    print("Measured molecules: 100; ensemble predictions: 600; individual predictions: 3000")
    print("Maximum ensemble/seed-mean error:", max_error)
    for row in rows:
        print(row["split"], row["model"], " ".join(f"{key}={row[key]:.3f}" for key in METRIC_NAMES))


if __name__ == "__main__":
    main()
