from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PHASE10 = ROOT / "moltoken"
CONFIG = PHASE10 / "configs" / "target_expansion.json"
TABLES = PHASE10 / "tables"
REPORTS = PHASE10 / "reports"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    series = frame[column]
    if series.dtype == bool:
        return series.fillna(default)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def build_parent_candidates(molecules: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rules = config["parent_candidate_filters"]
    out = molecules.copy()
    mask = pd.Series(True, index=out.index)
    if rules.get("require_filter_pass", True):
        mask &= bool_series(out, "filter_pass", False)
    if rules.get("allowed_molecule_type"):
        mask &= out["molecule_type"].astype(str).eq(str(rules["allowed_molecule_type"]))
    mask &= numeric(out, "mw").between(float(rules["mw_min"]), float(rules["mw_max"]))
    mask &= numeric(out, "heavy_atoms").between(float(rules["heavy_atoms_min"]), float(rules["heavy_atoms_max"]))
    mask &= numeric(out, "qed") >= float(rules["qed_min"])
    mask &= numeric(out, "rotbonds") <= float(rules["rotbonds_max"])
    mask &= numeric(out, "tpsa") <= float(rules["tpsa_max"])
    selected = out[mask].copy()
    selected["phase10_parent_candidate"] = True
    selected["phase10_parent_smiles"] = selected["standardized_smiles"]
    return selected.reset_index(drop=True)


def inventory_libraries(manifest: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    observed_counts = membership.groupby("library_id").size().to_dict()
    rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        library_id = str(row["library_id"])
        count = int(observed_counts.get(library_id, 0))
        unique_scaffolds = int(row.get("unique_scaffolds", 0))
        rows.append(
            {
                "library_id": library_id,
                "membership_table": "data/external/library_membership.csv",
                "exists": count > 0,
                "molecule_count_manifest": int(row.get("molecule_count", 0)),
                "molecule_count_file": count,
                "unique_scaffolds": unique_scaffolds,
                "mw_mean": float(row.get("mw_mean", 0.0)),
                "mol_logp_mean": float(row.get("mol_logp_mean", 0.0)),
                "qed_mean": float(row.get("qed_mean", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    config = load_json(CONFIG)
    sources = config["source_libraries"]
    manifest = pd.read_csv(resolve_path(sources["mini_library_manifest"]))
    molecules = pd.read_csv(resolve_path(sources["filtered_external_molecules"]))
    membership = pd.read_csv(resolve_path(sources["mini_library_membership"]))

    inventory = inventory_libraries(manifest, membership)
    candidates = build_parent_candidates(molecules, config)
    inventory.to_csv(TABLES / "phase10_small_library_inventory.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(TABLES / "phase10_parent_screening_candidates.csv", index=False, encoding="utf-8-sig")

    gates = config["acceptance_gates"]
    checks = [
        {
            "name": "library_count_min",
            "value": int(len(inventory)),
            "threshold": int(gates["library_count_min"]),
            "passed": int(len(inventory)) >= int(gates["library_count_min"]),
        },
        {
            "name": "all_library_files_exist",
            "value": int(inventory["exists"].sum()),
            "threshold": int(len(inventory)),
            "passed": bool(inventory["exists"].all()),
        },
        {
            "name": "parent_candidate_count_min",
            "value": int(len(candidates)),
            "threshold": int(gates["parent_candidate_count_min"]),
            "passed": int(len(candidates)) >= int(gates["parent_candidate_count_min"]),
        },
    ]
    failures = [item for item in checks if not item["passed"]]
    summary = {
        "stage": "10-A",
        "verdict": "pass" if not failures else "needs_fix",
        "library_count": int(len(inventory)),
        "existing_library_files": int(inventory["exists"].sum()),
        "filtered_external_molecules": int(len(molecules)),
        "parent_screening_candidates": int(len(candidates)),
        "candidate_descriptor_summary": {
            "mw_mean": float(candidates["mw"].mean()) if not candidates.empty else 0.0,
            "mol_logp_mean": float(candidates["mol_logp"].mean()) if not candidates.empty else 0.0,
            "qed_mean": float(candidates["qed"].mean()) if not candidates.empty else 0.0,
            "unique_scaffolds": int(candidates["murcko_scaffold"].nunique()) if "murcko_scaffold" in candidates.columns else 0,
        },
        "checks": checks,
        "failures": failures,
        "outputs": {
            "inventory": str((TABLES / "phase10_small_library_inventory.csv").relative_to(ROOT)),
            "parent_candidates": str((TABLES / "phase10_parent_screening_candidates.csv").relative_to(ROOT)),
            "report": str((REPORTS / "phase10_small_library_inventory_report.md").relative_to(ROOT)),
        },
    }
    (REPORTS / "phase10_small_library_inventory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 10-A Small Library Inventory Report",
        "",
        f"- library count: `{summary['library_count']}`",
        f"- existing library files: `{summary['existing_library_files']}`",
        f"- filtered external molecules: `{summary['filtered_external_molecules']}`",
        f"- parent screening candidates: `{summary['parent_screening_candidates']}`",
        f"- candidate mean MW: `{summary['candidate_descriptor_summary']['mw_mean']:.3f}`",
        f"- candidate mean LogP: `{summary['candidate_descriptor_summary']['mol_logp_mean']:.3f}`",
        f"- candidate mean QED: `{summary['candidate_descriptor_summary']['qed_mean']:.3f}`",
        f"- candidate unique scaffolds: `{summary['candidate_descriptor_summary']['unique_scaffolds']}`",
        "",
        "## Checks",
        "",
    ]
    for item in checks:
        status = "pass" if item["passed"] else "fail"
        lines.append(f"- `{status}` {item['name']}: {item['value']} >= {item['threshold']}")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "- Run the main predictor on parent screening candidates.",
            "- Select molecules outside the LogD target window with balanced low/high sides.",
            "- Run contribution attribution and Mol Token segmentation only on the screened active parents.",
        ]
    )
    (REPORTS / "phase10_small_library_inventory_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "10-A", "verdict": summary["verdict"], "parent_candidates": summary["parent_screening_candidates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
