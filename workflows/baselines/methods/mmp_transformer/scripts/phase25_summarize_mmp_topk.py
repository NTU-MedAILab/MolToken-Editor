#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

BOOTSTRAP_METRICS = [
    "target_window_hit_at_k","strict_hit_at_k",
    "target_window_candidate_rate_topk","strict_candidate_rate_topk",
    "mean_delta_topk","mean_final_edit_score_topk",
    "mean_parent_tanimoto_topk","mean_delta_qed_topk","mean_delta_sa_topk",
]

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def load_module(path):
    spec=importlib.util.spec_from_file_location("phase24_compare_exact88",path)
    if spec is None or spec.loader is None: raise ImportError(path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def find_col(df, choices, label):
    for c in choices:
        if c in df.columns: return c
    raise KeyError(f"Cannot locate {label}: {choices}")

def bootstrap_mean(values,n_boot,seed):
    values=np.asarray(values,dtype=float); values=values[np.isfinite(values)]
    if not len(values): return {"n":0,"estimate":float("nan"),"ci_low":float("nan"),"ci_high":float("nan")}
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(values),size=(n_boot,len(values)))
    dist=values[idx].mean(axis=1)
    return {"n":int(len(values)),"estimate":float(values.mean()),
            "ci_low":float(np.quantile(dist,.025)),"ci_high":float(np.quantile(dist,.975))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase24-compare-script",type=Path,required=True)
    p.add_argument("--scored",type=Path,required=True)
    p.add_argument("--parent-table",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--method-group",default="mmp_transformer_single_logd")
    p.add_argument("--method-label",default="MMP-Transformer")
    p.add_argument("--top-k",nargs="+",type=int,default=[1,3,5])
    p.add_argument("--bootstrap",type=int,default=10000)
    p.add_argument("--bootstrap-seed",type=int,default=20260703)
    p.add_argument("--expected-parents",type=int,default=88)
    p.add_argument("--expected-raw-candidates",type=int,default=4400)
    p.add_argument("--expected-per-parent",type=int,default=50)
    a=p.parse_args()

    phase24_script=a.phase24_compare_script.expanduser().resolve()
    scored_path=a.scored.expanduser().resolve()
    parent_path=a.parent_table.expanduser().resolve()
    out=a.output_dir.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    for path in [phase24_script,scored_path,parent_path]:
        if not path.exists(): raise FileNotFoundError(path)

    mod=load_module(phase24_script)
    mod.METHODS=[a.method_group]
    mod.METHOD_LABELS={a.method_group:a.method_label}

    scored=pd.read_csv(scored_path,low_memory=False)
    parents=pd.read_csv(parent_path,low_memory=False)
    pid=find_col(parents,["parent_id","mol_id"],"parent ID")
    dcol=find_col(parents,["optimization_direction","replace_direction"],"optimization direction")
    meta=parents[[pid,dcol]].copy(); meta.columns=["mol_id","optimization_direction"]
    meta["mol_id"]=meta["mol_id"].astype(str).str.strip()
    meta["optimization_direction"]=meta["optimization_direction"].astype(str).str.strip()
    if (meta.groupby("mol_id")["optimization_direction"].nunique()>1).any():
        raise ValueError("Inconsistent parent directions")
    meta=meta.drop_duplicates("mol_id",keep="first")
    locked_ids=meta["mol_id"].tolist(); locked_set=set(locked_ids)
    direction=dict(zip(meta["mol_id"],meta["optimization_direction"]))
    if len(locked_ids)!=a.expected_parents: raise ValueError(f"Parent count {len(locked_ids)}")

    work=scored[scored["method_group"].astype(str).eq(a.method_group)].copy()
    work["mol_id"]=work["mol_id"].astype(str).str.strip()
    work=work[work["mol_id"].isin(locked_set)].copy()
    counts=work.groupby("mol_id").size()
    if len(work)!=a.expected_raw_candidates: raise ValueError(f"Raw rows {len(work)}")
    if work["mol_id"].nunique()!=a.expected_parents: raise ValueError("Parent coverage mismatch")
    if not counts.eq(a.expected_per_parent).all(): raise ValueError("Not exactly 50 candidates per parent")

    ranked,audit=mod.prepare_scored(work,a.method_group,locked_set)
    topk=sorted(set(a.top_k))
    parent=mod.parent_topk_table(ranked,locked_ids,direction,topk)
    expected=a.expected_parents*len(topk)
    if len(parent)!=expected: raise ValueError(f"Parent Top-K rows {len(parent)} != {expected}")
    summary=mod.method_summary(parent)

    direction_summary=(parent.groupby(
        ["method_group","method_label","optimization_direction","k"],as_index=False
    ).agg(
        n_parents=("mol_id","size"),
        evidence_coverage_rate=("evidence_available","mean"),
        target_window_candidate_rate_topk=("target_window_candidate_rate_topk","mean"),
        strict_candidate_rate_topk=("strict_candidate_rate_topk","mean"),
        target_window_hit_rate=("target_window_hit_at_k","mean"),
        strict_hit_rate=("strict_hit_at_k","mean"),
        mean_delta_topk=("mean_delta_topk","mean"),
        mean_final_edit_score_topk=("mean_final_edit_score_topk","mean"),
        mean_parent_tanimoto_topk=("mean_parent_tanimoto_topk","mean"),
        mean_delta_qed_topk=("mean_delta_qed_topk","mean"),
        mean_delta_sa_topk=("mean_delta_sa_topk","mean"),
    ).sort_values(["optimization_direction","k"]))

    brows=[]; offset=0
    for k in topk:
        sub=parent[parent["k"].eq(k)]
        for metric in BOOTSTRAP_METRICS:
            r=bootstrap_mean(pd.to_numeric(sub[metric],errors="coerce").to_numpy(),
                             a.bootstrap,a.bootstrap_seed+offset)
            brows.append({"method_group":a.method_group,"method_label":a.method_label,
                          "k":int(k),"metric":metric,**r,
                          "bootstrap_iterations":int(a.bootstrap),
                          "bootstrap_seed":int(a.bootstrap_seed+offset)})
            offset+=1
    boot=pd.DataFrame(brows)

    maxk=max(topk)
    top_selected=ranked[ranked["rank_in_group_parent"].le(maxk)].copy()
    top1=ranked[ranked["rank_in_group_parent"].eq(1)].copy()
    outputs={
        "ranked_candidates":out/"exact88_ranked_unique_evidence_candidates.csv",
        "selected_topk":out/f"exact88_top{maxk}_ranked_candidates.csv",
        "top1_candidates":out/"exact88_top1_candidates.csv",
        "parent_topk_metrics":out/"exact88_parent_topk_metrics.csv",
        "method_topk_summary":out/"exact88_method_topk_summary.csv",
        "direction_topk_summary":out/"exact88_direction_topk_summary.csv",
        "bootstrap_ci":out/"exact88_method_bootstrap_ci.csv",
    }
    ranked.to_csv(outputs["ranked_candidates"],index=False)
    top_selected.to_csv(outputs["selected_topk"],index=False)
    top1.to_csv(outputs["top1_candidates"],index=False)
    parent.to_csv(outputs["parent_topk_metrics"],index=False)
    summary.to_csv(outputs["method_topk_summary"],index=False)
    direction_summary.to_csv(outputs["direction_topk_summary"],index=False)
    boot.to_csv(outputs["bootstrap_ci"],index=False)

    checks={
        "raw_candidate_rows":len(work)==a.expected_raw_candidates,
        "raw_parent_coverage":work["mol_id"].nunique()==a.expected_parents,
        "raw_exactly_per_parent":bool(counts.eq(a.expected_per_parent).all()),
        "evidence_ready_parent_coverage":ranked["mol_id"].nunique()==a.expected_parents,
        "top1_parent_coverage":top1["mol_id"].nunique()==a.expected_parents,
        "parent_topk_rows":len(parent)==expected,
        "summary_rows":len(summary)==len(topk),
    }
    manifest={
        "phase":"phase25_mmp_transformer_exact88_topk",
        "status":"pass" if all(checks.values()) else "needs_review",
        "method_group":a.method_group,"method_label":a.method_label,"top_k":topk,
        "ranking_policy":[
            "target_window_success descending",
            "relaxed_guardrail_pass descending",
            "auxiliary_opposition_count ascending",
            "final_edit_score descending",
            "canonical isomeric SMILES ascending",
        ],
        "ranking_sort_kind":"mergesort","checks":checks,
        "counts":{
            "raw_rows":int(len(work)),"raw_parents":int(work["mol_id"].nunique()),
            "valid_and_scored_rows":int(audit["valid_and_scored_rows"]),
            "evidence_ready_rows_before_canonical_dedup":int(audit["evidence_ready_rows_before_canonical_dedup"]),
            "unique_evidence_ready_rows":int(audit["unique_evidence_ready_rows"]),
            "evidence_ready_parents":int(audit["evidence_ready_parent_coverage"]),
            "top1_rows":int(len(top1)),"parent_topk_rows":int(len(parent)),
        },
        "inputs":{
            "phase24_compare_script":{"path":str(phase24_script),"sha256":sha256(phase24_script)},
            "scored":{"path":str(scored_path),"sha256":sha256(scored_path)},
            "parent_table":{"path":str(parent_path),"sha256":sha256(parent_path)},
        },
        "outputs":{n:{"path":str(p.resolve()),"sha256":sha256(p)} for n,p in outputs.items()},
    }
    mpath=out/"topk_manifest.json"
    mpath.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
    print("\n===== OVERALL TOP-K SUMMARY =====")
    print(summary.to_string(index=False))
    print("\n===== DIRECTION TOP-K SUMMARY =====")
    print(direction_summary.to_string(index=False))
    print("\nPHASE25_MMP_TRANSFORMER_TOPK_COMPLETE")
    if manifest["status"]!="pass": raise SystemExit("Top-K integrity checks failed")

if __name__=="__main__":
    main()
