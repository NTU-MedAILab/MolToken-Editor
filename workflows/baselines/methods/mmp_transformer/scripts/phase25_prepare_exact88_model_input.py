#!/usr/bin/env python
from pathlib import Path
import hashlib
import json
import os
import pickle
from collections import Counter

import pandas as pd
from rdkit import Chem

from sys import path as sys_path

p25 = Path(os.environ["P25"])
repo = Path(os.environ["REPO"])
exact_dir = Path(os.environ["EXACT_DIR"])

source_path = exact_dir / "phase25_exact88_single_logd.csv"
output_path = exact_dir / "phase25_exact88_single_logd_model_input.csv"
vocab_path = exact_dir / "vocab.pkl"

audit_path = (
    p25
    / "tables/exact88"
    / "phase25_exact88_model_input_normalization.csv"
)

manifest_path = (
    p25
    / "tables/exact88"
    / "phase25_exact88_model_input_manifest.json"
)

for file_path in [source_path, vocab_path]:
    if not file_path.exists():
        raise FileNotFoundError(file_path)

sys_path.insert(0, str(repo))

from preprocess.vocabulary import SMILESTokenizer  # noqa: E402

with vocab_path.open("rb") as handle:
    vocabulary = pickle.load(handle)

if callable(getattr(vocabulary, "tokens", None)):
    vocab_tokens = {
        str(token)
        for token in vocabulary.tokens()
    }
elif hasattr(vocabulary, "_tokens"):
    raw_tokens = vocabulary._tokens
    if isinstance(raw_tokens, dict):
        vocab_tokens = {
            str(token)
            for token in raw_tokens.keys()
        }
    else:
        vocab_tokens = {
            str(token)
            for token in raw_tokens
        }
else:
    raise RuntimeError("Could not read official vocabulary.")

tokenizer = SMILESTokenizer()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def tokenize_oov(smiles: str):
    tokens = tokenizer.tokenize(smiles)
    oov = [
        str(token)
        for token in tokens
        if str(token) not in vocab_tokens
    ]
    return tokens, oov


def graph_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # 忽略立体信息，仅比较原子—键连接关系。
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )


def remove_bond_stereo(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    for bond in mol.GetBonds():
        bond.SetBondDir(Chem.BondDir.NONE)
        bond.SetStereo(Chem.BondStereo.STEREONONE)

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


def remove_all_stereo(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    Chem.RemoveStereochemistry(mol)

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )


data = pd.read_csv(source_path)

if len(data) != 88:
    raise RuntimeError(
        f"Expected 88 rows, found {len(data)}."
    )

records = []
normalized_smiles = []
normalization_modes = []
oov_counter = Counter()

for _, row in data.iterrows():
    parent_id = str(row["parent_id"])
    original = str(row["Source_Mol"]).strip()

    _, oov_before = tokenize_oov(original)
    oov_counter.update(oov_before)

    selected = original
    mode = "unchanged"

    _, selected_oov = tokenize_oov(selected)

    if selected_oov:
        candidate = remove_bond_stereo(original)
        _, candidate_oov = tokenize_oov(candidate)

        if not candidate_oov:
            selected = candidate
            selected_oov = []
            mode = "remove_bond_stereo"
        else:
            candidate = remove_all_stereo(original)
            _, candidate_oov = tokenize_oov(candidate)

            if not candidate_oov:
                selected = candidate
                selected_oov = []
                mode = "remove_all_stereo"
            else:
                raise RuntimeError(
                    f"Unresolved vocabulary tokens for "
                    f"{parent_id}: {candidate_oov}; "
                    f"SMILES={candidate}"
                )

    original_graph = graph_key(original)
    selected_graph = graph_key(selected)
    connectivity_match = (
        original_graph == selected_graph
    )

    if not connectivity_match:
        raise RuntimeError(
            f"Connectivity changed for {parent_id}."
        )

    normalized_smiles.append(selected)
    normalization_modes.append(mode)

    records.append(
        {
            "parent_id": parent_id,
            "original_source_smiles": original,
            "model_input_source_smiles": selected,
            "normalization_mode": mode,
            "oov_tokens_before": "|".join(oov_before),
            "oov_tokens_after": "|".join(selected_oov),
            "connectivity_match": connectivity_match,
            "original_graph_key": original_graph,
            "model_input_graph_key": selected_graph,
        }
    )

output = data.copy()

output.insert(
    output.columns.get_loc("Source_Mol"),
    "Source_Mol_locked",
    output["Source_Mol"],
)

output["Source_Mol"] = normalized_smiles
output["model_input_normalization"] = normalization_modes

output.to_csv(output_path, index=False)

audit = pd.DataFrame(records)
audit.to_csv(audit_path, index=False)

# 最终再检查全部SMILES无OOV。
remaining_oov = []

for _, row in output.iterrows():
    _, oov = tokenize_oov(str(row["Source_Mol"]))
    if oov:
        remaining_oov.append(
            {
                "parent_id": row["parent_id"],
                "oov": oov,
            }
        )

if remaining_oov:
    raise RuntimeError(
        f"Remaining OOV tokens: {remaining_oov}"
    )

mode_counts = {
    str(key): int(value)
    for key, value in audit[
        "normalization_mode"
    ].value_counts().items()
}

manifest = {
    "phase":
        "phase25_exact88_model_input_normalization",
    "status": "prepared",
    "rows": int(len(output)),
    "unique_parent_ids": int(
        output["parent_id"].nunique()
    ),
    "source_csv": str(source_path.resolve()),
    "model_input_csv": str(output_path.resolve()),
    "normalization_mode_counts": mode_counts,
    "oov_token_counts_before": {
        str(key): int(value)
        for key, value in oov_counter.items()
    },
    "remaining_oov_count": 0,
    "connectivity_match_count": int(
        audit["connectivity_match"].sum()
    ),
    "normalization_policy": (
        "Unsupported stereochemical bond-direction "
        "tokens are removed for MMP-Transformer input. "
        "The locked parent SMILES and molecular "
        "connectivity are retained separately."
    ),
    "source_sha256": sha256(source_path),
    "model_input_sha256": sha256(output_path),
    "vocab_sha256": sha256(vocab_path),
    "audit_csv": str(audit_path.resolve()),
}

manifest_path.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print("Rows:", len(output))
print("Unique parents:", output["parent_id"].nunique())
print("Normalization modes:", mode_counts)
print("OOV before:", dict(oov_counter))
print("Remaining OOV:", 0)
print(
    "Connectivity matches:",
    int(audit["connectivity_match"].sum()),
)
print()
print("EXACT88_MODEL_INPUT_PREPARED")
print("Input:", output_path)
print("Audit:", audit_path)
print("Manifest:", manifest_path)
