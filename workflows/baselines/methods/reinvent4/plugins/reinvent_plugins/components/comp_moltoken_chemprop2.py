"""REINVENT4 scoring adapter that exactly preserves MolToken's Chemprop-v2 oracle.

REINVENT4's built-in ChemProp2 component calls ``dataset.normalize_inputs``
for RDKit2D descriptors. MolToken's checkpoints already contain the scaler
learned during training, and the standard Chemprop predict path passes raw
217-dimensional descriptors to that stored transform. Re-normalizing on each
RL batch changes predictions and makes the comparison invalid.

This component is intentionally narrow: Chemprop 2.2.3, one single-task logD
endpoint, a directory containing the locked ensemble, and RDKit2D features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chemprop
import numpy as np
import torch
from chemprop import data, featurizers, models
from chemprop.models.utils import load_output_columns
from lightning.pytorch import Trainer
from pydantic import Field
from pydantic.dataclasses import dataclass
from rdkit import Chem

from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.normalize import normalize_smiles


logger = logging.getLogger("reinvent")
REQUIRED_CHEMPROP_VERSION = "2.2.3"


@add_tag("__parameters")
@dataclass
class Parameters:
    model_path: list[str]
    target_column: Optional[list[str]] = Field(default_factory=lambda: [None])
    expected_models: list[int] = Field(default_factory=lambda: [10])


@add_tag("__component")
class MolTokenChemProp2:
    def __init__(self, params: Parameters):
        if chemprop.__version__ != REQUIRED_CHEMPROP_VERSION:
            raise RuntimeError(
                f"MolTokenChemProp2 requires chemprop {REQUIRED_CHEMPROP_VERSION}; "
                f"found {chemprop.__version__}"
            )

        self.smiles_type = "rdkit_smiles"
        model_path = Path(params.model_path[0])
        if model_path.is_dir():
            paths = sorted(model_path.rglob("*.pt"))
        elif model_path.is_file():
            paths = [model_path]
        else:
            raise FileNotFoundError(model_path)

        expected_models = int(params.expected_models[0])
        if len(paths) != expected_models:
            raise ValueError(
                f"Expected {expected_models} locked Chemprop models; found {len(paths)} in {model_path}"
            )
        self.models = [models.load_model(path) for path in paths]
        first_path = paths[0]

        if any(model.n_tasks != 1 for model in self.models):
            raise ValueError("MolToken main logD oracle must be single-task")
        output_columns = load_output_columns(first_path)
        self.column_indices = []
        for target_col in params.target_column:
            if target_col is None:
                self.column_indices.append(0)
            else:
                if not output_columns or target_col not in output_columns:
                    raise ValueError(
                        f"target_column {target_col!r} not present; available={output_columns}"
                    )
                self.column_indices.append(output_columns.index(target_col))

        self.number_of_endpoints = len(self.column_indices)
        self.descriptor_featurizer = featurizers.RDKit2DFeaturizer()
        self.graph_featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        expected_dims = {
            int(model.X_d_transform.mean.shape[-1])
            for model in self.models
            if hasattr(model.X_d_transform, "mean")
        }
        if expected_dims != {217}:
            raise ValueError(f"Locked oracle must expect 217 RDKit2D descriptors; got {expected_dims}")

        logger.info(
            "MolTokenChemProp2 loaded %d models with chemprop %s and raw RDKit2D(217)",
            len(self.models),
            chemprop.__version__,
        )

    @normalize_smiles
    def __call__(self, smilies: list[str]) -> ComponentResults:
        mols = [Chem.MolFromSmiles(smiles) for smiles in smilies]
        features = [self.descriptor_featurizer(mol) for mol in mols]
        if features and any(len(row) != 217 for row in features):
            raise ValueError("RDKit2D feature dimension changed from the locked value 217")

        datapoints = [
            data.MoleculeDatapoint(mol, x_d=np.asarray(row, dtype=float))
            for mol, row in zip(mols, features)
        ]
        dataset = data.MoleculeDataset(
            datapoints,
            featurizer=self.graph_featurizer,
            n_workers=0,
        )
        # Deliberately do NOT call dataset.normalize_inputs(). The stored
        # X_d_transform inside each checkpoint applies its training scaler.
        dataloader = data.build_dataloader(dataset, shuffle=False)

        all_preds = []
        with torch.inference_mode():
            for model in self.models:
                trainer = Trainer(
                    logger=None,
                    enable_progress_bar=False,
                    accelerator="auto",
                    devices=1,
                )
                batches = trainer.predict(model, dataloader)
                preds = np.concatenate(batches, axis=0)
                if preds.ndim == 1:
                    preds = preds[:, np.newaxis]
                all_preds.append(preds)

        scores = np.mean(all_preds, axis=0)
        return ComponentResults(
            [scores[:, index].astype(float) for index in self.column_indices]
        )
