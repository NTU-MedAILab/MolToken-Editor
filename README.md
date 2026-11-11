# MolToken-Editor

Code and locked numerical evidence for **MolToken-Editor: An
Attribution-Guided Fragment Editing Method for Interpretable Molecular
Optimization**.

MolToken-Editor predicts molecular lipophilicity, locates connected regions by
counterfactual attribution, applies deletion/replacement/addition edits, and
checks proposed candidates with structural, applicability-domain, and
auxiliary-model gates. The paper evaluates directional logD optimization in a
predefined target window of 1.0–3.4.

## Repository contents

```text
MolToken-Editor/
├── data/          # Lipophilicity data, fixed splits, and external-library inputs
├── environment/   # Linux Conda environment
├── workflows/     # Predictor, MolToken, comparison, baseline, and ablation code
├── results/       # Compact manuscript tables and figure source data
├── reproduce.py   # Unified reproduction entry point
└── validate.py    # Static and numerical consistency checks
```

The repository intentionally excludes manuscript drafts, historical figures,
temporary audits, failed runs, model checkpoints, logs, and the measured-logD
MMP retrospective experiment that is not part of the submitted manuscript.
Large third-party model assets are downloaded from their official sources.

All manuscript-facing numerical results can be recalculated from the locked
inputs included in this release. Fresh reruns are supported for the predictor,
MolToken, localization, and ablation workflows. External-baseline generation
additionally requires the official third-party software and model assets. The
A05 attribution-matched control is reproduced from locked matching and scoring
records; its original matching, candidate-generation, inference, and scoring
stages are not rerun by this archive.

## Environment

Linux:

```bash
conda env create -f environment/linux.yml
conda activate moltoken
```

Check the package before running expensive experiments:

```bash
python validate.py
python reproduce.py all --dry-run
```

## Reproduction commands

```bash
# D-MPNN, XGBoost, and Random Forest training/evaluation
python reproduce.py predictors --batch-size 64

# Parent screening, attribution, MolToken construction, editing, and gating
python reproduce.py moltoken --batch-size 64

# MolToken, BRICS, RECAP, Bemis–Murcko side-chain, and action-matched random localization
python reproduce.py localization

# Attribution-localization and decision-gate ablations
python reproduce.py ablation

# Recalculate the locked Exact88, B=5 three-method comparison
python reproduce.py equal-budget

# Recalculate the locked A05 attribution-matched control evidence
python reproduce.py attribution-control

# Print the original end-to-end command sequence without executing it
python reproduce.py all --dry-run
```

Fresh predictor and MolToken runs write generated artifacts inside the relevant
workflow directories, which are ignored by Git. Frozen paper-facing values are
kept in `results/`; fresh reruns must not overwrite these files without a
corresponding manuscript update.

The legacy machine-readable identifier `rgroup_enumeration` denotes the
Bemis–Murcko side-chain localization strategy. It is retained in internal file
names and historical provenance records to preserve compatibility with the
frozen workflow.

## Reproducibility modes

Two complementary reproduction modes are supported.

### Locked-evidence reproduction

The files under `results/` contain the numerical evidence used in the
manuscript. The following commands recalculate the reported ablation,
fixed-slot external comparison, and A05 matched-control statistics from the
included frozen inputs:

```bash
python validate.py
python reproduce.py ablation
python reproduce.py equal-budget
python reproduce.py attribution-control
```

The external comparison uses five requested final evaluation slots per
starting molecule (B=5); unavailable slots remain in the denominator and count
as unsuccessful. The A05 command verifies the frozen matching, parent-level
effects, and bootstrap distributions. It does not rerun matching, candidate
generation, model inference, property scoring, or B=5 selection.

### Full retraining reproduction

To rerun the original predictor, MolToken, localization, ablation, and
external-comparison workflow:

```bash
python reproduce.py all --batch-size 64
```

A05 is intentionally excluded from `all` because its historical model
invocations depend on archived checkpoints and machine-specific environments.
Use `python reproduce.py attribution-control` for its portable locked-evidence
recalculation.

D-MPNN training can show small platform-dependent numerical variation even
with fixed seeds. Later discrete thresholds can therefore change some
candidate counts. Fresh retraining is a conclusion-level replication, while
`results/` remains the locked source for exact manuscript values.

Generated checkpoints, predictions, logs, and intermediate tables are ignored
by Git and must not overwrite the locked files under `results/`.

## Holdout-set evaluation

Table S1 is calculated from frozen ensemble predictions for 100 molecules with measured logD. The predictions and metric summaries are in [`results/holdout100/`](results/holdout100/).

```bash
python workflows/predictors/scripts/recalculate_holdout100.py
```

## External baselines

The locked candidate tables needed to recalculate the reported comparison are
included under `workflows/baselines/inputs`. Full candidate generation requires
the official external software:

- REINVENT4 v4.8 / package version 4.8.24 and the `mol2mol_mmp.prior`.
- MMP-Transformer from `MolecularAI/deep-molecular-optimization` and its
  official pretrained checkpoint.

The method-specific adapters and setup scripts are under
`workflows/baselines/methods`.

## Locked comparison

Under the fixed final evaluation-slot protocol, five slots are requested per
starting molecule (B=5) on the common Exact88 set. The starting-molecule-level evidence-qualified target hit rates were 79.55% for MolToken-Editor, 52.65% ± 8.83% for REINVENT4, and 66.29% ± 4.59% for MMP-Transformer. Unavailable slots remain in
the fixed denominator and count as unsuccessful. This protocol normalizes the
final requested evaluation slots; it does not establish equal whole-process
search effort, model-call counts, or runtime. See
`results/manuscript_tables/equal_budget_comparison.csv` for all endpoints.

## Generation counts and structural guardrails

[`results/table_s2/`](results/table_s2/) contains the per-run generation and model–molecule input counts and the candidate-level structural guardrail records for Table S2.

```bash
python workflows/baselines/scripts/recalculate_table_s2.py
```

## Attribution-matched control

The added A05 analysis compares attribution-guided MolToken localization with
strictly matched non-attribution controls. The matching universe contained 178
attribution-selected regions from 90 parents. Across five matching seeds, 92
regions were strictly matched, covering 65 parents.

The 65-parent primary paired analysis estimated a target-window B=5 effect of
+10.15 percentage points (95% CI, +3.14 to +17.42) and a strict-success effect
of +6.09 percentage points (95% CI, +0.55 to +12.12). Target-distance and final
edit-score endpoints are limited to a 58-parent complete-observation
sensitivity analysis. Parent is the inferential unit. These computational
results do not support causal or wet-lab claims.

See `results/manuscript_tables/attribution_matched_control.csv` and the
corresponding files under `results/figure_source_data/` and
`results/provenance/a05/`.

## Citation and license

Citation metadata are provided in `CITATION.cff`. A project-level license must
be selected before the repository is made public; see `LICENSE_NOTICE.md`.
