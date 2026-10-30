# Explainable Longitudinal NACC Progression

Research prototype for predicting Alzheimer dementia progression from longitudinal UDS clinical data and quantitative NACC Mixed-protocol MRI measures. This is not a clinical diagnostic tool.

## Approved Protocol

- Use NACC Freeze 74 only, with UDS clinical data and quantitative Mixed-protocol MRI. Do not substitute CLARITI, PET, another dataset, or raw image volumes.
- Require at least two clinical visits through the anchor and allow at most three visits in the GRU sequence.
- Require a quantitative MRI at the MCI anchor, acquired on or before that visit and within the approved 90-day matching window. Earlier clinical visits may have no MRI.
- Represent an unavailable historical MRI with a neutral zero vector after training-fitted scaling and an all-missing MRI mask. Never interpret that placeholder as observed anatomy.
- Predict Alzheimer dementia progression within 730 days after the anchor. Verify prior diagnoses, sufficient negative follow-up, and competing dementia events before freezing labels.
- Create one stratified patient-level 70/15/15 split with seed 42. Fit MRI feature filtering, imputation, and scaling only on available training MRI visits.

## Layout

- `src/nacc_progression/`: contracts, model, training, evaluation, explainability, integration, and final runner.
- `src/nacc_progression/workstream1/`: clinical cohort and label pipeline.
- `src/nacc_progression/workstream2/`: MRI validation and legacy audit utilities.
- `configs/`: public schema and private-config template.
- `docs/`: approved training instructions, privacy guidance, and MRI-workstream status.
- `tests/`: synthetic unit and integration tests; no participant data.
- `NACC-DATA/`: local restricted source data. This directory is Git-ignored and must never be committed.

## Install and Test

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

## Private Data Workflow

Use the approved Freeze 74 schema and authorized UDS/Mixed-protocol MRI exports. Put completed private configuration, source tables, identity secret, preprocessing artifacts, and checkpoints outside the repository. The public file `configs/final.private.template.json` is only a placeholder; it is not ready to run.

Run the preflight before training:

```bash
nacc-final --config /private/path/final.private.json --out /private/path/nacc-run
```

Only train after the preflight confirms a valid cohort, quantitative MRI at every anchor, available training MRI, and disjoint patient splits:

```bash
nacc-final --config /private/path/final.private.json --out /private/path/nacc-run --train
```

The completed run validated 75 participants (55 negatives, 20 progressions), with a single 52/11/12 patient-level split. It retained 149 of 152 dictionary-approved MRI features from 69 available training MRI visits. The test set contains only 12 participants and 3 progressions; any metrics and XAI must be reported with this uncertainty and must not support clinical utility or calibration claims. The primary model test results have already been inspected; do not use that test set for model selection. See [the latest preflight status](docs/workstream2_status.txt).

## Results and Exploratory Comparisons

The primary multimodal GRU outputs are stored locally under `artifacts/private_runs/freeze74/primary_multimodal/`. The six planned comparison arms (clinical/MRI/multimodal logistic regression on the last visit, plus clinical/MRI/multimodal longitudinal GRUs) are stored under `artifacts/private_runs/freeze74/exploratory_comparisons_2026-10-28/`. Both locations are Git-ignored because they contain model checkpoints and participant-derived artifacts.

To reproduce the exploratory comparisons with the fixed split:

```bash
nacc-compare \
	--config "$HOME/Library/Application Support/nacc-progression-private/final.private.json" \
	--out "$PWD/artifacts/private_runs/freeze74/new_comparison_run" \
	--bootstrap-repetitions 1000
```

The comparison command refuses to overwrite an existing run. Because the test set was already consulted for the primary model, all comparison test metrics are exploratory and must not be used to select a final model. A confirmatory comparison needs a newly reserved independent test set.

## Data Protection

All NACC source files and participant-level derivatives remain restricted, even after pseudonymization. Do not commit raw identifiers, cohort/visit tables, split membership, secret keys, preprocessing objects, or model checkpoints. See [data protection notes](docs/data_protection.md).

## Author

Nada Belarbi
