# Explainable Multimodal Longitudinal Medical Imaging

An early-stage research project exploring how medical images and clinical data collected across multiple patient visits can be combined to predict disease progression and make model predictions more understandable.

## Objective

The project aims to investigate two complementary questions:

1. Can combining medical imaging and clinical information across visits improve disease-progression prediction?
2. How can we inspect which visits, image regions, and clinical variables influence a prediction?

The initial candidate use case is progression from Mild Cognitive Impairment (MCI) to Alzheimer’s disease. The final task and prediction horizon remain to be defined.

## Core Concepts

- **Longitudinal learning:** using repeated observations of the same patient over time.
- **Multimodal learning:** combining medical images with clinical information.
- **Explainable AI:** examining which inputs influence model predictions and assessing the reliability of those explanations.

Explanations will describe model behavior; they should not be interpreted as evidence of clinical causality.

## Preliminary Approach

The initial architecture under consideration includes:

1. An imaging encoder to extract features from each MRI.
2. A clinical encoder to represent the clinical variables available at each visit.
3. A fusion module to combine both representations.
4. A temporal model to process the sequence of visits.
5. A prediction head to estimate risk for a defined progression event.
6. Explainability methods to investigate spatial, temporal, and clinical contributions.

These choices are provisional and will be refined according to data availability and baseline experiments.

## Planned Evaluation

The evaluation will investigate:

- Predictive performance and calibration.
- The added value of patient history compared with a single visit.
- The added value of combining imaging and clinical data.
- The fidelity and stability of explanations.

The protocol will use patient-level data splits and restrict each prediction to information available at prediction time.

## Roadmap

- [ ] Review the relevant concepts and literature.
- [ ] Define the prediction task, horizon, and cohort.
- [ ] Confirm dataset access and permitted use.
- [ ] Explore longitudinal clinical data.
- [ ] Build reproducible preprocessing and data splits.
- [ ] Implement simple reference models.
- [ ] Develop the multimodal longitudinal model.
- [ ] Evaluate predictions and explanations.
- [ ] Build an interactive research prototype.


## Intended Use

This project is intended for research and education. It is not a clinical diagnostic tool.

## Author

Nada Belarbi
