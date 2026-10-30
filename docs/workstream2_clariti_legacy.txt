NACC74 WORKSTREAM 2: MRI FEATURE PIPELINE
======================================
RELEASE STATUS: BLOCKED, NOT VALID FOR MODEL TRAINING

This package was constructed using the provided Member 1 NACC74 cohort,
Member 1 visit matches/splits, and CLARITI structural MRI measurements.
The correct explicit NACC74 filenames were used; their generic aliases
were byte-identical. No patient-level training MRI matched the Member 1
cohort, so it would be incorrect to compute a training imputer or scaler.

Verified facts
--------------
142 subjects, 325 Member 1 clinical visits.
99 train subjects (229 visits), 21 validation (48 visits), 22 test (48 visits).
7,396 CLARITI structural MRI scan rows.
One scan matched a Member 1 visit by subject and the recorded MRI date.
The matched visit is validation, not training.
ZERO train visits have matching anatomical MRI features.
240 quantitative anatomical candidate fields were identified.
The retained model feature count M is undefined until a valid training fit.
The Member 1 source subject_key field equals raw NACCID; this pipeline
replaces it with a private HMAC pseudonym in the exported visit table.

Required named outputs
----------------------
1. mri_feature_dictionary.json
   Lists all 240 anatomical candidates with human readable anatomical
   names, candidate order, excluded metadata, and UNSELECTED model status.
   Retained model indices, training medians and standardization values
   are intentionally null while the matching training dataset is absent.

2. mri_preprocessor.pkl
   An unfit configuration dictionary with imputer=None, scaler=None,
   training_only_fitted=False and an explicit failure reason. This is NOT
   a fitted scikit-learn transformer. Never use this to transform model data.

3. visits_imaging.parquet
   A genuine Parquet file with one record for each Member 1 clinical visit.
   325 rows, 15 columns, pseudonymous identifiers. One row has an actual
   matching scan and raw candidate anatomical measurements and masks.
   All imaging_vector and imaging_mask entries are NULL because no valid
   training-only preprocessor exists. Unmatched rows also have NULL raw
   arrays. Its columns are nullable UTF8; raw arrays are JSON encoded
   strings for this blocked audit export, not native list arrays.
   This is NOT an inference-ready visit table.

4. mri_audit.json
   Join statistics, split breakdown, source mismatch, filtering status,
   feature counts, and explicit blockers.

Additional files
----------------
build_mri_workstream2.py: reproducible fit-or-block pipeline.
simple_parquet_writer.py: small nullable text Parquet exporter used only
                         for the blocked audit table without pyarrow.
validate_parquet.py: smoke tests for this writer and its output structure.
visits_imaging_review.csv: restricted summary of subject/visit match status.

What is needed to finish the workstream
--------------------------------------
Upload the quantitative anatomical MRI analysis records corresponding to
Member 1's NACC74 visit matches. Member 1's data audit names
investigator_mri_nacc74.csv as its original MRI source, not the CLARITI
structural analysis file used here. Check whether that MRI source contains
quantitative regional measures or a linkage to the required anatomy file.
Alternatively, rebuild Member 1 visit matches, outcome cohort and the
patient partitions consistently for the CLARITI structural MRI scans.
Do not match just on NACCID or assign temporally distant MRI scans.

Once valid training scans exist, rerun with a compatible --mri input:
  python build_mri_workstream2.py --mri /private/path/compatible_mri.csv

The script applies the documented <=20% TRAIN visit missingness rule,
nonconstant selection, TRAIN median imputation, and TRAIN z score scaling.
If valid training matches exist, it requires pyarrow to write native list
columns for the actual model-ready Parquet file:
  pip install pyarrow
The outputs must be regenerated and downstream contracts rechecked.

Security and integration
------------------------
The generated Parquet and review CSV contain private participant visit
information; do not commit or publish them. The HMAC pseudonym secret
is stored separately in:
  /mnt/data/MRI_Workstream2_private_subject_key.bin
Never commit or publicly distribute this key. Reuse the same secret when
Member 1 rekeys its cohort and clinical visits; otherwise the patient keys
will not match. The JSON dictionary is not an authoritative NACC variable
codebook, so measurement units should be checked against the official
regional analysis data dictionary.

Definition of Done is NOT met. This is an accurate blocked handoff with
valid file structures, not a completed fitted MRI feature pipeline.
