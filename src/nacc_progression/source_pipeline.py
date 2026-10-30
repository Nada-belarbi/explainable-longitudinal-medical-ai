"""Build the approved Freeze 74 cohort and model sequences from authorized raw CSVs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from nacc_progression.integration import CLINICAL_ORDER, prepare
from nacc_progression.workstream1.pipeline import (
    build_cohort,
    load_config as load_cohort_config,
    make_patient_splits,
    match_mri_to_clinical,
)


def _read_sources(
    clinical_path: str | Path,
    mri_path: str | Path,
    cohort_config: dict[str, Any],
    mri_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical_config = cohort_config["clinical"]
    mri_config = cohort_config["mri"]
    clinical_columns = {
        clinical_config[key]
        for key in (
            "participant_id",
            "site_id",
            "visit_date",
            "diagnosis",
            "alzheimers_etiology",
        )
    }
    clinical_columns.update(clinical_config["features"])
    mri_columns = {
        mri_config[key] for key in ("participant_id", "site_id", "year", "month", "day")
    }
    mri_columns.add(mri_config["availability_column"])
    mri_columns.update(mri_features)

    clinical = pd.read_csv(
        clinical_path,
        usecols=sorted(clinical_columns),
        dtype={
            clinical_config["participant_id"]: "string",
            clinical_config["site_id"]: "string",
        },
        low_memory=False,
    )
    mri = pd.read_csv(
        mri_path,
        usecols=sorted(mri_columns),
        dtype={
            mri_config["participant_id"]: "string",
            mri_config["site_id"]: "string",
        },
        low_memory=False,
    )
    return clinical, mri


def _dated_mri_table(
    mri: pd.DataFrame,
    cohort_config: dict[str, Any],
    mri_features: list[str],
) -> pd.DataFrame:
    mri_config = cohort_config["mri"]
    year = pd.to_numeric(mri[mri_config["year"]], errors="coerce")
    month = pd.to_numeric(mri[mri_config["month"]], errors="coerce")
    day = pd.to_numeric(mri[mri_config["day"]], errors="coerce")
    columns = {
        "NACCADC": mri[mri_config["site_id"]].astype("string"),
        "NACCID": mri[mri_config["participant_id"]].astype("string"),
        "mri_date": pd.to_datetime(
            {"year": year, "month": month, "day": day}, errors="coerce"
        ),
        "NACCMVOL": pd.to_numeric(
            mri[mri_config["availability_column"]], errors="coerce"
        ),
    }
    columns.update({feature: mri[feature] for feature in mri_features})
    result = pd.DataFrame(columns)
    result["mri_date"] = result["mri_date"].dt.strftime("%Y-%m-%d").fillna("")
    return result


def _adapter_inputs(
    sequences: pd.DataFrame,
    splits: dict[str, list[str]],
    mri: pd.DataFrame,
    cohort_config: dict[str, Any],
    mri_schema: dict[str, Any],
    clinical_feature_columns: dict[str, str],
    secret: bytes,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    clinical_config = cohort_config["clinical"]
    expected_map = {
        "age": clinical_config["features"][0],
        "sex": clinical_config["features"][1],
        "education": clinical_config["features"][2],
        "cdr_global": clinical_config["features"][3],
        "cdr_sum_boxes": clinical_config["features"][4],
        "mmse": clinical_config["features"][5],
    }
    if clinical_feature_columns != expected_map:
        raise ValueError("Private clinical feature mapping must match the six approved UDS variables")

    clinical = pd.DataFrame(
        {
            "NACCADC": sequences["clinical_NACCADC"].astype("string"),
            "NACCID": sequences["clinical_NACCID"].astype("string"),
            "visit_date": pd.to_datetime(sequences["visit_date"], errors="coerce")
            .dt.strftime("%Y-%m-%d")
            .fillna(""),
            "mri_date": pd.to_datetime(sequences["mri_date"], errors="coerce")
            .dt.strftime("%Y-%m-%d")
            .fillna(""),
            "is_anchor": sequences["is_anchor"].astype(bool),
            "label": sequences["label"].astype(int),
        }
    )
    for canonical, source in clinical_feature_columns.items():
        clinical[canonical] = sequences[f"clinical_{source}"]

    split_rows = []
    for split_name, participant_keys in splits.items():
        for composite_key in participant_keys:
            site, participant = str(composite_key).split("\x1f", maxsplit=1)
            split_rows.append(
                {"NACCADC": site, "NACCID": participant, "split": split_name}
            )
    split_table = pd.DataFrame(split_rows)

    feature_groups = mri_schema["feature_groups"]
    mri_features = [name for names in feature_groups.values() for name in names]
    if len(mri_features) != mri_schema["feature_count"] or len(set(mri_features)) != len(mri_features):
        raise ValueError("MRI feature schema count or uniqueness check failed")
    mri_table = _dated_mri_table(mri, cohort_config, mri_features)

    variable_schema: dict[str, dict[str, Any]] = {}
    for canonical, source in clinical_feature_columns.items():
        rule = {"missing_codes": cohort_config["feature_missing_codes"][source]}
        if canonical == "sex":
            rule["encoding"] = {"1": 0.0, "2": 1.0}
        variable_schema[canonical] = rule
    for group, names in feature_groups.items():
        missing_codes = [float(code) for code in group.split("|")]
        for name in names:
            variable_schema[name] = {"missing_codes": missing_codes}

    schema = {
        "freeze": 74,
        "mri_family": "mixed_protocol_quantitative",
        "horizon_days": int(cohort_config["cohort"]["progression_window_days"]),
        "minimum_sequence_visits": int(cohort_config["cohort"]["minimum_sequence_visits"]),
        "matching_window_days": int(cohort_config["cohort"]["max_mri_clinical_delta_days"]),
        "codebook_reference": "NACC UDSv4 Researcher's Data Dictionary (April 2026) and NACC Mixed Protocol MRI Researcher's Data Dictionary (2026)",
        "clinical_order": list(CLINICAL_ORDER),
        "clinical_features": list(CLINICAL_ORDER),
        "mri_date_column": "mri_date",
        "mri_availability_column": mri_schema["availability_column"],
        "mri_features": mri_features,
        "variables": variable_schema,
    }
    return clinical, mri_table, split_table, schema


def prepare_from_raw_sources(
    clinical_path: str | Path,
    mri_path: str | Path,
    cohort_config_path: str | Path,
    mri_schema_path: str | Path,
    clinical_feature_columns: dict[str, str],
    secret: bytes,
) -> tuple[dict[str, list], dict[str, Any], dict[str, Any]]:
    """Construct labels, one fixed split, training-only transforms, and audit in memory."""

    cohort_config = load_cohort_config(cohort_config_path)
    mri_schema = json.loads(Path(mri_schema_path).read_text(encoding="utf-8"))
    if not cohort_config["source_freeze"].startswith("NACC Freeze 74"):
        raise ValueError("Clinical source config is not Freeze 74")
    if mri_schema.get("freeze") != 74:
        raise ValueError("MRI feature dictionary is not Freeze 74")
    mri_features = [name for group in mri_schema["feature_groups"].values() for name in group]
    clinical_source, mri_source = _read_sources(
        clinical_path, mri_path, cohort_config, mri_features
    )
    matched, match_audit = match_mri_to_clinical(
        clinical_source, mri_source, cohort_config
    )
    manifest, visit_sequences, cohort_audit = build_cohort(
        matched, clinical_source, cohort_config
    )
    splits = make_patient_splits(manifest, cohort_config)
    clinical, mri, split_table, schema = _adapter_inputs(
        visit_sequences,
        splits,
        mri_source,
        cohort_config,
        mri_schema,
        clinical_feature_columns,
        secret,
    )
    partitioned_sequences, fitted, counts = prepare(
        clinical, mri, split_table, schema, secret
    )
    counts["cohort_audit"] = cohort_audit
    counts["matching_audit"] = match_audit
    counts["split_subject_counts"] = {
        name: len(values) for name, values in splits.items()
    }
    counts["split_label_counts"] = {
        name: {
            str(label): sum(
                int(sequence.label == label) for sequence in partitioned_sequences[name]
            )
            for label in (0, 1)
        }
        for name in ("train", "validation", "test")
    }
    return partitioned_sequences, fitted, counts
