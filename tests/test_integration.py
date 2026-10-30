"""Artificial fixtures only. These are not NACC training results."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from nacc_progression.identity import subject_key
from nacc_progression.integration import prepare, read_table, DataBlocked, CLINICAL_ORDER
from nacc_progression.run_final import execute

SECRET = b'artificial-fixture-secret-only!!!'


def fixture():
    clinical, mri, splits = [], [], []
    for i in range(6):
        split = ('train', 'validation', 'test')[i // 2]
        site, participant = str(i // 2), '000' + str(i % 2)
        splits.append(dict(NACCADC=site, NACCID=participant, split=split))
        for j, date in enumerate(('2020-01-01', '2021-01-01')):
            clinical.append(dict(NACCADC=site, NACCID=participant, visit_date=date,
                mri_date=date, is_anchor=j == 1, label=i % 2,
                age=60+i+j, sex='F' if i % 2 else 'M', education=12+i,
                cdr_global=.5, cdr_sum_boxes=1+j, mmse=28-j))
            mri.append(dict(NACCADC=site, NACCID=participant, SCANDT=date,
                            NACCMVOL=1, hippocampus=10+i+j, constant=5))
    schema = dict(freeze=74, mri_family='mixed_protocol_quantitative',
        clinical_order=list(CLINICAL_ORDER), clinical_features=list(CLINICAL_ORDER),
        mri_features=['hippocampus', 'constant'], mri_date_column='SCANDT',
        mri_availability_column='NACCMVOL',
        minimum_sequence_visits=2, matching_window_days=90, horizon_days=730,
        codebook_reference='ARTIFICIAL FIXTURE ONLY',
        variables={x: {'missing_codes': []} for x in (*CLINICAL_ORDER, 'hippocampus', 'constant')})
    schema['variables']['sex']['encoding'] = {'M': 0, 'F': 1}
    return pd.DataFrame(clinical), pd.DataFrame(mri), pd.DataFrame(splits), schema


def test_composite_identity():
    assert subject_key('1', '23', SECRET) != subject_key('12', '3', SECRET)
    assert subject_key('1', '001', SECRET) != subject_key('2', '001', SECRET)
    assert subject_key('1', '001', SECRET) == subject_key('1', '001', SECRET)
    from nacc_progression.workstream1.pipeline import _subject_key
    assert _subject_key('1\x1f001', SECRET) == subject_key('1', '001', SECRET)


def test_summary_availability_is_not_a_measurement():
    c, m, s, schema = fixture()
    m.loc[0, 'NACCMVOL'] = 0
    sequences, _, counts = prepare(c, m, s, schema, SECRET)
    subject = subject_key('0', '0000', SECRET)
    patient = next(item for item in sequences['train'] if item.subject_key == subject)
    assert patient.imaging_missing_mask[0].all()
    np.testing.assert_array_equal(patient.imaging_values[0], np.zeros(1, dtype=np.float32))
    assert counts['by_split']['train']['visits_without_quantitative_mri'] == 1


def test_historical_visit_without_mri_is_neutral_but_anchor_mri_is_required():
    c, m, s, schema = fixture()
    c.loc[(c.NACCID == '0000') & ~c.is_anchor, 'mri_date'] = ''
    m = m.loc[~((m.NACCID == '0000') & (m.SCANDT == '2020-01-01'))].copy()

    sequences, _, _ = prepare(c, m, s, schema, SECRET)
    subject = subject_key('0', '0000', SECRET)
    patient = next(item for item in sequences['train'] if item.subject_key == subject)
    assert patient.imaging_missing_mask[0].all()
    np.testing.assert_array_equal(patient.imaging_values[0], np.zeros(1, dtype=np.float32))
    assert not patient.imaging_missing_mask[1].all()

    anchor_missing = c.copy()
    anchor_missing.loc[(anchor_missing.NACCID == '0000') & anchor_missing.is_anchor, 'mri_date'] = ''
    with pytest.raises(DataBlocked, match='ANCHOR_WITHOUT_QUANTITATIVE_MRI'):
        prepare(anchor_missing, m, s, schema, SECRET)


def test_masks_and_twenty_percent_boundary():
    c, m, s, schema = fixture()
    extra_c, extra_m = c.iloc[[0]].copy(), m.iloc[[0]].copy()
    extra_c['visit_date'] = extra_c['mri_date'] = '2019-01-01'
    extra_m['SCANDT'] = '2019-01-01'
    c, m = pd.concat([c, extra_c], ignore_index=True), pd.concat([m, extra_m], ignore_index=True)
    m['constant'] = np.arange(len(m)) + 5
    m.loc[0, 'hippocampus'] = -99
    schema['variables']['hippocampus']['missing_codes'] = [-99]
    seq, fit, _ = prepare(c, m, s, schema, SECRET)
    assert fit['mri_feature_order'] == ['hippocampus', 'constant']
    assert sum(int(x.imaging_missing_mask.sum()) for x in seq['train']) == 1
    assert all(np.isfinite(x.imaging_values).all() for x in seq['train'])


def test_train_only_and_contract():
    c, m, s, schema = fixture()
    m.loc[0, 'hippocampus'] = -99
    schema['variables']['hippocampus']['missing_codes'] = [-99]
    # 25% missingness removes this sole variable; never fit on validation instead.
    with pytest.raises(DataBlocked, match='NO_RETAINED'):
        prepare(c, m, s, schema, SECRET)
    c, m, s, schema = fixture()
    seq, fit, counts = prepare(c, m, s, schema, SECRET)
    m.loc[4:, 'hippocampus'] = 1e9
    c.loc[4:, 'age'] = 1e9
    _, altered, _ = prepare(c, m, s, schema, SECRET)
    for name in ('clinical', 'imaging'):
        np.testing.assert_array_equal(fit[name][1].mean_, altered[name][1].mean_)
    assert fit['mri_feature_order'] == ['hippocampus']
    assert counts['by_split']['train']['exact_matches'] == 4
    assert seq['train'][0].clinical_values.shape == (2, 6)
    assert seq['train'][0].delta_time[0, 0] == 0
    keys = [{x.subject_key for x in seq[p]} for p in ('train', 'validation', 'test')]
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]


@pytest.mark.parametrize('failure', ['duplicate_split', 'bad_date',
    'future_mri', 'wrong_source', 'empty_train', 'missing_sentinel', 'unknown_sex', 'missing_anchor'])
def test_fail_closed(failure):
    c, m, s, schema = fixture()
    if failure == 'duplicate_split': s = pd.concat([s, s.iloc[:1]])
    if failure == 'bad_date': c.loc[0, 'mri_date'] = '01/02/2020'
    if failure == 'future_mri': c.loc[1, 'mri_date'] = '2021-01-02'
    if failure == 'wrong_source': schema['mri_family'] = 'clariti'
    if failure == 'empty_train': m = m.iloc[4:]
    if failure == 'missing_sentinel': del schema['variables']['age']['missing_codes']
    if failure == 'unknown_sex': c.loc[0, 'sex'] = 'UNKNOWN'
    if failure == 'missing_anchor': m = m.drop(index=1)
    with pytest.raises(DataBlocked) as caught:
        prepare(c, m, s, schema, SECRET)
    if failure == 'empty_train':
        assert caught.value.counts['by_split']['train']['exact_matches'] == 0


def test_identical_duplicate_mri_rows_are_removed_with_audit_count():
    c, m, s, schema = fixture()
    m = pd.concat([m, m.iloc[:1]], ignore_index=True)

    sequences, _, counts = prepare(c, m, s, schema, SECRET)

    assert counts['mri_rows_before_exact_deduplication'] == len(m)
    assert counts['exact_duplicate_mri_rows_removed'] == 1
    assert sum(len(partition) for partition in sequences.values()) == len(s)


def test_conflicting_duplicate_mri_rows_block_preflight():
    c, m, s, schema = fixture()
    duplicate = m.iloc[[0]].copy()
    duplicate['hippocampus'] = duplicate['hippocampus'] + 100
    m = pd.concat([m, duplicate], ignore_index=True)

    with pytest.raises(DataBlocked, match='Conflicting MRI rows'):
        prepare(c, m, s, schema, SECRET)


def test_csv_parquet_roundtrip(tmp_path):
    c, _, _, _ = fixture()
    c.to_csv(tmp_path / 'c.csv', index=False)
    c.to_parquet(tmp_path / 'c.parquet', index=False)
    assert read_table(tmp_path / 'c.csv').NACCID.tolist() == read_table(tmp_path / 'c.parquet').NACCID.tolist()
    assert read_table(tmp_path / 'c.csv').NACCID.iloc[0] == '0000'
    c.NACCID = c.NACCID.astype(int)
    c.to_parquet(tmp_path / 'numeric.parquet', index=False)
    with pytest.raises(DataBlocked): read_table(tmp_path / 'numeric.parquet')


def test_full_synthetic_chain(tmp_path):
    c, m, s, schema = fixture()
    inputs = {}
    for name, table in [('clinical', c), ('mri', m), ('splits', s)]:
        path = tmp_path / (name + '.parquet')
        table.to_parquet(path, index=False)
        inputs[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    secret = tmp_path / 'secret.bin'
    secret.write_bytes(SECRET)
    secret.chmod(0o600)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(authorized_use_confirmed=True, inputs=inputs,
        secret_path=str(secret), schema=schema, training=dict(max_epochs=2, patience=1, seed=42))))
    out = tmp_path / 'restricted'
    assert execute(config, out)['status'] == 'PREFLIGHT_VALID'
    assert execute(config, out, train=True)['status'] == 'FINAL_TRAINED'
    report = json.loads((out / 'evaluation.json').read_text())
    assert report['xai']['mri_feature_order'] == ['hippocampus']
    assert 'subject_key' not in (out / 'evaluation.json').read_text()
    assert (out / 'training_history.json').is_file()
    assert (out / 'learning_curve.png').is_file()
    assert (out / 'xai_mri_top_features.png').is_file()
    assert (out / 'xai_clinical_features.png').is_file()
    assert (out / 'xai_visit_importance.png').is_file()
    assert report['test_sample_limits']['participants'] == 2
    assert report['test_sample_limits']['progressions'] == 1
    with pytest.raises(FileExistsError): execute(config, out, train=True)


def test_raw_freeze74_sources_build_cohort_and_preflight(tmp_path):
    from tests.test_pipeline import _make_data
    from nacc_progression.source_pipeline import prepare_from_raw_sources

    clinical, mri = _make_data(n=30)
    mri["HIPPOVOL"] = 100.0 + np.arange(len(mri), dtype=float)
    clinical_path = tmp_path / "clinical_raw.csv"
    mri_path = tmp_path / "mri_raw.csv"
    clinical.to_csv(clinical_path, index=False)
    mri.to_csv(mri_path, index=False)

    cohort_config = {
        "source_freeze": "NACC Freeze 74 (synthetic test only)",
        "clinical": {
            "participant_id": "NACCID",
            "site_id": "NACCADC",
            "visit_date": "VISITDATE",
            "diagnosis": "NACCUDSD",
            "alzheimers_etiology": "NACCALZD",
            "features": ["NACCAGE", "NACCSEX", "EDUC", "CDRGLOB", "CDRSUM", "NACCMMSE"],
        },
        "mri": {
            "participant_id": "NACCID",
            "site_id": "NACCADC",
            "year": "MRIYR",
            "month": "MRIMO",
            "day": "MRIDY",
            "availability_column": "NACCMVOL",
        },
        "diagnosis_codes": {"mci": 3, "dementia": 4, "alzheimers_etiology": 1},
        "feature_missing_codes": {
            "NACCAGE": [], "NACCSEX": [8, 9], "EDUC": [99],
            "CDRGLOB": [99], "CDRSUM": [99], "NACCMMSE": [88, 95, 96, 97, 98, -4],
        },
        "cohort": {
            "max_mri_clinical_delta_days": 90,
            "progression_window_days": 730,
            "minimum_sequence_visits": 2,
            "maximum_sequence_visits": 3,
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "random_seed": 42,
        },
    }
    cohort_config_path = tmp_path / "cohort.json"
    cohort_config_path.write_text(json.dumps(cohort_config), encoding="utf-8")
    mri_schema = {
        "freeze": 74,
        "availability_column": "NACCMVOL",
        "feature_count": 1,
        "feature_groups": {"8888.8888|9999.9999": ["HIPPOVOL"]},
    }
    mri_schema_path = tmp_path / "mri_schema.json"
    mri_schema_path.write_text(json.dumps(mri_schema), encoding="utf-8")

    sequences, fitted, counts = prepare_from_raw_sources(
        clinical_path,
        mri_path,
        cohort_config_path,
        mri_schema_path,
        {
            "age": "NACCAGE", "sex": "NACCSEX", "education": "EDUC",
            "cdr_global": "CDRGLOB", "cdr_sum_boxes": "CDRSUM", "mmse": "NACCMMSE",
        },
        SECRET,
    )

    assert counts["cohort_audit"]["eligible_participants"] == 30
    assert set(sequences) == {"train", "validation", "test"}
    assert counts["mri_missingness_denominator"] > 0
    assert fitted["mri_feature_order"] == ["HIPPOVOL"]
    assert all(
        len(history.imaging_values) <= 3
        and history.clinical_values.shape[1] == 6
        for partition in sequences.values()
        for history in partition
    )
