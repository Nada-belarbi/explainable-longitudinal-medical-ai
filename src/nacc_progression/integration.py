"""Strict raw-export boundary. No cohort/label creation and no source substitution.

Input schema and sentinel/category rules must come from the authorized delivery.
Only training rows determine feature selection, imputation and scaling.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .contracts import PatientSequence, MAX_VISITS
from .identity import subject_key

CLINICAL_ORDER = ('age', 'sex', 'education', 'cdr_global', 'cdr_sum_boxes', 'mmse')


class DataBlocked(ValueError):
    def __init__(self, reason, counts=None):
        super().__init__(reason)
        self.counts = counts or {}


def read_table(path, id_columns=('NACCADC', 'NACCID')):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        table = pd.read_csv(path, dtype={c: 'string' for c in id_columns},
                            keep_default_na=False, low_memory=False)
    elif path.suffix.lower() == '.parquet':
        table = pd.read_parquet(path, engine='pyarrow')
    else:
        raise DataBlocked('Only CSV or Parquet tables are supported')
    for col in id_columns:
        if col not in table or table[col].isna().any() or not table[col].map(
            lambda x: isinstance(x, str) and bool(x) and x == x.strip()).all():
            raise DataBlocked('Composite IDs must be complete strings; numeric Parquet IDs are ambiguous')
    return table


def _dates(values):
    # Explicit ISO only: never infer day/month or silently discard invalid dates.
    text = values.map(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x)
    if not text.map(lambda x: isinstance(x, str) and len(x) == 10).all():
        raise DataBlocked('Dates must be canonical YYYY-MM-DD')
    result = pd.to_datetime(text, format='%Y-%m-%d', errors='coerce')
    if result.isna().any() or not result.dt.strftime('%Y-%m-%d').eq(text).all():
        raise DataBlocked('Invalid or ambiguous dates')
    return result


def _numeric(frame, features, schema):
    columns = {}
    for name in features:
        rule = schema[name]
        if 'missing_codes' not in rule:
            raise DataBlocked('Every feature needs reviewed missing_codes, including an explicit empty list')
        values = frame[name].replace(rule['missing_codes'], np.nan).replace('', np.nan)
        if 'encoding' in rule:
            values = values.map(lambda x: str(int(x)) if isinstance(x, (int, float))
                                and np.isfinite(x) and float(x).is_integer()
                                else str(x) if pd.notna(x) else np.nan)
            unknown = values.notna() & ~values.isin(rule['encoding'])
            if unknown.any():
                raise DataBlocked('Unknown categorical code')
            values = values.map(rule['encoding'])
        numeric = pd.to_numeric(values, errors='coerce')
        if (values.notna() & numeric.isna()).any():
            raise DataBlocked('Unexpected nonnumeric feature value')
        columns[name] = numeric.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(columns, index=frame.index)


def prepare(clinical, mri, splits, schema, secret):
    """Return partitioned PatientSequences, train-fitted transformers, aggregate audit.

    Clinical rows are the ALREADY approved histories with one final is_anchor.
    Historical visits may lack MRI, but every anchor must have a quantitative
    MRI acquired no later than its clinical visit. This adapter never changes
    cohort membership or partitions.
    """
    if schema.get('freeze') != 74 or schema.get('mri_family') != 'mixed_protocol_quantitative':
        raise DataBlocked('Only Freeze 74 Mixed-protocol quantitative MRI is authorized')
    if schema.get('clinical_order') != list(CLINICAL_ORDER) or not schema.get('codebook_reference'):
        raise DataBlocked('Reviewed clinical order and authoritative codebook reference required')
    minimum = schema.get('minimum_sequence_visits')
    window = schema.get('matching_window_days')
    if type(minimum) is not int or not 1 <= minimum <= MAX_VISITS or type(window) is not int or window < 0:
        raise DataBlocked('Approved history length and MRI matching window required')
    if schema.get('horizon_days') != 730:
        raise DataBlocked('The fixed anchor-outcome horizon must be 730 days')
    cf = schema['clinical_features']
    if len(cf) != 6 or len(set(cf)) != 6 or 'encoding' not in schema['variables'][cf[1]]:
        raise DataBlocked('Six distinct clinical columns and explicit sex encoding required')
    mf = schema['mri_features']
    availability = schema.get('mri_availability_column')
    if availability != 'NACCMVOL':
        raise DataBlocked('Mixed-protocol calculated-measure availability must use NACCMVOL')
    if not mf or len(mf) != len(set(mf)) or set(mf) & set(cf):
        raise DataBlocked('Explicit distinct MRI feature whitelist required')
    tables = [t.copy() for t in (clinical, mri, splits)]
    for table in tables:
        for col in ('NACCADC', 'NACCID'):
            if col not in table or not table[col].map(
                lambda x: isinstance(x, str) and bool(x) and x == x.strip()).all():
                raise DataBlocked('Missing or noncanonical composite site/participant identity')
        table['subject_key'] = [subject_key(s, p, secret) for s, p in zip(table.NACCADC, table.NACCID)]
    clinical, mri, splits = tables
    raw_mri_row_count = len(mri)
    if splits.subject_key.duplicated().any():
        raise DataBlocked('Duplicate participants in splits (including repeated identical assignments)')
    if set(splits['split']) != {'train', 'validation', 'test'}:
        raise DataBlocked('Exactly train, validation and test partitions required')
    clinical['split'] = clinical.subject_key.map(splits.set_index('subject_key')['split'])
    if clinical['split'].isna().any():
        raise DataBlocked('Clinical participants missing from approved partitions')
    clinical['visit_date'] = _dates(clinical['visit_date'])
    raw_mri_dates = clinical['mri_date']
    has_mri_date = raw_mri_dates.notna() & raw_mri_dates.astype('string').str.strip().ne('').fillna(False)
    clinical['mri_date'] = pd.Series(pd.NaT, index=clinical.index, dtype='datetime64[ns]')
    clinical.loc[has_mri_date, 'mri_date'] = _dates(raw_mri_dates.loc[has_mri_date])
    mri['mri_date'] = _dates(mri[schema['mri_date_column']])
    if clinical.duplicated(['subject_key', 'visit_date']).any():
        raise DataBlocked('Duplicate clinical visits')
    duplicate_scan_rows = mri.duplicated(['subject_key', 'mri_date'], keep=False)
    duplicate_groups = mri.loc[duplicate_scan_rows].groupby(
        ['subject_key', 'mri_date'], sort=False, dropna=False
    )
    conflicting_duplicate_groups = 0
    for _, duplicate_group in duplicate_groups:
        if duplicate_group[[availability, *mf]].nunique(dropna=False).gt(1).any().any():
            conflicting_duplicate_groups += 1
    if conflicting_duplicate_groups:
        raise DataBlocked('Conflicting MRI rows share a composite identity and scan date')
    mri = mri.drop_duplicates(['subject_key', 'mri_date'], keep='first').copy()
    exact_duplicate_rows_removed = raw_mri_row_count - len(mri)
    if not clinical.is_anchor.map(lambda x: isinstance(x, (bool, np.bool_, int, np.integer)) and x in (0, 1)).all():
        raise DataBlocked('is_anchor must be boolean or integer 0/1, not a string')
    clinical['is_anchor'] = clinical.is_anchor.astype(bool)
    if not clinical.label.isin([0, 1]).all():
        raise DataBlocked('Labels must already be approved binary anchor outcomes')
    for _, history in clinical.groupby('subject_key'):
        history = history.sort_values('visit_date')
        if not minimum <= len(history) <= MAX_VISITS:
            raise DataBlocked('History length violates the approved cohort; no automatic truncation')
        if history.is_anchor.sum() != 1 or not history.iloc[-1].is_anchor or history.label.nunique() != 1:
            raise DataBlocked('One final anchor and a consistent anchor label required')
        if (history.mri_date.dropna() > history.iloc[-1].visit_date).any():
            raise DataBlocked('MRI after prediction anchor would leak future information')
    matched_date = clinical.mri_date.notna()
    if not (clinical.loc[matched_date, 'visit_date'] - clinical.loc[matched_date, 'mri_date']).dt.days.between(0, window).all():
        raise DataBlocked('Recorded MRI match violates the approved matching window')
    clinical = clinical.drop(columns=[x for x in mf if x in clinical])
    joined = clinical.merge(mri[['subject_key', 'mri_date', availability] + mf],
                            on=['subject_key', 'mri_date'], how='left',
                            validate='many_to_one', indicator=True)
    quantitative_available = (
        joined['_merge'].eq('both')
        & pd.to_numeric(joined[availability], errors='coerce').eq(1)
    )
    counts = {
        'clinical_visits': len(clinical),
        'mri_rows_before_exact_deduplication': raw_mri_row_count,
        'exact_duplicate_mri_rows_removed': exact_duplicate_rows_removed,
        'mri_rows': len(mri),
        'by_split': {},
    }
    for split in ('train', 'validation', 'test'):
        rows = joined[joined['split'].eq(split)]
        split_available = quantitative_available.loc[rows.index]
        counts['by_split'][split] = {'subjects': int(rows.subject_key.nunique()),
            'visits': len(rows), 'exact_matches': int(rows['_merge'].eq('both').sum()),
            'quantitative_available_visits': int(split_available.sum()),
            'visits_without_quantitative_mri': int((~split_available).sum())}
    anchor_rows = joined.loc[joined['is_anchor'].astype(bool)]
    if anchor_rows.empty or not quantitative_available.loc[anchor_rows.index].all():
        raise DataBlocked('BLOCKED_ANCHOR_WITHOUT_QUANTITATIVE_MRI', counts)
    if counts['by_split']['train']['quantitative_available_visits'] == 0:
        raise DataBlocked('BLOCKED_NO_MATCHED_TRAINING_SCANS', counts)
    train = joined['split'].eq('train')
    mri_train = train & quantitative_available
    cv = _numeric(joined, cf, schema['variables'])
    iv = _numeric(joined, mf, schema['variables'])
    iv.loc[~quantitative_available, :] = np.nan
    counts['mri_missingness_denominator'] = int(mri_train.sum())
    selected = [f for f in mf if iv.loc[mri_train, f].isna().mean() <= .20
                and iv.loc[mri_train, f].nunique() > 1]
    counts['candidate_mri_features'] = len(mf)
    counts['retained_mri_features'] = len(selected)
    if not selected:
        raise DataBlocked('BLOCKED_NO_RETAINED_TRAIN_FEATURES', counts)
    if cv.loc[train].isna().all().any():
        raise DataBlocked('All-missing clinical train feature', counts)
    if iv.loc[mri_train, selected].isna().all(axis=1).any():
        raise DataBlocked('Quantitative training MRI has no usable selected features', counts)
    result, fitted = {}, {'mri_feature_order': selected, 'clinical_feature_order': cf}
    transformed = []
    for name, values, fit_rows in [('clinical', cv, train), ('imaging', iv[selected], mri_train)]:
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        train_values = imputer.fit_transform(values.loc[fit_rows])
        scaler = StandardScaler().fit(train_values)
        scaled = scaler.transform(imputer.transform(values))
        if name == 'imaging':
            scaled[~quantitative_available.to_numpy(), :] = 0.0
        transformed.append((scaled, values.isna().to_numpy()))
        fitted[name] = (imputer, scaler)
    (cvalues, cmask), (ivalues, imask) = transformed
    for split in ('train', 'validation', 'test'):
        sequences = []
        for key, history in joined[joined['split'].eq(split)].groupby('subject_key'):
            history = history.sort_values('visit_date')
            idx = history.index.to_numpy()
            delta = history.visit_date.diff().dt.total_seconds().fillna(0).to_numpy() / (86400 * 365.25)
            sequences.append(PatientSequence(key, int(history.iloc[-1].label),
                ivalues[idx], imask[idx], cvalues[idx], cmask[idx], delta[:, None], tuple(selected)))
        if not sequences or {x.label for x in sequences} != {0, 1}:
            raise DataBlocked('Each final partition must contain both outcome classes', counts)
        result[split] = sequences
    return result, fitted, counts
