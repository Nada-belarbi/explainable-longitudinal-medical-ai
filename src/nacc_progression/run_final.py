"""Explicit preflight/train entry point for the single approved experiment."""
import argparse
import hashlib
import json
import os
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .integration import DataBlocked, prepare, read_table, CLINICAL_ORDER
from .evaluation import (
    bootstrap_confidence_intervals,
    calculate_metrics,
    select_balanced_accuracy_threshold,
)
from .explainability import explain_patient
from .model import ModelConfig
from .training import TrainingConfig, train_model, predict_probabilities
from .source_pipeline import prepare_from_raw_sources


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _write_learning_curve(training_losses, validation_losses, output_path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(training_losses) + 1)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, training_losses, label='Training BCE')
    axis.plot(epochs, validation_losses, label='Validation BCE')
    axis.set(xlabel='Epoch', ylabel='Binary cross-entropy', title='Training history')
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_xai_figures(explanations, sequences, fitted, output):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mri_names = fitted['mri_feature_order']
    clinical_names = fitted['clinical_feature_order']
    mean_mri = np.mean([item.mri_feature_importance for item in explanations], axis=0)
    mean_clinical = np.mean(
        [item.clinical_feature_importance for item in explanations], axis=0
    )
    for values, names, filename, title in (
        (mean_mri, mri_names, 'xai_mri_top_features.png', 'MRI regional measures'),
        (mean_clinical, clinical_names, 'xai_clinical_features.png', 'Clinical variables'),
    ):
        order = np.argsort(values)[-min(20, len(values)):]
        figure, axis = plt.subplots(figsize=(8, max(3.5, len(order) * 0.28)))
        axis.barh([names[index] for index in order], values[order])
        axis.set(xlabel='Mean absolute Integrated Gradients', title=title)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)

    max_visits = max(sequence.visit_count for sequence in sequences)
    visit_values = [
        float(np.mean([
            explanation.visit_importance[index]
            for explanation, sequence in zip(explanations, sequences)
            if sequence.visit_count > index
        ]))
        for index in range(max_visits)
    ]
    figure, axis = plt.subplots(figsize=(6, 3.5))
    positions = np.arange(1, max_visits + 1)
    axis.bar(positions, visit_values)
    axis.set(
        xlabel='Visit position in available history',
        ylabel='Mean absolute attribution',
        title='Visit contribution (test set)',
        xticks=positions,
    )
    figure.tight_layout()
    figure.savefig(output / 'xai_visit_importance.png', dpi=160)
    plt.close(figure)
    return visit_values


def _finite_interval(interval):
    return [float(value) if np.isfinite(value) else None for value in interval]


def execute(config_path, output, train=False):
    # Private outputs must live outside the code snapshot / repository.
    repo = Path(__file__).resolve().parents[2]
    output = Path(output).resolve()
    if output == repo or repo in output.parents:
        raise DataBlocked('Private outputs must be outside the code repository')
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config_path == repo or repo in config_path.parents:
        raise DataBlocked('Private configuration must stay outside the code repository')
    if config.get('authorized_use_confirmed') is not True:
        raise DataBlocked('Authorized local NACC use must be confirmed in private configuration')
    secret_path = Path(config['secret_path'])
    if not secret_path.is_absolute() or repo in secret_path.parents or secret_path.stat().st_mode & 0o077:
        raise DataBlocked('Secret must be an absolute private path with mode 0600')

    inputs = config['inputs']
    raw_source_keys = {'clinical_raw', 'mri_raw', 'cohort_config', 'mri_schema'}
    if raw_source_keys.issubset(inputs):
        for name in raw_source_keys:
            spec = inputs[name]
            if not Path(spec['path']).is_absolute() or _digest(spec['path']) != spec['sha256']:
                raise DataBlocked('Input must have an absolute path and match its approved SHA-256')
        sequences, fitted, counts = prepare_from_raw_sources(
            inputs['clinical_raw']['path'],
            inputs['mri_raw']['path'],
            inputs['cohort_config']['path'],
            inputs['mri_schema']['path'],
            config['clinical_feature_columns'],
            secret_path.read_bytes(),
        )
        input_hashes = {name: inputs[name]['sha256'] for name in raw_source_keys}
    else:
        tables = []
        for name in ('clinical', 'mri', 'splits'):
            spec = inputs[name]
            if not Path(spec['path']).is_absolute() or _digest(spec['path']) != spec['sha256']:
                raise DataBlocked('Input must have an absolute path and match its approved SHA-256')
            tables.append(read_table(spec['path']))
        sequences, fitted, counts = prepare(*tables, config['schema'], secret_path.read_bytes())
        input_hashes = {name: inputs[name]['sha256'] for name in ('clinical', 'mri', 'splits')}
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    preflight = {
        'status': 'PREFLIGHT_VALID',
        'counts': counts,
        'input_sha256': input_hashes,
        'configuration_sha256': _digest(config_path),
    }
    (output / 'preflight.json').write_text(json.dumps(preflight, indent=2, allow_nan=False))
    if not train:
        return preflight
    # A reused directory cannot silently rerun the final test.
    (output / 'RUN_STARTED').open('x').close()
    model_config = ModelConfig(imaging_features=len(fitted['mri_feature_order']),
                               imaging_feature_names=tuple(fitted['mri_feature_order']))
    training_config = TrainingConfig(**config['training'])
    with (output / 'preprocessing.pkl').open('xb') as stream:
        pickle.dump(fitted, stream)
    result = train_model(sequences['train'], sequences['validation'], model_config,
                         training_config, output / 'model.pt')
    training_history = {
        'best_epoch': result.best_epoch,
        'best_validation_loss': result.best_validation_loss,
        'training_losses': result.training_losses,
        'validation_losses': result.validation_losses,
    }
    (output / 'training_history.json').write_text(
        json.dumps(training_history, indent=2, allow_nan=False)
    )
    _write_learning_curve(
        result.training_losses, result.validation_losses, output / 'learning_curve.png'
    )
    validation, _ = predict_probabilities(result.model, sequences['validation'])
    threshold = select_balanced_accuracy_threshold(
        [x.label for x in sequences['validation']], validation)
    (output / 'TEST_STARTED').open('x').close()
    probabilities, _ = predict_probabilities(result.model, sequences['test'])
    test_labels = [x.label for x in sequences['test']]
    test_keys = [x.subject_key for x in sequences['test']]
    metrics = calculate_metrics(test_labels, probabilities, threshold)
    confidence_intervals = bootstrap_confidence_intervals(
        test_labels, probabilities, test_keys, threshold, repetitions=2000,
        seed=training_config.seed,
    )
    test_sequence_count = sum(sequence.visit_count for sequence in sequences['test'])
    masked_mri_visits = sum(
        int(sequence.imaging_missing_mask.all(axis=1).sum())
        for sequence in sequences['test']
    )
    explanations = [explain_patient(result.model, x, clinical_feature_names=CLINICAL_ORDER)
                    for x in sequences['test']]
    visit_importance = _write_xai_figures(
        explanations, sequences['test'], fitted, output
    )
    report = {'status': 'FINAL_TRAINED', 'freeze': 74,
        'mri_family': 'mixed_protocol_quantitative', 'counts': counts,
        'input_sha256': input_hashes,
        'configuration_sha256': _digest(config_path),
        'model_config': model_config.to_dict(), 'training_config': asdict(training_config),
        'best_epoch': result.best_epoch, 'threshold_from_validation': threshold,
        'test_metrics': metrics,
        'test_metric_confidence_intervals_95_percent': {
            name: _finite_interval(interval)
            for name, interval in confidence_intervals.items()
        },
        'test_sample_limits': {
            'participants': len(sequences['test']),
            'progressions': int(sum(test_labels)),
            'clinical_visits': test_sequence_count,
            'visits_without_quantitative_mri': masked_mri_visits,
            'calibration_or_clinical_utility_claim': False,
        },
        'xai': {
            'mri_feature_order': fitted['mri_feature_order'],
            'clinical_order': list(CLINICAL_ORDER),
            'mean_mri_importance': np.mean([e.mri_feature_importance for e in explanations], axis=0).tolist(),
            'mean_clinical_importance': np.mean([e.clinical_feature_importance for e in explanations], axis=0).tolist(),
            'mean_visit_importance_by_sequence_position': visit_importance,
            'visits_without_mri_are_neutral_masked_inputs': True,
            'max_abs_completeness_delta_logit': max(abs(e.completeness_delta_logit) for e in explanations),
            'interpretation': 'Model behavior conditional on fixed missingness masks; not clinical causality'}}
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return {'status': 'FINAL_TRAINED', 'best_epoch': result.best_epoch}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    try:
        print(json.dumps(execute(args.config, args.out, args.train), indent=2))
    except DataBlocked as error:
        print(json.dumps({'status': 'BLOCKED', 'reason': str(error), 'counts': error.counts}))
        raise SystemExit(2)
    except Exception:
        # Parser, pandas and OS exceptions can contain participant values or paths.
        print(json.dumps({'status': 'BLOCKED', 'reason': 'Input/configuration/runtime error; inspect privately'}))
        raise SystemExit(2)


if __name__ == '__main__':
    main()
