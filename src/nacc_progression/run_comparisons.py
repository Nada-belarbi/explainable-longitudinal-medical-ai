"""Run the planned six model-comparison arms on the frozen NACC74 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from nacc_progression.evaluation import ModelEvaluation, run_model_comparison
from nacc_progression.integration import DataBlocked
from nacc_progression.model import ModelConfig
from nacc_progression.source_pipeline import prepare_from_raw_sources
from nacc_progression.training import TrainingConfig


def _digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_comparison_plot(results: dict[str, ModelEvaluation], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    metrics = ("auroc", "auprc", "balanced_accuracy")
    x = np.arange(len(names), dtype=float)
    width = 0.24
    figure, axis = plt.subplots(figsize=(11, 5))
    for index, metric in enumerate(metrics):
        values = [results[name].test_metrics[metric] for name in names]
        axis.bar(x + (index - 1) * width, values, width, label=metric.upper())
    axis.set_xticks(x, names, rotation=30, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Exploratory test-set metric")
    axis.set_title("Exploratory six-arm comparison; test set previously inspected")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def execute(config_path: str | Path, output_path: str | Path, bootstrap_repetitions: int = 1000):
    project_root = Path(__file__).resolve().parents[2]
    private_root = (project_root / "artifacts" / "private_runs").resolve()
    output = Path(output_path).resolve()
    if private_root not in output.parents:
        raise DataBlocked("Comparison outputs must be below artifacts/private_runs")
    config_file = Path(config_path).resolve()
    if project_root in config_file.parents:
        raise DataBlocked("Private config must stay outside the code repository")
    if output.exists():
        raise DataBlocked("Output directory already exists; do not silently repeat test comparisons")

    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("authorized_use_confirmed") is not True:
        raise DataBlocked("Authorized local NACC use must be confirmed in private configuration")
    secret_path = Path(config["secret_path"])
    if (
        not secret_path.is_absolute()
        or project_root in secret_path.parents
        or secret_path.stat().st_mode & 0o077
    ):
        raise DataBlocked("Secret must be outside the repository and have mode 0600")

    inputs = config["inputs"]
    required_inputs = {"clinical_raw", "mri_raw", "cohort_config", "mri_schema"}
    if not required_inputs.issubset(inputs):
        raise DataBlocked("All four raw-source and schema inputs are required")
    for name in required_inputs:
        item = inputs[name]
        if not Path(item["path"]).is_absolute() or _digest(item["path"]) != item["sha256"]:
            raise DataBlocked("Input must have an absolute path and match its approved SHA-256")

    os.umask(0o077)
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(private_root, 0o700)
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)
    (output / "COMPARISON_STARTED").open("x").close()

    sequences, fitted, counts = prepare_from_raw_sources(
        inputs["clinical_raw"]["path"],
        inputs["mri_raw"]["path"],
        inputs["cohort_config"]["path"],
        inputs["mri_schema"]["path"],
        config["clinical_feature_columns"],
        secret_path.read_bytes(),
    )
    model_config = ModelConfig(
        imaging_features=len(fitted["mri_feature_order"]),
        imaging_feature_names=tuple(fitted["mri_feature_order"]),
    )
    training_config = TrainingConfig(**config["training"])
    results = run_model_comparison(
        sequences["train"],
        sequences["validation"],
        sequences["test"],
        model_config,
        training_config,
        checkpoint_dir=output / "checkpoints",
        bootstrap_repetitions=bootstrap_repetitions,
    )
    report = {
        "status": "EXPLORATORY_COMPARISONS_COMPLETE",
        "interpretation_warning": (
            "The primary model's test metrics were already inspected before these comparisons. "
            "Treat all test-arm results as exploratory; do not select a model or claim confirmatory performance from them."
        ),
        "protocol": {
            "freeze": 74,
            "mri_family": "mixed_protocol_quantitative",
            "split": "one stratified 70/15/15 participant split, seed 42",
            "test_used_for_model_selection": False,
            "training_config": asdict(training_config),
            "bootstrap_repetitions": bootstrap_repetitions,
        },
        "counts": counts,
        "input_sha256": {name: inputs[name]["sha256"] for name in sorted(required_inputs)},
        "configuration_sha256": _digest(config_file),
        "mri_feature_order": fitted["mri_feature_order"],
        "clinical_feature_order": fitted["clinical_feature_order"],
        "models": {name: asdict(result) for name, result in results.items()},
    }
    (output / "comparisons.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_comparison_plot(results, output / "comparison_metrics.png")
    return {
        "status": "EXPLORATORY_COMPARISONS_COMPLETE",
        "models": list(results),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run six exploratory NACC74 model comparisons on the fixed patient split."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    try:
        result = execute(args.config, args.out, args.bootstrap_repetitions)
        print(json.dumps(result, indent=2))
    except DataBlocked as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}))
        raise SystemExit(2)
    except Exception:
        print(json.dumps({"status": "BLOCKED", "reason": "Runtime error; inspect privately"}))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
