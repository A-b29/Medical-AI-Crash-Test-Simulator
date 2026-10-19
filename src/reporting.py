"""Machine- and human-readable experiment exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def export_json(result: dict[str, object], path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return destination


def export_csv(result: dict[str, object], path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = result["perturbation_results"]
    if not isinstance(rows, list):
        raise TypeError("perturbation_results must be a list")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["perturbation"])
        writer.writeheader()
        writer.writerows(rows)
    return destination


def export_markdown(result: dict[str, object], path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    baseline = result["baseline_metrics"]
    manifest = result["manifest"]
    abstention = result["abstention"]
    lines = [
        "# Medical AI Crash-Test Report",
        "",
        str(result["project_notice"]),
        "",
        "## Experimental contract",
        "",
        f"- Dataset: `{manifest['dataset_id']}`; splits: {manifest['dataset_samples']}",
        f"- Model: `{manifest['model_id']}`",
        f"- Configuration: `{result['config']}`",
        "",
        "## Baseline test metrics",
        "",
        *[f"- {name}: {value:.4f}" for name, value in baseline.items()],
        "",
        "## Confidence and abstention",
        "",
        f"- Expected calibration error: {result['calibration']['expected_calibration_error']:.4f}",
        f"- Threshold selected only on: {abstention['selection_split']}",
        f"- Test selective metrics: {abstention['test_evaluation']}",
        "",
        "## Interpretation limits",
        "",
        "Synthetic tabular stressors measure sensitivity to these explicit transformations only. Prediction stability is not correctness, and this report does not establish clinical safety, fairness, or real-world performance.",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
