"""Headless reproducible entry point for the crash-test suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.reporting import export_csv, export_json, export_markdown
from src.runner import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the non-clinical medical AI crash-test demonstration.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/latest"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--risk-budget", type=float, default=0.05, help="Validation selective-risk budget, from 0 to 1.")
    args = parser.parse_args()
    result = run_experiment(ExperimentConfig(seed=args.seed, risk_budget=args.risk_budget))
    export_json(result, args.output_dir / "results.json")
    export_csv(result, args.output_dir / "perturbation_results.csv")
    export_markdown(result, args.output_dir / "report.md")
    print(f"Crash-test suite completed; reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
