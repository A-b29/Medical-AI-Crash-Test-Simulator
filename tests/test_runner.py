from src.runner import ExperimentConfig, run_experiment


def test_end_to_end_runner_emits_three_perturbations_and_manifest():
    result = run_experiment(ExperimentConfig(seed=7, severities=(0.0, 0.5)))
    assert result["manifest"]["dataset_samples"]["test"] > 0
    assert len(result["perturbation_results"]) == 6
    assert {row["perturbation"] for row in result["perturbation_results"]} == {
        "gaussian_noise", "attenuation", "feature_dropout"
    }
