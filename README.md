# Medical AI Crash-Test Simulator

A reproducible, non-clinical ML evaluation suite. It evaluates one fixed `StandardScaler + LogisticRegression` baseline on scikit-learn’s educational breast-cancer dataset under three explicit tabular perturbations.

## What runs

1. A stratified 60/20/20 train/validation/test split, seeded and recorded.
2. A baseline model whose scaler is fit only on training data.
3. Gaussian noise, attenuation toward training medians, and seeded feature dropout at fixed severities.
4. Test metrics, prediction stability, confidence reliability, and a confidence-abstention policy selected only on validation data.
5. JSON, CSV, and Markdown outputs including software and configuration manifest data.

## Run

```powershell
python -m pytest -q
python cli.py --output-dir reports/demo
streamlit run app.py
```

The `configs/demo.json` file documents the default experimental contract. The CLI records its actual values in `results.json`; it does not silently read a config whose contents it cannot validate.

## Interpretation boundaries

The dataset and stressors are educational. Stability under the selected synthetic tabular transformations is not proof of correctness, calibration, real-world robustness, clinical safety, fairness, or diagnostic usefulness. Thresholds are selected on validation data, never by optimizing test results.
