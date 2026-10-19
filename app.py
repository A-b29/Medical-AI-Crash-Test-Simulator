"""Interactive report viewer. The experiment is rerun only when parameters change."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.runner import ExperimentConfig, run_experiment


st.set_page_config(page_title="Medical AI Crash-Test", page_icon="🧪", layout="wide")
st.title("🧪 Medical AI Crash-Test Simulator")
st.caption("Educational ML evaluation workflow — not a clinical validation or safety certificate.")

seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
risk_budget = st.sidebar.slider("Validation risk budget", 0.01, 0.20, 0.05, 0.01)

@st.cache_data(show_spinner="Running deterministic evaluation suite…")
def load_result(selected_seed: int, selected_budget: float) -> dict[str, object]:
    return run_experiment(ExperimentConfig(seed=selected_seed, risk_budget=selected_budget))


result = load_result(int(seed), float(risk_budget))
baseline = result["baseline_metrics"]
first, second, third, fourth = st.columns(4)
first.metric("Accuracy", f"{baseline['accuracy']:.1%}")
second.metric("F1", f"{baseline['f1']:.3f}")
third.metric("AUROC", f"{baseline['auroc']:.3f}")
fourth.metric("ECE", f"{result['calibration']['expected_calibration_error']:.3f}")

results = pd.DataFrame(result["perturbation_results"])
chart = px.line(
    results,
    x="severity",
    y="prediction_flip_rate",
    color="perturbation",
    markers=True,
    title="Prediction flip rate by controlled perturbation severity",
    labels={"prediction_flip_rate": "Flip rate", "severity": "Severity"},
)
st.plotly_chart(chart, use_container_width=True)
st.dataframe(results, use_container_width=True)

st.subheader("Validation-selected abstention policy")
policy = result["abstention"]
st.json({"risk_budget": policy["risk_budget"], "validation_policy": policy["validation_policy"], "test_evaluation": policy["test_evaluation"]})
st.caption("The confidence threshold is chosen from validation predictions; test metrics are shown only for evaluation.")
