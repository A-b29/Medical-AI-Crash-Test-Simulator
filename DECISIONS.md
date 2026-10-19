# Decisions — Medical AI Crash-Test Simulator

## One fixed, inspectable evaluation contract

The MVP uses a seeded 60/20/20 stratified split, a `StandardScaler + LogisticRegression` baseline, and three named perturbations. A small, explicit contract is easier to reproduce and interpret than a platform that mixes models, data, and protocols from the outset.

## Training-only reference statistics

The scaler, perturbation standard deviations, and median replacement values all come from training data. This avoids leaking held-out distribution information into preprocessing or stress-test design.

## Severity zero is an identity test

Every perturbation defines `severity=0` as exact identity and has a unit test for it. This anchors the results table: any change at zero severity is a pipeline bug rather than an ML finding.

## Validation-selected abstention

The operating threshold is selected for maximum validation coverage within a stated risk budget, then evaluated unchanged on test data. This blocks a common but misleading form of test-set tuning.

## JSON and CSV as primary outputs

The runner exports results before the UI is considered. A Streamlit page can visualise the same data, but reproducibility should not require clicking through an interface.
