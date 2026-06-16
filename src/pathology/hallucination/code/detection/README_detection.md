# Pathology Detection

This directory contains the NeuroCogMap pathology-detection utilities:

- an interpretable detector based on parcel and capability features
- statistical tests for comparing generated detection-result files
- batch scripts for running cross-validated detection experiments

## Directory Layout

```text
detection/
├── feature_extractor.py          # Feature extraction utilities
├── config_builder.py             # Automatic indicator-configuration builder
├── train_cv.py                   # Cross-validation training script
├── significance_test.py          # Statistical comparison utilities
├── run_detection_overall.sh      # Batch detection runner
└── README_detection.md           # This document
```

## Quick Start

### 1. Train the Detector

```bash
# Check the local environment and required inputs.
bash run_detection_overall.sh check

# Train detectors for all datasets listed in MODEL_DATA_LIST.
bash run_detection_overall.sh
```

### 2. Run Significance Tests

```bash
python significance_test.py \
  --method1_results /path/to/method1/cv_metrics.json \
  --method2_results /path/to/method2/cv_metrics.json \
  --method1_name NeuroCogMap \
  --method2_name Comparator \
  --output_dir /tmp/neurocogmap_release_outputs/pathology/significance_test
```

## Detector Features

The detector builds a 12-dimensional feature vector from the analysis outputs:

- **Indicator axes**: `M_plus`, `G_minus`, `Cpos_cap`, `Cneg_cap`
- **Connectivity mismatch features**: `C_plus`, `C_minus`, `C_plus_cap`, `C_minus_cap`
- **Prototype-similarity differences**: `s_da`, `s_dc`, `s_dF`, `s_dG`
- **Decision threshold**: selected on the training fold by maximizing F1

## Statistical Tests

The significance-test utility supports:

- **DeLong test** for AUROC comparison
- **Wilcoxon test** for F1 comparison
- **McNemar test** for accuracy comparison
- **Bootstrap confidence intervals**

## Outputs

### Detector Outputs

- `cv_metrics.json`: cross-validation metrics, including accuracy, precision, recall, F1, AUROC, AUPRC, and threshold
- `hallucination_detector.joblib`: trained detector with scaler, classifier, threshold, prototypes, indicator configuration, and feature names

### Significance-Test Outputs

- `significance_test_results.json`: full statistical-test results
- `fold_metrics.csv`: per-fold metric comparison
- `significance_report.txt`: human-readable summary

## Configuration

### Dataset List

Edit `MODEL_DATA_LIST` in the shell script to select the datasets to process:

```bash
MODEL_DATA_LIST=(
    "truthfulqa_gemma-2-2b"
    "MedHallu_gemma-2-2b"
    "HaluEval_gemma-2-2b"
    # Add more datasets here.
)
```

### Model and Training Parameters

- `FOLDS`: number of cross-validation folds (default: `5`)
- `RANDOM_STATE`: random seed (default: `42`)
- `MODEL_TYPE`: detector model, one of `lr`, `ridge`, `svm`, or `rf` (default: `lr`)
- `LR_PENALTY`: LogisticRegression penalty (default: `l2`)
- `LR_SOLVER`: LogisticRegression solver (default: `lbfgs`)
- `LR_C`: inverse regularization strength for LogisticRegression (default: `1.0`)
- `CLASS_WEIGHT`: class weighting, one of `balanced` or `none` (default: `balanced`)
- `TUNE_HYPERPARAMS`: whether to run the internal LogisticRegression hyperparameter search (default: `false`)
- `SKIP_EXISTING`: whether to skip datasets with existing output files (default: `false`)

By default, the detector uses a fixed L2 LogisticRegression and no longer selects the best model by mean AUROC across several model families. To manually switch models, set `MODEL_TYPE` before running the script:

```bash
MODEL_TYPE=ridge bash run_detection_overall.sh
MODEL_TYPE=svm bash run_detection_overall.sh
MODEL_TYPE=rf bash run_detection_overall.sh
```

## Dependencies

```bash
pip install scikit-learn joblib numpy pandas scipy matplotlib
```

## Notes

1. Input JSONL files must contain the `token_parcel_acts` field.
2. The analysis outputs should be generated before training the detector, because they are used to construct the indicator configuration.
3. Scripts report errors explicitly; they are not designed to silently ignore failures.
4. Use `--skip_existing` when rerunning a batch and you want to avoid recomputing completed datasets.

## Troubleshooting

Common issues:

1. **ModuleNotFoundError**: run the Python scripts through the expected absolute paths used in the shell wrapper.
2. **FileNotFoundError**: check that all input files and mapping files exist.
3. **ValueError**: verify the JSONL input format, especially the shape of `token_parcel_acts`.

Debugging checklist:

1. Run `bash run_detection_overall.sh check` first.
2. Confirm that the input files exist and are readable.
3. Inspect the printed error message and the failing dataset path.
