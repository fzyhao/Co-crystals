"""Centralized configuration for the co-crystal prediction pipeline."""
from pathlib import Path

# Base directory for the project (directory containing the code package)
BASE_DIR = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Input data paths
# -----------------------------------------------------------------------------

# Raw training data used by preprocessing.py
RAW_REGRESSION_DATA = BASE_DIR / "data" / "processed_data_reg.xlsx"

# Preprocessed training data
PROCESSED_REGRESSION_DATA = BASE_DIR / "data" / "processed_data_regression.xlsx"
PROCESSED_CLASSIFICATION_DATA = BASE_DIR / "data" / "processed_data.xlsx"

# Data augmented by GAN
GENERATED_DATA = BASE_DIR / "data" / "generated_data.xlsx"

# External validation sets (kept separate, never split/merged into training data)
EXTERNAL_VALIDATION_REG = BASE_DIR / "data" / "external_validation_reg.xlsx"
EXTERNAL_VALIDATION_CLS = BASE_DIR / "data" / "external_validation_cls.xlsx"

# -----------------------------------------------------------------------------
# Feature selection output paths
# -----------------------------------------------------------------------------

FEATURE_SELECTION_DIR = BASE_DIR / "results" / "feature_selection"
FEATURE_SELECTION_REG_DIR = FEATURE_SELECTION_DIR / "regression"
FEATURE_SELECTION_CLS_DIR = FEATURE_SELECTION_DIR / "classification"

SPEARMAN_RESULTS = FEATURE_SELECTION_REG_DIR / "spearman_results.xlsx"
DISTANCE_CORR_RESULTS = FEATURE_SELECTION_REG_DIR / "distance_correlation_results.xlsx"
RF_RESULTS = FEATURE_SELECTION_REG_DIR / "rf_results.xlsx"
LASSO_RESULTS = FEATURE_SELECTION_REG_DIR / "lasso_results.xlsx"

# Aggregated (minmax-normalized) feature importance scores
FEATURE_IMPORTANCE_REG = FEATURE_SELECTION_REG_DIR / "minmax_normalized.xlsx"
FEATURE_IMPORTANCE_CLS = FEATURE_SELECTION_CLS_DIR / "minmax_normalized.xlsx"

# Weighted feature importance produced by kendall.py
FEATURE_IMPORTANCE_REG_WEIGHTED = FEATURE_SELECTION_REG_DIR / "minmax_normalized_weighted.xlsx"

# -----------------------------------------------------------------------------
# Prediction output paths
# -----------------------------------------------------------------------------

PREDICTION_RESULTS = BASE_DIR / "results" / "prediction_results.xlsx"

# -----------------------------------------------------------------------------
# Tunable parameters
# -----------------------------------------------------------------------------

RANDOM_STATE = 42
CV_N_SPLITS = 5
FEATURE_TOP_PERCENT = 0.5
