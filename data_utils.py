"""Shared data loading and preprocessing utilities."""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


def load_feature_importance(task: str = "regression", top_percent: float = config.FEATURE_TOP_PERCENT) -> list:
    """Load aggregated feature importance and return the top feature names.

    Parameters
    ----------
    task : str
        Either 'regression' or 'classification'.
    top_percent : float
        Percentage of top features to keep (0 < top_percent <= 1).

    Returns
    -------
    list
        Names of selected features.
    """
    if task == "regression":
        path = config.FEATURE_IMPORTANCE_REG_WEIGHTED
    elif task == "classification":
        path = config.FEATURE_IMPORTANCE_CLS
    else:
        raise ValueError(f"Unknown task: {task}. Choose 'regression' or 'classification'.")

    df = pd.read_excel(path)
    importance_column = df["importance"]
    threshold_index = int(len(importance_column) * top_percent)
    threshold_index = max(1, threshold_index)
    top_rows = importance_column.nlargest(threshold_index).index
    feature_names = df.loc[top_rows, "Feature"].tolist()
    return feature_names


def load_data(
    feature_names: list,
    task: str = "regression",
    use_generated: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data for the given task.

    Parameters
    ----------
    feature_names : list
        Selected feature names.
    task : str
        Either 'regression' or 'classification'.
    use_generated : bool
        If True, load GAN-augmented data (regression only).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Feature matrix X and target vector Y.
    """
    if use_generated:
        path = config.GENERATED_DATA
        target_col = "Regression"
    elif task == "regression":
        path = config.PROCESSED_REGRESSION_DATA
        target_col = "Regression"
    elif task == "classification":
        path = config.PROCESSED_CLASSIFICATION_DATA
        target_col = "Classify"
    else:
        raise ValueError(f"Unknown task: {task}. Choose 'regression' or 'classification'.")

    df = pd.read_excel(path)
    X = np.array(df[feature_names])
    Y = np.array(df[target_col])
    return X, Y


def load_external_validation(
    feature_names: list,
    task: str = "regression",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the full external validation set without any splitting.

    Parameters
    ----------
    feature_names : list
        Selected feature names.
    task : str
        Either 'regression' or 'classification'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        External validation features and targets.
    """
    if task == "regression":
        path = config.EXTERNAL_VALIDATION_REG
        target_col = "Regression"
    elif task == "classification":
        path = config.EXTERNAL_VALIDATION_CLS
        target_col = "Classify"
    else:
        raise ValueError(f"Unknown task: {task}. Choose 'regression' or 'classification'.")

    df_val = pd.read_excel(path)
    X_val = np.array(df_val[feature_names])
    Y_val = np.array(df_val[target_col])
    return X_val, Y_val


def standardize_data(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """Fit a StandardScaler on X_train and transform both sets.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    X_val : np.ndarray, optional
        External validation features.
    scaler : StandardScaler, optional
        Existing scaler to reuse. If None, a new scaler is fitted.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]
        Scaled training features, scaled validation features (if provided),
        and the fitted scaler.
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    return X_train_scaled, X_val_scaled, scaler
