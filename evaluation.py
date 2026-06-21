"""Shared cross-validation and evaluation utilities."""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, mean_squared_error, roc_auc_score, roc_curve
from sklearn.model_selection import KFold

import config


def cross_validation_regression(
    model,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = config.RANDOM_STATE,
    n_splits: int = config.CV_N_SPLITS,
) -> Tuple[List[float], List[float]]:
    """K-Fold cross-validation for sklearn regressors.

    Returns internal fold RMSE scores and external validation RMSE scores.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmse_scores = []
    rmse_vals = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse_score)

        y_val_pred = model.predict(X_val)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
        rmse_vals.append(rmse_val)

    return rmse_scores, rmse_vals


def cross_validation_classification(
    model,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = config.RANDOM_STATE,
    n_splits: int = config.CV_N_SPLITS,
) -> Tuple[List[float], List[float], float, List[float]]:
    """K-Fold cross-validation for sklearn classifiers.

    Returns accuracies, fold AUCs, overall AUC, and external validation accuracies.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracies = []
    aucs = []
    acc_vals = []
    all_y_true = []
    all_y_pred_proba = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        accuracies.append(accuracy)

        acc_val = model.score(X_val, y_val)
        acc_vals.append(acc_val)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        aucs.append(auc_score)

        all_y_true.extend(y_test)
        all_y_pred_proba.extend(y_pred_proba)

    fpr, tpr, _ = roc_curve(all_y_true, all_y_pred_proba)
    overall_auc = auc(fpr, tpr)

    return accuracies, aucs, overall_auc, acc_vals


def _check_tensor(tensor: torch.Tensor, name: str) -> None:
    """Print a warning if a tensor contains NaN or Inf values."""
    if torch.isnan(tensor).any():
        print(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        print(f"{name} contains Inf values")


def kan_cross_validation_regression(
    model,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = config.RANDOM_STATE,
    n_splits: int = config.CV_N_SPLITS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """K-Fold cross-validation for KAN regression models."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmse_scores = []
    rmse_vals = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float).unsqueeze(1)

        _check_tensor(X_train_tensor, "X_train_tensor")
        _check_tensor(y_train_tensor, "y_train_tensor")
        _check_tensor(X_test_tensor, "X_test_tensor")
        _check_tensor(y_test_tensor, "y_test_tensor")
        _check_tensor(X_val_tensor, "X_val_tensor")
        _check_tensor(y_val_tensor, "y_val_tensor")

        dataset = {
            "train_input": X_train_tensor,
            "train_label": y_train_tensor,
            "test_input": X_test_tensor,
            "test_label": y_test_tensor,
            "val_input": X_val_tensor,
            "val_label": y_val_tensor,
        }

        model.train(dataset, opt="LBFGS", steps=20)

        y_pred = model(dataset["test_input"])
        mse_test = nn.MSELoss()(y_pred, dataset["test_label"])
        rmse_scores.append(torch.sqrt(mse_test))

        y_val_pred = model(dataset["val_input"])
        mse_val = nn.MSELoss()(y_val_pred, dataset["val_label"])
        rmse_vals.append(torch.sqrt(mse_val))

    rmse_scores = torch.stack(rmse_scores)
    rmse_vals = torch.stack(rmse_vals)
    return rmse_scores, rmse_vals


def kan_cross_validation_classification(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 3,
    random_state: int = config.RANDOM_STATE,
) -> List[torch.Tensor]:
    """K-Fold cross-validation for KAN classification models."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        dataset = {
            "train_input": X_train_tensor,
            "test_input": X_test_tensor,
            "train_label": y_train_tensor,
            "test_label": y_test_tensor,
        }

        model.train(dataset, opt="LBFGS", steps=20)

        accuracy = torch.mean(
            (torch.argmax(model(dataset["test_input"]), dim=1) == dataset["test_label"]).float()
        )
        accuracies.append(accuracy)

    return accuracies
