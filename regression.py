"""Regression pipeline for predicting co-crystal donor/acceptor ratios."""
import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor

import config
import data_utils
import evaluation
from kan import *


def build_models(input_dim: int = None):
    """Build the set of regression models used in the pipeline.

    Parameters
    ----------
    input_dim : int, optional
        Number of input features; required for the KAN model.

    Returns
    -------
    dict
        Mapping from model name to estimator instance.
    """
    models = {
        "mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32, 16, 1),
            activation="relu",
            solver="adam",
            random_state=42,
            learning_rate="adaptive",
        ),
        "xgb": XGBRegressor(learning_rate=0.1, max_depth=9, n_estimators=22),
        "linear": LinearRegression(),
    }

    if input_dim is not None:
        models["kan"] = KAN(width=[input_dim, 4, 2, 1], grid=5, k=4, seed=1)

    return models


def run_regression_pipeline(model_name: str = "xgb", run_cv: bool = False) -> None:
    """Load data, optionally run cross-validation, and evaluate on the full external validation set.

    Parameters
    ----------
    model_name : str
        Model to use for the final evaluation: 'mlp', 'xgb', 'linear', or 'kan'.
    run_cv : bool
        Whether to run K-Fold cross-validation before final evaluation.
    """
    feature_names = data_utils.load_feature_importance(task="regression")
    X, Y = data_utils.load_data(feature_names, task="regression")
    X_val, Y_val = data_utils.load_external_validation(feature_names, task="regression")

    X, X_val, _ = data_utils.standardize_data(X, X_val)

    models = build_models(input_dim=X.shape[1])
    if model_name not in models:
        raise ValueError(f"Unknown model_name: {model_name}. Choose from {list(models.keys())}")
    model = models[model_name]

    if run_cv:
        if model_name == "kan":
            rmse_scores, rmse_vals = evaluation.kan_cross_validation_regression(
                model, X, Y, X_val, Y_val
            )
            rmse_scores = rmse_scores.detach().numpy()
            rmse_vals = rmse_vals.detach().numpy()
        else:
            rmse_scores, rmse_vals = evaluation.cross_validation_regression(
                model, X, Y, X_val, Y_val
            )
        print("内部验证RMSE：")
        for score in rmse_scores:
            print(score)
        print("测试集RMSE：")
        for val in rmse_vals:
            print(val)

    # Final model trained on the full training set and evaluated on the full external validation set
    if model_name == "kan":
        X_tensor = torch.tensor(X, dtype=torch.float)
        Y_tensor = torch.tensor(Y, dtype=torch.float).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float)
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.float).unsqueeze(1)
        dataset = {
            "train_input": X_tensor,
            "train_label": Y_tensor,
            "test_input": X_val_tensor,
            "test_label": Y_val_tensor,
        }
        model.train(dataset, opt="LBFGS", steps=20)
        y_pred = model(X_val_tensor).detach().numpy().flatten()
    else:
        model_all = model.fit(X, Y)
        y_pred = model_all.predict(X_val)

    print("真实值：")
    print(Y_val)
    print("预测值")
    print(y_pred)
    print("全部测试集预测：")
    rmse_all = np.sqrt(mean_squared_error(Y_val, y_pred))
    print(rmse_all)

    results = pd.DataFrame({"真实值": Y_val, "预测值": y_pred})
    config.PREDICTION_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results.to_excel(config.PREDICTION_RESULTS, index=False)
    print(f"预测结果已保存到: {config.PREDICTION_RESULTS}")


if __name__ == "__main__":
    run_regression_pipeline(model_name="xgb", run_cv=False)
