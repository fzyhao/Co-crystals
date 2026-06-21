"""Compute feature importance using four selection methods."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import config


def compute_spearman(X: pd.DataFrame, Y: pd.Series) -> pd.DataFrame:
    """Compute Spearman correlation between each feature and the target."""
    spearman_corr = X.corrwith(Y, method="spearman")
    result = pd.DataFrame({
        "Column_Name": spearman_corr.index,
        "Spearman_Correlation": spearman_corr.values,
    })
    return result


def distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute distance correlation between a feature and the target."""
    A = np.array([[np.linalg.norm(x_i - x_j) for x_j in X] for x_i in X])
    B = np.array([[np.linalg.norm(y_i - y_j) for y_j in Y] for y_i in Y])
    n = X.shape[0]
    H = np.eye(n) - 1 / n * np.ones((n, n))
    dcov2_XY = 1 / n**2 * np.trace(H @ A @ H @ B)
    dvar_X = 1 / n * (np.trace(H @ A @ H @ A))
    dvar_Y = 1 / n * (np.trace(H @ B @ H @ B))
    dcor = np.sqrt(dcov2_XY / np.sqrt(dvar_X * dvar_Y))
    return dcor


def compute_distance_correlation(X: pd.DataFrame, Y: pd.Series) -> pd.DataFrame:
    """Compute distance correlation for all features and return a DataFrame."""
    result = {}
    for col in X.columns:
        feature = X[col].to_numpy()
        dcor = distance_correlation(feature, Y)
        result[col] = dcor
    df_result = pd.DataFrame({
        "Feature": list(result.keys()),
        "Distance Correlation with Y": list(result.values()),
    })
    return df_result


def compute_rf_importance(
    X: pd.DataFrame,
    Y: pd.Series,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Train a Random Forest regressor and return feature importances."""
    X_train, _, y_train, _ = train_test_split(
        X, Y, test_size=0.2, random_state=random_state)
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    feature_importance = rf.feature_importances_
    df = pd.DataFrame({
        "Feature": X.columns,
        "feature_importance": feature_importance,
    })
    return df


def compute_lasso_coefficients(
    X: pd.DataFrame,
    Y: pd.Series,
    alpha: float = 0.01,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Train a Lasso regressor and return coefficients."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random_state)
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("测试集上的均方误差：", mse)
    coefficients = lasso.coef_
    df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": coefficients,
    })
    return df


def run_feature_selection(
    data_path: Path,
    output_dir: Path,
    target_col: str,
    random_state: int = config.RANDOM_STATE,
) -> None:
    """Run all four feature selection methods and save results.

    Parameters
    ----------
    data_path : Path
        Path to the preprocessed Excel file.
    output_dir : Path
        Directory where the four result files will be saved.
    target_col : str
        Name of the target column.
    random_state : int
        Random seed for reproducibility.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(data_path)
    X = df.iloc[:, 7:]
    Y = df[target_col]

    result_spearman = compute_spearman(X, Y)
    result_spearman.to_excel(output_dir / "spearman_results.xlsx", index=False)

    result_dcor = compute_distance_correlation(X, Y)
    result_dcor.to_excel(output_dir / "distance_correlation_results.xlsx", index=False)

    result_rf = compute_rf_importance(X, Y, random_state=random_state)
    result_rf.to_excel(output_dir / "rf_results.xlsx", index=False)

    result_lasso = compute_lasso_coefficients(X, Y, random_state=random_state)
    result_lasso.to_excel(output_dir / "lasso_results.xlsx", index=False)

    print(f"Feature selection results saved to: {output_dir}")


if __name__ == "__main__":
    run_feature_selection(
        config.PROCESSED_REGRESSION_DATA,
        config.FEATURE_SELECTION_REG_DIR,
        target_col="Regression",
    )
