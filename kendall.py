"""Aggregate feature selection results with Kendall tau weighted voting."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau

import config


def compute_kendall_weights(correlation_df: pd.DataFrame) -> pd.Series:
    """Compute normalized weights based on Kendall tau correlations."""
    weights = correlation_df.sum(axis=1) - 1  # subtract self-correlation
    weights_normalized = weights / weights.sum()
    return weights_normalized


def compute_weighted_importance(
    data: pd.DataFrame,
    columns: list,
    weights: pd.Series,
) -> np.ndarray:
    """Compute the weighted sum of normalized feature importance scores."""
    final_coefficients = np.zeros(data.shape[0])
    for i, col in enumerate(columns):
        final_coefficients += data[col].to_numpy() * weights.iloc[i]
    return final_coefficients


def plot_kendall_heatmap(correlation_df: pd.DataFrame) -> None:
    """Plot and display a heatmap of the Kendall tau correlation matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_df,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
    )
    plt.title("Kendall Tau Correlation Matrix")
    plt.show()


def run_kendall_aggregation(
    input_path: Path,
    output_path: Path,
    columns: list = None,
) -> None:
    """Read feature selection scores, aggregate with Kendall weights, and save.

    Parameters
    ----------
    input_path : Path
        Path to the Excel file containing the four normalized feature scores.
    output_path : Path
        Path where the aggregated result (with 'importance' column) will be saved.
    columns : list, optional
        Names of the score columns. Defaults to the four method names.
    """
    if columns is None:
        columns = ["Spearman", "dCor", "RF", "Lasso"]

    df = pd.read_excel(input_path)
    data = df[columns]

    # Compute Kendall tau correlation matrix
    correlation_matrix = np.zeros((len(columns), len(columns)))
    for i in range(len(columns)):
        for j in range(len(columns)):
            tau, _ = kendalltau(data[columns[i]], data[columns[j]])
            correlation_matrix[i, j] = tau

    correlation_df = pd.DataFrame(correlation_matrix, columns=columns, index=columns)
    plot_kendall_heatmap(correlation_df)

    # Compute weighted importance
    weights_normalized = compute_kendall_weights(correlation_df)
    final_coefficients = compute_weighted_importance(data, columns, weights_normalized)

    df["importance"] = final_coefficients
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"Weighted feature importance saved to: {output_path}")


if __name__ == "__main__":
    run_kendall_aggregation(
        config.FEATURE_IMPORTANCE_REG,
        config.FEATURE_IMPORTANCE_REG_WEIGHTED,
    )
