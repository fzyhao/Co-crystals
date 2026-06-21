"""Preprocess raw co-crystal data by removing zero-variance features."""
from pathlib import Path

import pandas as pd

import config


def preprocess_data(input_path: Path, output_path: Path) -> None:
    """Read raw data, drop zero-variance feature columns, and save the result.

    Parameters
    ----------
    input_path : Path
        Path to the raw Excel file.
    output_path : Path
        Path where the preprocessed Excel file will be saved.
    """
    df = pd.read_excel(input_path)

    # Keep the first 7 columns as metadata
    original_columns = df.iloc[:, :7]
    # Feature columns start from the 8th column
    selected_columns = df.iloc[:, 7:]

    std_dev = selected_columns.std()
    zero_std_columns = std_dev[std_dev == 0]
    filtered_columns = selected_columns.drop(columns=zero_std_columns.index)

    print(f"剔除掉的列数个数之和: {len(zero_std_columns)}")
    print(f"剔除掉的列名: {list(zero_std_columns.index)}")

    final_df = pd.concat([original_columns, filtered_columns], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_excel(output_path, index=False)
    print(f"处理后的数据已保存到: {output_path}")


if __name__ == "__main__":
    preprocess_data(config.RAW_REGRESSION_DATA, config.PROCESSED_REGRESSION_DATA)
