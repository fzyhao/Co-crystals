# Co-crystal Prediction Pipeline

This repository contains a Python pipeline for co-crystal prediction, including data preprocessing, feature selection, GAN-based data augmentation, classification, and regression.

## Structure

```
code/
├── config.py              # Centralized paths and parameters
├── data_utils.py          # Shared data loading and standardization utilities
├── evaluation.py          # Shared cross-validation and evaluation functions
├── preprocessing.py       # Remove zero-variance features
├── feature_selection.py   # Four feature selection methods
├── kendall.py             # Kendall tau weighted voting for feature aggregation
├── GAN.py                 # GAN-based data augmentation
├── classify.py            # Binary classification + extended analysis utilities
├── regression.py          # Regression (donor/acceptor ratio)
├── DNN.py                 # Deep neural network regression
├── requirements.txt       # Python dependencies
└── .gitignore             # Git ignore rules
```

## Usage

1. Place your datasets under `data/` using the generic filenames expected by `config.py`:
   - `data/processed_data_reg.xlsx`
   - `data/external_validation_reg.xlsx`
   - `data/external_validation_cls.xlsx`

2. Run the pipeline step by step:
   ```bash
   python preprocessing.py
   python feature_selection.py
   # Manually combine the four feature selection results into
   # results/feature_selection/regression/minmax_normalized.xlsx
   python kendall.py
   python regression.py
   python classify.py
   python DNN.py      # uses GAN-augmented data
   python GAN.py      # optional data augmentation
   ```

## Notes

- All hardcoded absolute paths and specific dataset filenames have been removed. Paths are now configured in `config.py`.
- The previous "split external validation data into supplement and test parts" logic has been removed. External validation sets are kept intact and evaluated as a whole.
- Each script is protected by `if __name__ == "__main__":` so modules can be imported safely.
- `classify.py` keeps the original commented analysis code as optional functions. Set `run_extended_analysis=True` in `run_classification_pipeline()` to enable individual classifier evaluation, multi-model ROC, confusion matrix, permutation importance, 3D visualization, and SHAP analysis.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
