"""Deep neural network regression for co-crystal ratio prediction."""
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import config
import data_utils


class DNNRegressor(nn.Module):
    """4-layer MLP regressor."""

    def __init__(self, input_dim: int):
        super(DNNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y.view(-1, 1))
                loss.backward()
                optimizer.step()

    def predict(self, X: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            return outputs.numpy().flatten()


def model_predict(model: nn.Module, input_data: np.ndarray) -> np.ndarray:
    """Wrapper for SHAP-compatible predictions."""
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    predictions = model(input_tensor)
    return predictions.detach().numpy()


def format_prediction(y_pred):
    """Format continuous predictions as discrete integer ranges with probabilities."""
    if isinstance(y_pred, (np.ndarray, list)):
        formatted = []
        integers = []
        for pred in y_pred:
            f, i = format_prediction(pred)
            formatted.append(f)
            integers.append(i)
        return formatted, integers

    pred = float(y_pred)
    lower = int(np.floor(pred))
    upper = int(np.ceil(pred))

    if lower <= 0:
        return f"{upper} (100%)", [upper]

    upper_prob = (pred - lower) * 100
    lower_prob = 100 - upper_prob
    upper_prob = int(round(upper_prob))
    lower_prob = int(round(lower_prob))

    if upper_prob == 0:
        return f"{lower} (100%)", [lower]
    elif lower_prob == 0:
        return f"{upper} (100%)", [upper]
    else:
        return f"{lower} ({lower_prob}%) 和 {upper} ({upper_prob}%)", [lower, upper]


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the percentage of true values that fall in the predicted integer range."""
    _, pred_integers_list = format_prediction(y_pred)
    correct = 0
    for true, pred_integers in zip(y_true, pred_integers_list):
        if int(true) in pred_integers:
            correct += 1
    accuracy = (correct / len(y_true)) * 100
    return accuracy


def _cross_validation_dnn(
    model_class: Callable[[], nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = config.RANDOM_STATE,
    n_splits: int = config.CV_N_SPLITS,
):
    """K-Fold CV for PyTorch DNN regressors instantiated via a factory."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmse_scores = []
    rmse_vals = []

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        model = model_class()
        model.fit(X_train_tensor, y_train_tensor)

        y_pred = model.predict(X_test_tensor)
        rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse_score)

        y_val_pred = model.predict(X_val_tensor)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
        rmse_vals.append(rmse_val)

    return rmse_scores, rmse_vals


def run_dnn_pipeline(run_cv: bool = False, n_splits: int = config.CV_N_SPLITS) -> None:
    """Load augmented data, train a DNN regressor, and evaluate on the full external validation set."""
    feature_names = data_utils.load_feature_importance(task="regression")
    X, Y = data_utils.load_data(feature_names, task="regression", use_generated=True)
    X_val, Y_val = data_utils.load_external_validation(feature_names, task="regression")

    X, X_val, _ = data_utils.standardize_data(X, X_val)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

    input_dim = X_tensor.shape[1]
    model_class = lambda: DNNRegressor(input_dim)

    if run_cv:
        rmse_scores, rmse_vals = _cross_validation_dnn(
            model_class, X_tensor.numpy(), Y_tensor.numpy(), X_val, Y_val, n_splits=n_splits
        )
        print("内部验证RMSE：")
        for score in rmse_scores:
            print(score)
        print("测试集RMSE：")
        for val in rmse_vals:
            print(val)

    # Train final model on the full augmented training set
    model_all = model_class()
    model_all.fit(X_tensor, Y_tensor)

    y_pred = model_all.predict(X_val_tensor)
    print("真实值：")
    print(Y_val)
    print("预测值")
    print(y_pred)
    print("全部测试集预测：")
    rmse_all = np.sqrt(np.mean((Y_val - y_pred) ** 2))
    print(rmse_all)

    results = pd.DataFrame({"真实值": Y_val, "预测值": y_pred})
    config.PREDICTION_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    results.to_excel(config.PREDICTION_RESULTS, index=False)
    print(f"预测结果已保存到: {config.PREDICTION_RESULTS}")

    formatted_pred, _ = format_prediction(y_pred)
    print("格式化后的预测值：")
    for i, (true, pred) in enumerate(zip(Y_val, formatted_pred)):
        print(f"样本{i}: 真实值={int(true)}, 预测值={pred}")

    accuracy = calculate_accuracy(Y_val, y_pred)
    print(f"\n预测正确率: {accuracy:.2f}%")


if __name__ == "__main__":
    run_dnn_pipeline(run_cv=False)
