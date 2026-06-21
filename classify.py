"""Classification pipeline for predicting co-crystal formation."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from pathlib import Path
from sklearn import metrics, neighbors, svm, tree
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier

# Optional classifiers with heavier dependencies
try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

try:
    from kan import *
    _HAS_KAN = True
except ImportError:
    _HAS_KAN = False

try:
    from mpl_toolkits.mplot3d import Axes3D
    _HAS_MPL3D = True
except ImportError:
    _HAS_MPL3D = False

import config
import data_utils
import evaluation


matplotlib.rcParams["font.sans-serif"] = ["KaiTi", "SimHei", "FangSong"]
matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["axes.unicode_minus"] = False


def build_voting_classifier() -> VotingClassifier:
    """Build the soft-voting ensemble classifier."""
    estimators = [
        ("bp", MLPClassifier(
            solver="adam", activation="relu", max_iter=1000,
            alpha=1e-3, hidden_layer_sizes=(16, 32, 32), random_state=5)),
        ("rf", RandomForestClassifier(n_estimators=50, max_depth=20, random_state=0, min_samples_split=3)),
        ("lr", LogisticRegression(max_iter=1000, solver="liblinear", random_state=0)),
        ("svm", SVC(kernel="rbf", probability=True)),
        ("knn", KNeighborsClassifier()),
        ("xgb", XGBClassifier(learning_rate=0.1, max_depth=20, n_estimators=50)),
        ("gbdt", GradientBoostingClassifier()),
        ("ada", AdaBoostClassifier()),
    ]

    if _HAS_LIGHTGBM:
        estimators.append(("lgbm", lgb.LGBMClassifier()))
    if _HAS_CATBOOST:
        estimators.append(("catboost", CatBoostClassifier(verbose=0)))

    voting_clf = VotingClassifier(estimators=estimators, voting="soft")
    return voting_clf


def plot_estimator_probabilities(model: VotingClassifier, X: np.ndarray) -> None:
    """Plot the predicted probability distribution of each base estimator."""
    probs = [estimator.predict_proba(X) for estimator in model.estimators_]

    n_estimators = len(probs)
    n_cols = 3
    n_rows = (n_estimators + n_cols - 1) // n_cols

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    for i, prob in enumerate(probs):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(prob[:, 1], bins=20, kde=True)
        plt.title(f"Model {i + 1} Predicted Probabilities")
        plt.xlabel("Probability of Class 1")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def train_individual_classifiers(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train and evaluate individual classifiers on an internal train/test split."""
    models = {
        "BP": MLPClassifier(
            solver="adam", activation="relu", max_iter=1000,
            alpha=1e-3, hidden_layer_sizes=(16, 32, 32), random_state=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=20, random_state=0, min_samples_split=3),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=0),
        "Decision Tree": tree.DecisionTreeClassifier(max_depth=10),
        "SVM": svm.SVC(C=1.0, kernel="linear", probability=True, max_iter=-1),
        "K Nearest Neighbor": neighbors.KNeighborsClassifier(),
        "XGBoost": XGBClassifier(learning_rate=0.1, max_depth=10, n_estimators=10),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        print(f"{name} 预测准确率：")
        print(accuracy)
        print("查准率:", metrics.precision_score(y_test, y_pred))
        print("召回率:", metrics.recall_score(y_test, y_pred))
        print("F1_score:", metrics.f1_score(y_test, y_pred))
        results[name] = model
    return results


def plot_multi_models_roc(
    model_names: list,
    models: list,
    colors: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Path,
    dpin: int = 200,
) -> None:
    """Plot ROC curves for multiple models on a single figure."""
    plt.figure(figsize=(20, 20), dpi=dpin)

    for name, model, colorname in zip(model_names, models, colors):
        y_test_preds = model.predict(X_test)
        y_test_predprob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_test_predprob, pos_label=1)

        # Save prediction probabilities
        savedata = pd.DataFrame({"y_test": y_test, "y_test_predprob": y_test_predprob})
        save_path.parent.mkdir(parents=True, exist_ok=True)
        savedata.to_excel(save_path.parent / "predict.xlsx", index=False)

        plt.plot(fpr, tpr, lw=5, label="{} (AUC={:.3f})".format(name, roc_auc_score(y_test, y_test_predprob)), color=colorname)
        plt.plot([0, 1], [0, 1], "--", lw=5, color="grey")
        plt.axis("square")
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.yticks(fontproperties="Arial", size=20)
        plt.xticks(fontproperties="Arial", size=20)
        plt.xlabel("1 - specificity", fontsize=40)
        plt.ylabel("Sensitivity", fontsize=40)
        plt.title("ROC Curve", fontsize=40)
        plt.legend(loc="lower right", fontsize=30)

    plt.savefig(save_path)
    plt.clf()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """Plot and save a confusion matrix heatmap."""
    confusion_mat = confusion_matrix(y_true, y_pred)
    sns.set()
    figure, ax = plt.subplots()
    sns.heatmap(confusion_mat, cmap="YlGnBu_r", annot=True, ax=ax)
    ax.set_xlabel("predictive value")
    ax.set_ylabel("true value")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.clf()
    plt.close(figure)


def plot_permutation_importance(model, X: np.ndarray, y: np.ndarray, save_path: Path) -> None:
    """Compute and save permutation importance for a model."""
    result = permutation_importance(model, X, y, n_repeats=10, random_state=2)
    print(result.importances_mean)
    np.save(save_path, result.importances_mean)


def plot_3d_projection(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: tuple = (1, 6, 8),
    save_path: Path = None,
) -> None:
    """Plot a 3D scatter of selected features colored by label."""
    if not _HAS_MPL3D:
        print("mpl_toolkits.mplot3d not available; skipping 3D plot.")
        return

    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=20, azim=20)
    ax.scatter(X[:, feature_indices[0]], X[:, feature_indices[1]], X[:, feature_indices[2]], marker="o", c=y)
    plt.title("True Label Map")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.clf()


def compute_shap_summary(
    model,
    X_test: np.ndarray,
    feature_names: list,
    output_dir: Path,
    max_display: int = 20,
) -> None:
    """Compute and save SHAP summary plots and values."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X_test_summary = shap.kmeans(X_test, 10)
    explainer = shap.KernelExplainer(model.predict, data=X_test_summary)
    shap_values = explainer.shap_values(X_test)

    # Bar summary
    plt.figure()
    plt.rcParams["font.sans-serif"] = ["SimSun"]
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.unicode_minus"] = False
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", max_display=max_display, show=False)
    plt.grid(False)
    plt.savefig(output_dir / "global_importance.png", transparent=True)

    # Dot summary
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, max_display=max_display, show=False)
    plt.grid(False)
    plt.savefig(output_dir / "summary_scatter.png", transparent=True)

    # Save SHAP values
    shap_values_flat = shap_values.reshape(shap_values.shape[0], -1)
    shap_df = pd.DataFrame(shap_values_flat)
    x_test_df = pd.DataFrame(X_test)
    combined_df = pd.concat([shap_df, x_test_df], axis=1)
    combined_df.to_excel(output_dir / "shap_values.xlsx", index=False)


def compute_shap_force_plot(
    model,
    X_test: np.ndarray,
    feature_names: list,
    sample_index: int,
    save_path: Path,
) -> None:
    """Compute and save a SHAP force plot for a single sample."""
    explainer = shap.KernelExplainer(model.predict, data=X_test)
    shap_values = explainer.shap_values(X_test)

    single_shap_values = shap_values[sample_index, :]
    single_features = X_test[sample_index, :]

    # Select top 3 positive and top 3 negative contributors
    top_positive_indices = np.argsort(-single_shap_values)[:3]
    top_negative_indices = np.argsort(single_shap_values)[:3]
    selected_indices = np.concatenate([top_positive_indices, top_negative_indices])

    filtered_shap_values = single_shap_values[selected_indices]
    filtered_features = single_features[selected_indices]
    filtered_feature_names = [feature_names[i] for i in selected_indices]

    plt.figure(figsize=(100, 60))
    shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=filtered_shap_values,
        features=filtered_features,
        feature_names=filtered_feature_names,
        matplotlib=True,
        show=False,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")


def run_kan_classification(X: np.ndarray, y: np.ndarray, n_splits: int = 3) -> None:
    """Optional KAN classification cross-validation."""
    if not _HAS_KAN:
        print("KAN library not available; skipping KAN classification.")
        return

    clf_kan = KAN(width=[X.shape[1], 8, 4, 1], grid=5, k=3, seed=2)
    accuracies = evaluation.kan_cross_validation_classification(clf_kan, X, y, n_splits=n_splits)
    print("三折交叉验证的平均准确率：")
    print(np.mean(accuracies))


def run_classification_pipeline(
    run_cv: bool = False,
    run_extended_analysis: bool = False,
) -> None:
    """Load data, optionally run cross-validation, and evaluate on the full external validation set."""
    feature_names = data_utils.load_feature_importance(task="classification")
    X, Y = data_utils.load_data(feature_names, task="classification")
    X_val, Y_val = data_utils.load_external_validation(feature_names, task="classification")

    X, X_val, _ = data_utils.standardize_data(X, X_val)

    output_dir = config.BASE_DIR / "results" / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional KAN classification
    # run_kan_classification(X, Y)

    voting_clf = build_voting_classifier()

    if run_cv:
        accuracies, aucs, overall_auc, acc_vals = evaluation.cross_validation_classification(
            voting_clf, X, Y, X_val, Y_val
        )
        print("准确率：")
        for accuracy in accuracies:
            print(accuracy)
        print("AUC：")
        for auc_score in aucs:
            print(auc_score)
        print("整体AUC：")
        print(overall_auc)
        print("测试集准确率：")
        for acc_val in acc_vals:
            print(acc_val)

    # Final model trained on the full training set and evaluated on the full external validation set
    model_all = voting_clf.fit(X, Y)
    y_pred = model_all.predict(X_val)
    print("真实值：")
    print(Y_val)
    print("预测值")
    print(y_pred)
    print("全部测试集预测：")
    accuracy = model_all.score(X_val, Y_val)
    print(accuracy)

    plot_estimator_probabilities(model_all, X)

    if run_extended_analysis:
        # Internal train/test split for additional analyses
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=config.RANDOM_STATE
        )

        # Individual classifiers
        individual_models = train_individual_classifiers(X_train, X_test, y_train, y_test)

        # Permutation importance on BP
        bp_model = individual_models.get("BP")
        if bp_model is not None:
            plot_permutation_importance(bp_model, X_train, y_train, output_dir / "permutation_importance.npy")

        # 3D projection
        plot_3d_projection(X_train, y_train, save_path=output_dir / "3d_true_label.png")
        predict_train = bp_model.predict(X_train) if bp_model is not None else None
        if predict_train is not None:
            plot_3d_projection(X_train, predict_train, save_path=output_dir / "3d_prediction_label.png")

        # Multi-model ROC
        model_names = ["neural network", "Logistic Regression", "Random Forest", "Decision Tree", "K Nearest Neighbor", "XGBoost"]
        sampling_methods = [
            individual_models["BP"],
            individual_models["Logistic Regression"],
            individual_models["Random Forest"],
            individual_models["Decision Tree"],
            individual_models["K Nearest Neighbor"],
            individual_models["XGBoost"],
        ]
        colors = ["red", "orange", "pink", "mediumseagreen", "steelblue", "cyan"]
        plot_multi_models_roc(
            model_names, sampling_methods, colors, X_test, y_test,
            save_path=output_dir / "multi_models_roc.png"
        )

        # Confusion matrix on BP
        if bp_model is not None:
            y_pred_bp = bp_model.predict(X_test)
            plot_confusion_matrix(y_test, y_pred_bp, output_dir / "confusion_matrix.png")

        # SHAP analysis on BP
        if bp_model is not None:
            compute_shap_summary(bp_model, X_test, feature_names, output_dir)
            compute_shap_force_plot(bp_model, X_test, feature_names, sample_index=1, save_path=output_dir / "force_plot.png")


if __name__ == "__main__":
    run_classification_pipeline(run_cv=False, run_extended_analysis=False)
