"""
modelling.py (versi MLflow Project untuk Workflow CI)

Sama seperti modelling_tuning.py pada Kriteria 2, tapi dibuat menerima
parameter lewat command line (sesuai spesifikasi entry point pada file
`MLProject`) agar bisa dipanggil otomatis oleh `mlflow run` di GitHub Actions.

Dipanggil oleh MLProject dengan:
    mlflow run . -P data_dir=mushroom_preprocessing -P cv_folds=5
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

TARGET_COL = "poisonous_label"
EXPERIMENT_NAME = "Mushroom_Classification_SVM_CI"

PARAM_GRID = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", "auto"],
}


def load_data(data_dir: str):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    return X_train, X_test, y_train, y_test


def log_confusion_matrix_artifact(y_test, y_pred, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Edible (0)", "Poisonous (1)"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - SVC (CI run)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(data_dir: str, cv_folds: int):
    running_inside_mlflow_run = "MLFLOW_RUN_ID" in os.environ

    if not running_inside_mlflow_run:
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data(data_dir)

    run_kwargs = {} if running_inside_mlflow_run else {"run_name": "svc_gridsearch_ci"}
    with mlflow.start_run(**run_kwargs):
        base_model = SVC(probability=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=PARAM_GRID,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("scoring", "accuracy")

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        loss = log_loss(y_test, y_proba)

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_roc_auc", auc)
        mlflow.log_metric("test_log_loss", loss)
        mlflow.log_metric("cv_best_accuracy", grid_search.best_score_)

        print(f"Best params : {best_params}")
        print(f"Test accuracy : {accuracy:.4f}")

        os.makedirs("tmp_artifacts", exist_ok=True)
        cm_path = "tmp_artifacts/confusion_matrix.png"
        log_confusion_matrix_artifact(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)

        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = "tmp_artifacts/classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)

        mlflow.sklearn.log_model(best_model, artifact_path="model")

    print("CI training run selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="mushroom_preprocessing")
    parser.add_argument("--cv_folds", type=int, default=5)
    args = parser.parse_args()

    main(args.data_dir, args.cv_folds)
