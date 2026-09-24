from __future__ import annotations

from pathlib import Path
import argparse

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def build_model(model_name: str):
    """Return a starter model configuration."""
    model_name = model_name.lower()

    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )

    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )

    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def chronological_split(
    df: pd.DataFrame,
    date_col: str,
    test_fraction: float = 0.2,
):
    """Split chronologically instead of randomly to better mimic future prediction."""
    ordered = df.sort_values(date_col).reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_fraction))
    return ordered.iloc[:split_index], ordered.iloc[split_index:]


def train_and_log(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    model_name: str,
):
    train_df, test_df = chronological_split(df, date_col=date_col)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = build_model(model_name)

    mlflow.set_experiment("nfl-game-prediction")

    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "brier_score": brier_score_loss(y_test, probabilities),
            "log_loss": log_loss(y_test, probabilities),
        }

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="NFLGameWinPredictor",
        )

        Path("models").mkdir(exist_ok=True)
        model_path = Path("models") / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        print(f"Finished run: {model_name}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/modeling_data.csv")
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "xgboost", "lightgbm"],
        default="logistic_regression",
    )
    parser.add_argument("--target", default="home_win")
    parser.add_argument("--date-col", default="game_date")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    excluded = {args.target, args.date_col}
    feature_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in excluded
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns found for training.")

    train_and_log(
        df=df,
        feature_cols=feature_cols,
        target_col=args.target,
        date_col=args.date_col,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
