from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd


def load_model(path: str | Path):
    """Load a locally saved model artifact."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def predict_win_probability(model, features: pd.DataFrame) -> float:
    """Return the positive-class win probability for one matchup."""
    if len(features) != 1:
        raise ValueError("Prediction input should contain exactly one matchup.")

    probability = float(model.predict_proba(features)[:, 1][0])
    return probability
