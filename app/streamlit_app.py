from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from predict import load_model, predict_win_probability


st.set_page_config(
    page_title="NFL Game Predictor",
    page_icon="🏈",
)

st.title("🏈 NFL Game Predictor")
st.write(
    "MVP interface for generating an NFL game win probability "
    "from pre-game team features."
)

model_path = ROOT / "models" / "logistic_regression.joblib"

if not model_path.exists():
    st.info(
        "No trained model is available yet. Train a model first, then return "
        "to this page for inference."
    )
    st.stop()

model = load_model(model_path)

st.subheader("Matchup Features")
st.caption(
    "These starter fields will be replaced with the final engineered "
    "home-vs-away feature set once the dataset schema is finalized."
)

yard_diff = st.number_input("Rolling yard differential", value=0.0)
turnover_diff = st.number_input("Rolling turnover differential", value=0.0)
third_down_diff = st.number_input(
    "Rolling third-down efficiency differential", value=0.0
)

if st.button("Predict Game"):
    features = pd.DataFrame(
        [
            {
                "yard_diff": yard_diff,
                "turnover_diff": turnover_diff,
                "third_down_diff": third_down_diff,
            }
        ]
    )

    try:
        probability = predict_win_probability(model, features)
        st.metric("Home Team Win Probability", f"{probability:.1%}")
    except Exception as exc:
        st.error(
            "The current model's training feature schema does not yet match "
            f"the starter UI. Details: {exc}"
        )
