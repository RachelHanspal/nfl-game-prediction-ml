# NFL Game Prediction ML

End-to-end machine learning MVP for predicting NFL game outcomes and win probabilities.

## MVP Goals

This project demonstrates:
- Data ingestion and validation
- Reproducible preprocessing and feature engineering
- Experiment tracking with MLflow
- Model comparison and versioning
- A simple Streamlit prediction interface

## Planned Models

- Logistic Regression
- XGBoost
- LightGBM

## Project Structure

```text
nfl-game-prediction-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── ingest.py
│   ├── validate.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── app/
│   └── streamlit_app.py
├── notebooks/
├── models/
└── diagrams/
```

## Important Modeling Rule

Features for a game must only use information available **before that game starts**. Rolling statistics should therefore be shifted so the current game's result and statistics are never included in its own prediction features.

## Quick Start

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place the raw CSV in `data/raw/`.
4. Run ingestion/validation and feature engineering.
5. Train models with MLflow tracking.
6. Launch the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```
