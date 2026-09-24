from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load the raw NFL dataset from a CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


if __name__ == "__main__":
    default_path = Path("data/raw/nfl_data.csv")
    dataframe = load_csv(default_path)
    print(f"Loaded {len(dataframe):,} rows from {default_path}")
    print(dataframe.head())
