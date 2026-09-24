from __future__ import annotations

import pandas as pd


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> None:
    """Run basic data-quality checks and raise ValueError on failure."""

    if df.empty:
        raise ValueError("Validation failed: dataframe is empty.")

    if required_columns:
        missing = sorted(set(required_columns) - set(df.columns))
        if missing:
            raise ValueError(
                f"Validation failed: missing required columns: {missing}"
            )

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"Validation failed: duplicate column names found: {duplicates}"
        )

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        print(f"Warning: found {duplicate_rows} duplicate rows.")

    null_pct = df.isna().mean().sort_values(ascending=False)
    high_null = null_pct[null_pct > 0.50]
    if not high_null.empty:
        print("Warning: columns with >50% missing values:")
        print(high_null)

    print(
        f"Validation passed basic checks for {len(df):,} rows "
        f"and {len(df.columns):,} columns."
    )


if __name__ == "__main__":
    from ingest import load_csv

    df = load_csv("data/raw/nfl_data.csv")
    validate_dataframe(df)
