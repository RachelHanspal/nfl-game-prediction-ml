from __future__ import annotations

import pandas as pd


def add_win_label(
    df: pd.DataFrame,
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
) -> pd.DataFrame:
    """Create a binary home-win label from final scores."""
    result = df.copy()

    if home_score_col not in result or away_score_col not in result:
        raise KeyError(
            f"Expected score columns '{home_score_col}' and '{away_score_col}'."
        )

    result["home_win"] = (
        result[home_score_col] > result[away_score_col]
    ).astype(int)

    return result


def add_shifted_rolling_average(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int = 5,
    sort_cols: list[str] | None = None,
    output_col: str | None = None,
) -> pd.DataFrame:
    """
    Add a leakage-safe rolling average.

    The shift(1) ensures the current game's value is excluded from its own
    prediction feature.
    """
    result = df.copy()

    if sort_cols:
        result = result.sort_values(sort_cols).copy()

    output_col = output_col or f"{value_col}_rolling_{window}"

    result[output_col] = (
        result.groupby(group_col)[value_col]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )

    return result


def add_stat_differential(
    df: pd.DataFrame,
    home_col: str,
    away_col: str,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Create a home-minus-away feature differential."""
    result = df.copy()
    output_col = output_col or f"{home_col}_minus_{away_col}"
    result[output_col] = result[home_col] - result[away_col]
    return result
