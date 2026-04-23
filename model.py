"""Model training, evaluation, and plotting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ModelKind = Literal["random_forest", "gradient_boosting"]


@dataclass
class TrainResult:
    """Container for training artifacts and diagnostics."""

    model: Pipeline
    x_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    mse: float
    r2: float


def _build_model_pipeline(model_kind: ModelKind = "random_forest") -> Pipeline:
    """Build preprocessing + regressor pipeline."""
    # Categorical encoding for compound; numerical columns kept as-is.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "compound_ohe",
                OneHotEncoder(handle_unknown="ignore"),
                ["Compound"],
            ),
        ],
        remainder="passthrough",
    )

    if model_kind == "gradient_boosting":
        regressor = GradientBoostingRegressor(random_state=42)
    else:
        regressor = RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def train_and_evaluate(
    df: pd.DataFrame,
    model_kind: ModelKind = "random_forest",
    test_size: float = 0.25,
    target_year: int | None = None,
) -> TrainResult:
    """
    Train model and compute metrics.

    We use a non-shuffled split to preserve race-time ordering. This better mimics
    real forecasting where future laps should not leak into training.
    """
    feature_cols = [
        "Compound",
        "TyreLife",
        "TrackTemp",
        "RollingLapTime",
        "LapNumber",
        "YearOffset",
        "HistMedianByCompoundTyreLife",
        "HistMedianByCompound",
    ]
    target_col = "LapTimeSeconds"

    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe missing required modeling columns: {missing}")

    x = df[feature_cols].copy()
    y = df[target_col].copy()

    if len(x) < 6:
        raise ValueError("Not enough rows to perform a train/test split.")

    # Evaluate on the selected season while allowing older seasons to enrich training.
    if target_year is not None and "SeasonYear" in df.columns:
        eval_mask = pd.to_numeric(df["SeasonYear"], errors="coerce") == float(target_year)
        eval_df = df[eval_mask].sort_values("LapNumber")
        if len(eval_df) >= 4:
            requested_test_n = max(2, int(round(len(eval_df) * test_size)))
            max_test_n = len(eval_df) - 2
            test_n = min(requested_test_n, max_test_n)
            test_n = max(2, test_n)

            test_idx = eval_df.tail(test_n).index
            train_idx = x.index.difference(test_idx)

            x_train, y_train = x.loc[train_idx], y.loc[train_idx]
            x_test, y_test = x.loc[test_idx], y.loc[test_idx]
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=max(2, min(len(x) - 2, int(round(len(x) * test_size)))),
                shuffle=False,
            )
    else:
        # Keep temporal ordering and enforce a valid split for short stints.
        requested_test_n = max(2, int(round(len(x) * test_size)))
        max_test_n = len(x) - 2
        test_n = min(requested_test_n, max_test_n)
        test_n = max(2, test_n)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_n,
            shuffle=False,
        )

    model = _build_model_pipeline(model_kind=model_kind)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return TrainResult(
        model=model,
        x_test=x_test,
        y_test=y_test,
        y_pred=y_pred,
        mse=float(mse),
        r2=float(r2),
    )


def build_actual_vs_pred_plot(df: pd.DataFrame, result: TrainResult):
    """Create a line chart comparing actual vs predicted lap times."""
    # Align the test indices back to original lap numbers for plotting.
    test_idx = result.x_test.index
    lap_numbers = df.loc[test_idx, "LapNumber"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        lap_numbers,
        result.y_test.to_numpy(),
        marker="o",
        linewidth=2,
        label="Actual Lap Time",
    )
    ax.plot(
        lap_numbers,
        result.y_pred,
        marker="s",
        linewidth=2,
        linestyle="--",
        label="Predicted Lap Time",
    )

    ax.set_title("Tire Degradation Modeling: Actual vs Predicted Lap Times")
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (seconds)")
    ax.grid(alpha=0.25)
    ax.legend()

    return fig


def get_feature_importance_table(model: Pipeline) -> pd.DataFrame:
    """Return model feature importance in a user-readable table."""
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    if not hasattr(regressor, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance", "importance_pct"])

    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_
    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    total = imp_df["importance"].sum()
    if total > 0:
        imp_df["importance_pct"] = (imp_df["importance"] / total) * 100
    else:
        imp_df["importance_pct"] = 0.0
    return imp_df
