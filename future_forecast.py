"""Future race forecasting utilities for 2026 all-driver predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data_pipeline import (
    MIN_REQUIRED_LAPS,
    add_historical_features,
    add_rolling_features,
    get_events_split_by_date,
    load_all_drivers_laps,
    preprocess_laps,
)
from model import train_and_evaluate


@dataclass
class DriverForecast:
    driver: str
    predicted_avg_lap: float
    predicted_deg_per_lap: float
    ai_ideal_pit_lap: int
    confidence_note: str


def _build_synthetic_future_stint(
    driver_df: pd.DataFrame,
    round_number: int,
    target_year: int,
    stint_laps: int = 20,
) -> pd.DataFrame:
    """Create a synthetic first-stint feature frame for future-race inference."""
    work = driver_df.copy().sort_values("LapNumber")

    # Estimate likely setup from history.
    default_compound = (
        work["Compound"].mode().iloc[0]
        if "Compound" in work.columns and not work["Compound"].dropna().empty
        else "MEDIUM"
    )
    baseline_lap = float(work["LapTimeSeconds"].median()) if "LapTimeSeconds" in work.columns else 95.0
    track_temp = float(work["TrackTemp"].median()) if "TrackTemp" in work.columns else 32.0

    future = pd.DataFrame(
        {
            "Compound": [default_compound] * stint_laps,
            "TyreLife": np.arange(1, stint_laps + 1, dtype=float),
            "TrackTemp": [track_temp] * stint_laps,
            "LapNumber": np.arange(1, stint_laps + 1, dtype=float),
            "RollingLapTime": [baseline_lap] * stint_laps,
            "SeasonYear": [target_year] * stint_laps,
            "RoundNumber": [float(round_number)] * stint_laps,
            "EventName": ["FUTURE_EVENT"] * stint_laps,
            "LapTimeSeconds": [baseline_lap] * stint_laps,
        }
    )

    # Attach historical prior features using the existing helper.
    combo = pd.concat([work, future], ignore_index=True, sort=False)
    combo = add_historical_features(combo, target_year=target_year)
    synth = combo.tail(stint_laps).copy().reset_index(drop=True)

    return synth


def _predict_stint_with_model(model, synth_df: pd.DataFrame) -> pd.DataFrame:
    """Predict lap-times for synthetic stint with iterative rolling update."""
    out = synth_df.copy()
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

    preds = []
    for i in range(len(out)):
        row = out.loc[[i], feature_cols].copy()
        pred = float(model.predict(row)[0])
        preds.append(pred)

        # Update rolling estimate for upcoming synthetic laps.
        if i + 1 < len(out):
            recent = preds[max(0, i - 2) : i + 1]
            out.loc[i + 1, "RollingLapTime"] = float(np.mean(recent))

    out["PredLapTime"] = preds
    out["PredDegDelta"] = out["PredLapTime"].diff().fillna(0.0)
    return out


def _derive_pit_lap_from_pred(pred_df: pd.DataFrame) -> int:
    """Choose an ideal first-stop lap before predicted degradation cliff."""
    threshold = max(0.15, float(pred_df["PredDegDelta"].median() + pred_df["PredDegDelta"].std()))
    cliff = pred_df[(pred_df["LapNumber"] >= 6) & (pred_df["PredDegDelta"] > threshold)]
    if not cliff.empty:
        return max(2, int(cliff.iloc[0]["LapNumber"]) - 1)

    if pred_df["PredDegDelta"].dropna().empty:
        return max(8, int(pred_df["LapNumber"].median()))

    idx = pred_df["PredDegDelta"].idxmax()
    if pd.isna(idx):
        return max(8, int(pred_df["LapNumber"].median()))
    return max(2, int(pred_df.loc[idx, "LapNumber"]) - 1)


def _monte_carlo_stint_bands(
    base_pred_stint: pd.DataFrame,
    n_sims: int = 250,
    random_state: int = 42,
) -> dict[str, Any]:
    """Estimate uncertainty bands for stint pace using Monte Carlo noise."""
    rng = np.random.default_rng(random_state)

    base = base_pred_stint["PredLapTime"].to_numpy(dtype=float)
    deltas = base_pred_stint["PredDegDelta"].to_numpy(dtype=float)

    # Dynamic noise scales: mild lap-time noise + degradation uncertainty.
    base_noise = max(0.08, float(np.nanstd(base) * 0.10))
    deg_noise = max(0.04, float(np.nanstd(deltas) * 0.25))

    sim_avg_lap = []
    sim_deg = []
    for _ in range(max(50, n_sims)):
        lap_noise = rng.normal(0.0, base_noise, size=len(base))
        deg_noise_vec = rng.normal(0.0, deg_noise, size=len(base))

        sim = base + lap_noise + np.cumsum(deg_noise_vec) * 0.15
        sim_avg_lap.append(float(np.mean(sim)))
        sim_deg.append(float((np.mean(sim[-5:]) - np.mean(sim[:5])) / 5.0))

    avg_arr = np.array(sim_avg_lap, dtype=float)
    deg_arr = np.array(sim_deg, dtype=float)

    return {
        "avg_p10": float(np.percentile(avg_arr, 10)),
        "avg_p50": float(np.percentile(avg_arr, 50)),
        "avg_p90": float(np.percentile(avg_arr, 90)),
        "deg_p10": float(np.percentile(deg_arr, 10)),
        "deg_p50": float(np.percentile(deg_arr, 50)),
        "deg_p90": float(np.percentile(deg_arr, 90)),
        "avg_samples": avg_arr,
        "deg_samples": deg_arr,
    }


def _add_probability_outcomes(forecast_df: pd.DataFrame, n_sims: int) -> pd.DataFrame:
    """Derive win/podium/top10 probabilities from Monte Carlo samples."""
    out = forecast_df.copy()
    out["WinProbabilityPct"] = np.nan
    out["PodiumProbabilityPct"] = np.nan
    out["Top10ProbabilityPct"] = np.nan

    if "_SimAvgLapSamples" not in out.columns:
        return out

    for _, event_idx in out.groupby("EventName").groups.items():
        idx_list = list(event_idx)

        valid_row_idx = []
        sample_arrays = []
        for ridx in idx_list:
            samples = out.at[ridx, "_SimAvgLapSamples"]
            if isinstance(samples, np.ndarray) and samples.size > 1:
                valid_row_idx.append(ridx)
                sample_arrays.append(samples)

        if len(valid_row_idx) < 2:
            continue

        sim_len = min(int(max(50, n_sims)), min(arr.size for arr in sample_arrays))
        if sim_len < 10:
            continue

        # Matrix shape: drivers x simulations; lower avg lap is better.
        sim_mat = np.vstack([arr[:sim_len] for arr in sample_arrays])

        order = np.argsort(sim_mat, axis=0)
        ranks = np.empty_like(order)
        for col in range(sim_len):
            ranks[order[:, col], col] = np.arange(1, len(valid_row_idx) + 1)

        win_prob = (ranks == 1).mean(axis=1) * 100.0
        podium_prob = (ranks <= 3).mean(axis=1) * 100.0
        top10_prob = (ranks <= 10).mean(axis=1) * 100.0

        for pos, ridx in enumerate(valid_row_idx):
            out.at[ridx, "WinProbabilityPct"] = float(win_prob[pos])
            out.at[ridx, "PodiumProbabilityPct"] = float(podium_prob[pos])
            out.at[ridx, "Top10ProbabilityPct"] = float(top10_prob[pos])

    return out


def build_2026_all_driver_forecast(
    cache_dir: str,
    target_year: int = 2026,
    lookback_years: int = 5,
    n_monte_carlo: int = 250,
) -> dict[str, Any]:
    """
    Train on all available prior race data and predict all-driver future races.

    Returns a dictionary with future race forecasts and explainable summaries.
    """
    completed_2026, future_2026 = get_events_split_by_date(target_year, cache_dir=cache_dir)

    if completed_2026.empty:
        return {
            "available": False,
            "reason": "No completed 2026 races available yet for training.",
            "future_events": [],
            "predictions": pd.DataFrame(),
        }

    # Collect race laps from all available completed rounds in 2026.
    raw_frames = []
    for _, ev in completed_2026.iterrows():
        try:
            frame = load_all_drivers_laps(
                year=target_year,
                grand_prix=int(ev["RoundNumber"]),
                cache_dir=cache_dir,
            )
            frame["SeasonYear"] = int(target_year)
            raw_frames.append(frame)
        except Exception:
            continue

    # Add lookback history (all rounds) for deeper context where available.
    for y in range(max(2018, target_year - lookback_years), target_year):
        try:
            completed_y, _ = get_events_split_by_date(y, cache_dir=cache_dir)
        except Exception:
            continue

        for _, ev in completed_y.iterrows():
            try:
                frame = load_all_drivers_laps(
                    year=y,
                    grand_prix=int(ev["RoundNumber"]),
                    cache_dir=cache_dir,
                )
                frame["SeasonYear"] = int(y)
                raw_frames.append(frame)
            except Exception:
                continue

    if not raw_frames:
        return {
            "available": False,
            "reason": "Could not load race data required for forecasting.",
            "future_events": [],
            "predictions": pd.DataFrame(),
        }

    raw_all = pd.concat(raw_frames, ignore_index=True, sort=False)

    # Keep only rows with a driver code for per-driver modeling.
    raw_all = raw_all[raw_all.get("Driver").notna()].copy()
    drivers = sorted(raw_all["Driver"].dropna().astype(str).unique().tolist())

    forecasts = []
    for future_event in future_2026.itertuples(index=False):
        future_round = int(getattr(future_event, "RoundNumber"))
        future_name = str(getattr(future_event, "EventName"))

        for driver in drivers:
            driver_raw = raw_all[raw_all["Driver"].astype(str) == driver].copy()
            if len(driver_raw) < MIN_REQUIRED_LAPS:
                continue

            try:
                driver_clean = preprocess_laps(driver_raw)
                driver_feat = add_rolling_features(driver_clean, rolling_window=3)
                driver_feat = add_historical_features(driver_feat, target_year=target_year)

                # Train with all known data for this driver and evaluate on 2026 rows.
                result = train_and_evaluate(
                    driver_feat,
                    model_kind="random_forest",
                    target_year=target_year,
                )

                synth = _build_synthetic_future_stint(
                    driver_feat,
                    round_number=future_round,
                    target_year=target_year,
                    stint_laps=20,
                )
                pred_stint = _predict_stint_with_model(result.model, synth)
                bands = _monte_carlo_stint_bands(
                    pred_stint,
                    n_sims=n_monte_carlo,
                    random_state=(future_round * 100 + abs(hash(driver)) % 10_000),
                )

                avg_lap = float(pred_stint["PredLapTime"].mean())
                deg = float(pred_stint["PredLapTime"].tail(5).mean() - pred_stint["PredLapTime"].head(5).mean()) / 5.0
                pit_lap = _derive_pit_lap_from_pred(pred_stint)

                forecasts.append(
                    {
                        "EventName": future_name,
                        "RoundNumber": future_round,
                        "Driver": driver,
                        "PredictedAvgLapTimeSec": avg_lap,
                        "PredictedDegSecPerLap": deg,
                        "BestCaseAvgLapTimeSec": bands["avg_p10"],
                        "ExpectedAvgLapTimeSec": bands["avg_p50"],
                        "WorstCaseAvgLapTimeSec": bands["avg_p90"],
                        "AvgLapTimeCI80Low": bands["avg_p10"],
                        "AvgLapTimeCI80High": bands["avg_p90"],
                        "BestCaseDegSecPerLap": bands["deg_p10"],
                        "ExpectedDegSecPerLap": bands["deg_p50"],
                        "WorstCaseDegSecPerLap": bands["deg_p90"],
                        "AIIdealFirstPitLap": pit_lap,
                        "ModelR2OnKnownData": float(result.r2),
                        "_SimAvgLapSamples": bands["avg_samples"],
                    }
                )
            except Exception:
                continue

    forecast_df = pd.DataFrame(forecasts)
    if forecast_df.empty:
        return {
            "available": False,
            "reason": "Forecasting ran but produced no valid driver predictions.",
            "future_events": future_2026["EventName"].dropna().tolist(),
            "predictions": pd.DataFrame(),
        }

    forecast_df = _add_probability_outcomes(forecast_df, n_sims=n_monte_carlo)

    # Convert predicted lap pace into a simple race-strength score per event.
    event_tables = []
    for event_name, g in forecast_df.groupby("EventName"):
        t = g.sort_values("ExpectedAvgLapTimeSec").reset_index(drop=True)
        t["PredictedRank"] = np.arange(1, len(t) + 1)
        event_tables.append(t)
    forecast_df = pd.concat(event_tables, ignore_index=True)

    if "_SimAvgLapSamples" in forecast_df.columns:
        forecast_df = forecast_df.drop(columns=["_SimAvgLapSamples"])

    return {
        "available": True,
        "reason": "ok",
        "future_events": future_2026["EventName"].dropna().astype(str).tolist(),
        "predictions": forecast_df,
    }
