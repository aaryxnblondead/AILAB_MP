"""Streamlit app for F1 tire degradation prediction."""

from __future__ import annotations

import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_pipeline import PipelineConfig, build_modeling_frame, get_event_names
from future_forecast import build_2026_all_driver_forecast
from model import build_actual_vs_pred_plot, get_feature_importance_table, train_and_evaluate


def _build_position_plot(model_df):
    """Plot lap-wise position with F1 ranking direction (P1 at the top)."""
    pos_df = model_df[["LapNumber", "Position"]].dropna().copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        pos_df["LapNumber"],
        pos_df["Position"],
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    ax.set_title("Lap-wise Driver Race Position")
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Race Position")
    # Invert axis so lower numeric value (P1) appears at the top, as in F1 graphics.
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    return fig


def _detect_pit_windows(model_df):
    """
    Infer pit windows from tyre-life reset events.

    We remove in/out-laps in preprocessing to keep model quality high, so we detect
    likely pit windows where tyre life suddenly drops between consecutive laps.
    """
    if "TyreLife" not in model_df.columns:
        return []

    work = model_df.sort_values("LapNumber").copy()
    work["PrevTyreLife"] = work["TyreLife"].shift(1)
    work["TyreLifeDrop"] = work["PrevTyreLife"] - work["TyreLife"]

    # A big negative jump in age usually means a fresh set after a pit stop.
    candidates = work[work["TyreLifeDrop"] >= 3].copy()
    windows = []

    for _, row in candidates.iterrows():
        lap = int(row["LapNumber"])
        pre_mask = (work["LapNumber"] >= lap - 3) & (work["LapNumber"] < lap)
        post_mask = (work["LapNumber"] > lap) & (work["LapNumber"] <= lap + 3)

        pre_lap = work.loc[pre_mask, "LapTimeSeconds"].mean()
        post_lap = work.loc[post_mask, "LapTimeSeconds"].mean()
        lap_gain = pre_lap - post_lap if (pre_lap == pre_lap and post_lap == post_lap) else None

        pre_pos = work.loc[pre_mask, "Position"].mean() if "Position" in work.columns else None
        post_pos = work.loc[post_mask, "Position"].mean() if "Position" in work.columns else None
        pos_delta = pre_pos - post_pos if (pre_pos is not None and post_pos is not None) else None

        windows.append(
            {
                "lap": lap,
                "compound": row.get("Compound", "Unknown"),
                "lap_gain": lap_gain,
                "pos_delta": pos_delta,
            }
        )

    return windows


def _pit_window_explanations(windows):
    """Create plain-language strategy notes for non-F1 viewers."""
    if not windows:
        return [
            "No clear pit window was inferred from tyre-life resets in this cleaned stint data."
        ]

    messages = []
    for idx, win in enumerate(windows, start=1):
        lap = win["lap"]
        compound = win["compound"]
        lap_gain = win["lap_gain"]
        pos_delta = win["pos_delta"]

        base = (
            f"Window {idx}: around lap {lap}, the tyre-age reset suggests a pit stop "
            f"onto {compound} tires."
        )

        if lap_gain is not None and lap_gain > 0.15:
            perf = (
                f"Average pace improved by about {lap_gain:.2f}s/lap in the following laps, "
                "indicating the stop likely avoided the tire degradation cliff."
            )
        elif lap_gain is not None and lap_gain < -0.15:
            perf = (
                f"Average pace changed by {lap_gain:.2f}s/lap after the stop; this can happen "
                "when traffic or warm-up effects offset fresh-tire gains."
            )
        else:
            perf = (
                "Post-stop pace remained broadly stable, which still can be strategic by "
                "protecting against a larger late-stint drop-off."
            )

        if pos_delta is not None and pos_delta > 0.25:
            pos_text = (
                f"Track position improved by roughly {pos_delta:.1f} places in nearby laps, "
                "consistent with an effective undercut/clean-air window."
            )
        elif pos_delta is not None and pos_delta < -0.25:
            pos_text = (
                f"Track position shifted by {pos_delta:.1f} places shortly after; this can still "
                "be acceptable if it sets up stronger tires for a later phase."
            )
        else:
            pos_text = (
                "Position remained similar in nearby laps; the value of the stop is mainly "
                "long-run tire health and pace consistency."
            )

        messages.append(f"{base} {perf} {pos_text}")

    return messages


def _build_ai_vs_actual_pit_analysis(model_df, fitted_model, target_year: int):
    """Compare inferred actual pit timing against an AI-estimated ideal pit lap."""
    work = model_df.copy()
    if "SeasonYear" in work.columns:
        work = work[pd.to_numeric(work["SeasonYear"], errors="coerce") == float(target_year)]
    work = work.sort_values("LapNumber").copy()

    if work.empty:
        return {
            "available": False,
            "reason": "No target-season laps available for AI pit comparison.",
        }

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
    missing = [c for c in feature_cols if c not in work.columns]
    if missing:
        return {
            "available": False,
            "reason": f"Missing columns for AI pit analysis: {missing}",
        }

    # AI view of pace trend for each lap, generated from the trained model.
    work["AIPredLapTime"] = fitted_model.predict(work[feature_cols])
    work["AIDegDelta"] = work["AIPredLapTime"].diff()

    actual_windows = _detect_pit_windows(work)
    actual_first_lap = actual_windows[0]["lap"] if actual_windows else None

    if actual_first_lap is not None:
        baseline = work[work["LapNumber"] < actual_first_lap].copy()
    else:
        cap = max(8, int(work["LapNumber"].max() * 0.55))
        baseline = work[work["LapNumber"] <= cap].copy()

    # Detect the first meaningful degradation cliff from the AI-predicted pace curve.
    threshold = 0.18
    cliff = baseline[(baseline["AIDegDelta"] > threshold) & (baseline["LapNumber"] >= 6)]

    if not cliff.empty:
        cliff_lap = int(cliff.iloc[0]["LapNumber"])
        ideal_pit_lap = max(2, cliff_lap - 1)
    else:
        # Fallback: choose lap before the largest pace-loss jump.
        candidate = baseline[baseline["LapNumber"] >= 6].copy()
        candidate_valid = candidate.dropna(subset=["AIDegDelta", "LapNumber"]).copy()
        if candidate_valid.empty:
            lap_values = pd.to_numeric(work["LapNumber"], errors="coerce").dropna()
            ideal_pit_lap = int(lap_values.median()) if not lap_values.empty else 2
        else:
            max_idx = candidate_valid["AIDegDelta"].idxmax()
            if pd.isna(max_idx) or max_idx not in candidate_valid.index:
                lap_values = pd.to_numeric(candidate_valid["LapNumber"], errors="coerce").dropna()
                ideal_pit_lap = int(lap_values.median()) if not lap_values.empty else 2
                max_jump_row = None
            else:
                max_jump_row = candidate_valid.loc[max_idx]
            if max_jump_row is None:
                ideal_pit_lap = max(2, ideal_pit_lap)
            else:
                ideal_pit_lap = max(2, int(max_jump_row["LapNumber"]) - 1)

    lap_delta = None
    est_time_impact = None
    pos_impact_text = "Position impact could not be estimated from available windows."

    if actual_first_lap is not None:
        lap_delta = actual_first_lap - ideal_pit_lap

        # Approximate timing impact from AI degradation increments between windows.
        if lap_delta > 0:
            mask = (work["LapNumber"] > ideal_pit_lap) & (work["LapNumber"] <= actual_first_lap)
            deltas = work.loc[mask, "AIDegDelta"].fillna(0)
            est_time_impact = float(deltas[deltas > 0].sum())
        elif lap_delta < 0:
            mask = (work["LapNumber"] > actual_first_lap) & (work["LapNumber"] <= ideal_pit_lap)
            deltas = work.loc[mask, "AIDegDelta"].fillna(0)
            est_time_impact = float(deltas[deltas > 0].sum())
        else:
            est_time_impact = 0.0

        first_window = actual_windows[0]
        if first_window.get("pos_delta") is not None:
            pdlt = float(first_window["pos_delta"])
            if pdlt > 0.25:
                pos_impact_text = (
                    f"After the actual stop, the driver gained about {pdlt:.1f} places nearby, "
                    "suggesting a favorable traffic window."
                )
            elif pdlt < -0.25:
                pos_impact_text = (
                    f"After the actual stop, position shifted by {pdlt:.1f} places nearby; "
                    "the stop may have traded short-term track position for tire pace."
                )
            else:
                pos_impact_text = (
                    "Track position remained broadly stable around the actual stop window."
                )

    return {
        "available": True,
        "df": work,
        "ideal_pit_lap": ideal_pit_lap,
        "actual_windows": actual_windows,
        "actual_first_lap": actual_first_lap,
        "lap_delta": lap_delta,
        "est_time_impact": est_time_impact,
        "pos_impact_text": pos_impact_text,
    }


def _build_ai_vs_actual_plot(analysis):
    """Plot actual and AI-predicted lap time with ideal/actual pit markers."""
    work = analysis["df"]
    fig, ax = plt.subplots(figsize=(11, 4.8))

    ax.plot(
        work["LapNumber"],
        work["LapTimeSeconds"],
        label="Actual Lap Time",
        linewidth=2,
        alpha=0.85,
    )
    ax.plot(
        work["LapNumber"],
        work["AIPredLapTime"],
        label="AI-Predicted Lap Time",
        linewidth=2,
        linestyle="--",
        alpha=0.9,
    )

    ideal_lap = analysis["ideal_pit_lap"]
    ax.axvline(ideal_lap, color="green", linestyle="-.", linewidth=2, label=f"AI Ideal Pit (L{ideal_lap})")

    for idx, win in enumerate(analysis["actual_windows"], start=1):
        lap = int(win["lap"])
        label = "Actual Pit" if idx == 1 else None
        ax.axvline(lap, color="crimson", linestyle=":", linewidth=2, label=label)

    ax.set_title("AI Ideal Pit Timing vs Actual Pit Stops")
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (seconds)")
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


def _feature_description(name: str) -> str:
    """Translate technical feature names into plain-language explanations."""
    if "Compound" in name:
        return "Tire type (soft/medium/hard) strongly changes base pace and wear profile."
    if "TyreLife" in name:
        return "Older tires are generally slower, so tire age is central to degradation prediction."
    if "TrackTemp" in name:
        return "Track temperature affects grip and overheating risk, which shifts lap times."
    if "RollingLapTime" in name:
        return "Recent pace trend helps the model smooth traffic noise and identify true performance trend."
    if "HistMedianByCompoundTyreLife" in name:
        return "Historical same-circuit pace for this tire age anchors expected performance."
    if "HistMedianByCompound" in name:
        return "Historical same-circuit pace by tire type adds long-term context from prior seasons."
    if "YearOffset" in name:
        return "How old the reference season is; recent seasons are usually more relevant."
    if "LapNumber" in name:
        return "Race phase matters because fuel burn and strategy context change over laps."
    return "This variable contributes to the model's lap-time estimate."


st.set_page_config(
    page_title="Motorsport Analytics - Tire Degradation",
    page_icon="🏁",
    layout="wide",
)

st.title("Motorsport Analytics: Tire Degradation Prediction")
st.caption(
    "Predicting lap-time drop-off in race stints using FastF1 + scikit-learn"
)

with st.expander("Why these features?", expanded=False):
    st.markdown(
        """
- **Tire Compound** captures baseline grip and wear profile differences.
- **Tyre Life** directly measures aging of the current tire set.
- **Track Temperature** affects thermal degradation and grip consistency.
- **Rolling Lap Time** smooths traffic/noise so the model sees cleaner degradation signal.
        """
    )

# Sidebar controls
st.sidebar.header("Race Inputs")
current_year = datetime.now().year
selected_year = st.sidebar.number_input(
    "Season Year",
    min_value=2018,
    max_value=current_year,
    value=max(2024, current_year - 1),
    step=1,
)

cache_dir = st.sidebar.text_input("FastF1 Cache Directory", value="./cache")
rolling_window = st.sidebar.slider("Rolling Window (laps)", min_value=1, max_value=6, value=3)
history_years = st.sidebar.slider("Historical Seasons To Include", min_value=0, max_value=5, value=2)
model_kind = st.sidebar.selectbox(
    "Regressor",
    options=["random_forest", "gradient_boosting"],
)
driver_code = st.sidebar.text_input("Driver Code (e.g., VER, NOR, LEC)", value="VER").upper().strip()

# Dynamically fetch available event names for the selected year.
try:
    events = get_event_names(int(selected_year), cache_dir=cache_dir)
except Exception:
    events = []

if events:
    # Default to latest completed race in calendar order.
    selected_gp = st.sidebar.selectbox("Grand Prix", options=events, index=max(0, len(events) - 1))
else:
    selected_gp = st.sidebar.text_input("Grand Prix Name", value="Bahrain Grand Prix")

run_button = st.sidebar.button("Run Analysis", use_container_width=True)
run_future_button = st.sidebar.button("Run 2026 All-Driver Forecast", use_container_width=True)

if run_button:
    try:
        cfg = PipelineConfig(
            cache_dir=cache_dir,
            rolling_window=rolling_window,
            history_years=history_years,
        )

        with st.spinner("Loading and preprocessing race data..."):
            model_df = build_modeling_frame(
                year=int(selected_year),
                grand_prix=selected_gp,
                driver=driver_code,
                config=cfg,
            )

        with st.spinner("Training model and generating predictions..."):
            result = train_and_evaluate(
                model_df,
                model_kind=model_kind,
                target_year=int(selected_year),
            )
            fig = build_actual_vs_pred_plot(model_df, result)

        race_df = model_df.copy()
        if "SeasonYear" in race_df.columns:
            race_df = race_df[
                pd.to_numeric(race_df["SeasonYear"], errors="coerce") == float(selected_year)
            ].copy()
            race_df = race_df.sort_values("LapNumber")

        col1, col2 = st.columns(2)
        col1.metric("Mean Squared Error", f"{result.mse:.4f}")
        col2.metric("R-squared", f"{result.r2:.4f}")

        seasons_used = (
            sorted(pd.to_numeric(model_df.get("SeasonYear", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist())
            if "SeasonYear" in model_df.columns
            else [int(selected_year)]
        )
        st.caption(f"AI trained with seasons: {', '.join(map(str, seasons_used))}")

        st.subheader("Actual vs Predicted Lap Times")
        st.pyplot(fig, clear_figure=True)

        if "Position" in race_df.columns and race_df["Position"].notna().any():
            st.subheader("Lap-wise Driver Race Position")
            pos_fig = _build_position_plot(race_df)
            st.pyplot(pos_fig, clear_figure=True)
            st.caption("Lower values are better: position 1 is race leader.")

        st.subheader("Pit Window Strategy Notes")
        st.caption(
            "These notes are inferred from tire-age resets and nearby pace/position changes "
            "to explain pit timing in non-technical language."
        )
        pit_windows = _detect_pit_windows(race_df)
        pit_notes = _pit_window_explanations(pit_windows)
        for note in pit_notes:
            st.markdown(f"- {note}")

        st.subheader("AI Ideal vs Actual Pit Strategy")
        st.caption(
            "This compares the model's estimated best first-stop timing against what the driver "
            "actually did, and summarizes likely race impact."
        )
        pit_analysis = _build_ai_vs_actual_pit_analysis(model_df, result.model, int(selected_year))
        if pit_analysis.get("available"):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("AI Ideal First Pit Lap", f"L{pit_analysis['ideal_pit_lap']}")

            actual_first_lap = pit_analysis.get("actual_first_lap")
            col_b.metric(
                "Actual First Pit Lap",
                f"L{actual_first_lap}" if actual_first_lap is not None else "Not detected",
            )

            lap_delta = pit_analysis.get("lap_delta")
            if lap_delta is None:
                col_c.metric("Timing Difference", "N/A")
            elif lap_delta > 0:
                col_c.metric("Timing Difference", f"{lap_delta} lap(s) later")
            elif lap_delta < 0:
                col_c.metric("Timing Difference", f"{abs(lap_delta)} lap(s) earlier")
            else:
                col_c.metric("Timing Difference", "On target")

            est_time_impact = pit_analysis.get("est_time_impact")
            if est_time_impact is not None:
                if lap_delta is not None and lap_delta > 0:
                    st.markdown(
                        f"- Estimated degradation cost of pitting later than AI ideal: "
                        f"about **{est_time_impact:.2f}s** before the stop."
                    )
                elif lap_delta is not None and lap_delta < 0:
                    st.markdown(
                        f"- Estimated pace opportunity potentially left on table by pitting earlier: "
                        f"about **{est_time_impact:.2f}s** over nearby laps."
                    )
                else:
                    st.markdown("- Estimated timing impact: **~0.00s**, actual timing matched AI ideal.")

            st.markdown(f"- {pit_analysis['pos_impact_text']}")

            ai_pit_fig = _build_ai_vs_actual_plot(pit_analysis)
            st.pyplot(ai_pit_fig, clear_figure=True)
        else:
            st.info(pit_analysis.get("reason", "AI pit comparison is unavailable for this run."))

        st.subheader("Why AI Predicted This (Layman Explainability)")
        st.caption(
            "The model ranks which signals mattered most for lap-time prediction, then we translate "
            "them into plain English."
        )
        imp_df = get_feature_importance_table(result.model)
        if not imp_df.empty:
            top_imp = imp_df.head(6).copy()
            show_imp = top_imp[["feature", "importance_pct"]].rename(columns={"importance_pct": "Importance (%)"})
            st.dataframe(show_imp, use_container_width=True)
            for _, row in top_imp.head(3).iterrows():
                st.markdown(
                    f"- **{row['feature']}** ({row['importance_pct']:.1f}% impact): "
                    f"{_feature_description(str(row['feature']))}"
                )
        else:
            st.info("Feature-importance output is not available for the selected model type.")

        if "SeasonYear" in model_df.columns and len(seasons_used) > 1:
            st.subheader("Historical Context: Same Circuit, Previous Seasons")
            hist_summary = (
                model_df.groupby(["SeasonYear", "Compound"], as_index=False)["LapTimeSeconds"]
                .median()
                .rename(columns={"LapTimeSeconds": "MedianLapTimeSec"})
                .sort_values(["SeasonYear", "Compound"])
            )
            st.dataframe(hist_summary, use_container_width=True)
            st.caption(
                "This table helps compare whether this year's pace on each tire is faster or slower "
                "than previous seasons at the same circuit."
            )

        st.subheader("Modeling Dataset Preview")
        preview_cols = [
            "SeasonYear",
            "LapNumber",
            "Position",
            "Compound",
            "TyreLife",
            "TrackTemp",
            "LapTimeSeconds",
            "RollingLapTime",
            "HistMedianByCompoundTyreLife",
            "HistMedianByCompound",
            "YearOffset",
        ]
        preview_cols = [c for c in preview_cols if c in race_df.columns]
        st.dataframe(race_df[preview_cols], use_container_width=True)

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.info(
            "Tip: choose a completed race weekend and a valid 3-letter driver code "
            "(for example VER, LEC, NOR)."
        )
        st.code(traceback.format_exc())
else:
    st.info("Configure race parameters in the sidebar and click 'Run Analysis'.")

if run_future_button:
    st.header("AI Predictor: 2026 Future Races (All Drivers)")
    st.caption(
        "This module trains per-driver models using all available completed races and historical seasons, "
        "then forecasts future 2026 race pace and ideal first pit windows."
    )

    with st.spinner("Building all-driver forecast for future 2026 races..."):
        forecast_result = build_2026_all_driver_forecast(
            cache_dir=cache_dir,
            target_year=2026,
            lookback_years=max(1, int(history_years) + 2),
            n_monte_carlo=400,
        )

    if not forecast_result.get("available"):
        st.warning(forecast_result.get("reason", "Forecast unavailable."))
    else:
        pred = forecast_result["predictions"].copy()
        events = forecast_result.get("future_events", [])
        st.success(f"Generated AI predictions for {len(events)} upcoming 2026 race(s).")

        event_choice = st.selectbox(
            "Select Future Race",
            options=sorted(pred["EventName"].unique().tolist()),
        )
        view = pred[pred["EventName"] == event_choice].copy().sort_values("PredictedRank")

        st.subheader(f"Predicted Driver Ranking - {event_choice}")
        st.dataframe(
            view[
                [
                    "PredictedRank",
                    "Driver",
                    "ExpectedAvgLapTimeSec",
                    "BestCaseAvgLapTimeSec",
                    "WorstCaseAvgLapTimeSec",
                    "WinProbabilityPct",
                    "PodiumProbabilityPct",
                    "Top10ProbabilityPct",
                    "ExpectedDegSecPerLap",
                    "AIIdealFirstPitLap",
                    "ModelR2OnKnownData",
                ]
            ],
            use_container_width=True,
        )

        uncertainty_view = view[
            [
                "Driver",
                "BestCaseAvgLapTimeSec",
                "ExpectedAvgLapTimeSec",
                "WorstCaseAvgLapTimeSec",
            ]
        ].copy()
        uncertainty_view = uncertainty_view.sort_values("ExpectedAvgLapTimeSec")

        fig_unc, ax_unc = plt.subplots(figsize=(10, 5))
        ax_unc.errorbar(
            uncertainty_view["Driver"],
            uncertainty_view["ExpectedAvgLapTimeSec"],
            yerr=[
                uncertainty_view["ExpectedAvgLapTimeSec"] - uncertainty_view["BestCaseAvgLapTimeSec"],
                uncertainty_view["WorstCaseAvgLapTimeSec"] - uncertainty_view["ExpectedAvgLapTimeSec"],
            ],
            fmt="o",
            ecolor="gray",
            capsize=4,
        )
        ax_unc.set_title(f"Uncertainty Bands (Best / Expected / Worst) - {event_choice}")
        ax_unc.set_ylabel("Avg Lap Time (sec)")
        ax_unc.set_xlabel("Driver")
        ax_unc.grid(alpha=0.25)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_unc, clear_figure=True)

        st.subheader("Layman Explanation")
        st.markdown(
            "- **Best/Expected/Worst Avg Lap Time**: Monte Carlo uncertainty band. Best and worst are plausible bounds, expected is the median outcome."
        )
        st.markdown(
            "- **ExpectedDegSecPerLap**: expected tire degradation slope per lap under simulated uncertainty."
        )
        st.markdown(
            "- **Win/Podium/Top-10 Probability**: chance (%) from Monte Carlo race simulations where the driver finishes in that bracket."
        )
        st.markdown(
            "- **AIIdealFirstPitLap**: model's best estimate of when to pit before a major tire drop-off."
        )
        st.markdown(
            "- **ModelR2OnKnownData**: how well that driver's model matched known races; higher is generally more reliable."
        )

        st.subheader("Top 5 Predicted Drivers")
        top5 = view.head(5).copy()
        st.bar_chart(top5.set_index("Driver")["ExpectedAvgLapTimeSec"])

        st.subheader("Win Probability Leaderboard")
        prob_view = view[["Driver", "WinProbabilityPct"]].copy().sort_values("WinProbabilityPct", ascending=False)
        st.bar_chart(prob_view.set_index("Driver")["WinProbabilityPct"])
