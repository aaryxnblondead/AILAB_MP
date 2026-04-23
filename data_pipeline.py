"""Data ingestion and preprocessing utilities for F1 tire degradation modeling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import DataNotLoadedError


MIN_REQUIRED_LAPS = 6


def _apply_filter_with_floor(df: pd.DataFrame, mask: pd.Series, min_rows: int) -> pd.DataFrame:
    """Apply a row filter only if enough samples remain for modeling."""
    candidate = df[mask].copy()
    if len(candidate) >= min_rows:
        return candidate
    return df


@dataclass
class PipelineConfig:
    """Configuration object for data ingestion and preprocessing."""

    cache_dir: str = "./cache"
    rolling_window: int = 3
    history_years: int = 1


def enable_fastf1_cache(cache_dir: str = "./cache") -> None:
    """
    Enable FastF1 local cache to avoid repeated downloads.

    Caching is critical for both speed and API friendliness when running multiple
    experiments from Streamlit.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path.resolve()))


def _is_valid_track_status(track_status: object) -> bool:
    """
    Filter function for TrackStatus.

    FastF1 TrackStatus is often an encoded string of status codes. We remove
    laps that likely happened under Safety Car or VSC signals:
    - 4: Safety Car
    - 6/7: VSC related statuses
    """
    if pd.isna(track_status):
        return True

    status_str = str(track_status)
    return not any(code in status_str for code in ["4", "6", "7"])


def load_driver_laps(
    year: int,
    grand_prix: str,
    driver: str,
    cache_dir: str = "./cache",
) -> pd.DataFrame:
    """
    Load race laps for a specific driver.

    Parameters
    ----------
    year : int
        Season year.
    grand_prix : str
        Event name from FastF1 schedule.
    driver : str
        Driver identifier (e.g., VER, NOR, LEC).
    cache_dir : str
        Local cache location for FastF1.

    Returns
    -------
    pd.DataFrame
        Raw driver lap data with weather columns merged.
    """
    enable_fastf1_cache(cache_dir)

    session = fastf1.get_session(year, grand_prix, "R")

    # Explicitly request lap data and keep a compatibility fallback for
    # FastF1 versions where `laps=` may not be accepted.
    try:
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except TypeError:
        session.load(telemetry=False, weather=True, messages=False)

    # Some environments may return without populated lap data; retry once
    # with a full default load before failing with a clear message.
    try:
        _ = session.laps
    except DataNotLoadedError:
        session.load()
        try:
            _ = session.laps
        except DataNotLoadedError as exc:
            raise RuntimeError(
                "FastF1 session loaded but lap data is unavailable. "
                "Please retry and verify the selected event exists and network "
                "access to FastF1 data providers is available."
            ) from exc

    if session.laps.empty:
        raise ValueError(
            f"No lap timing data is available for {year} {grand_prix} race yet. "
            "Select a completed event with published race timing data."
        )

    laps = session.laps.pick_drivers(driver).copy()
    if laps.empty:
        raise ValueError(
            f"No race laps found for driver '{driver}' in {year} {grand_prix}."
        )

    # Weather rows are aligned by lap index and include TrackTemp among others.
    laps_df = laps.reset_index(drop=True)
    laps_df["EventName"] = str(session.event["EventName"])
    laps_df["RoundNumber"] = int(session.event["RoundNumber"])
    laps_df["EventDate"] = pd.to_datetime(session.event["EventDate"], errors="coerce")
    try:
        weather = laps.get_weather_data().reset_index(drop=True)
        laps_df = pd.concat([laps_df, weather], axis=1)
    except Exception:
        # Keep going even if weather endpoint is unavailable; downstream
        # preprocessing can fall back to AirTemp where possible.
        pass

    return laps_df


def load_all_drivers_laps(
    year: int,
    grand_prix: str | int,
    cache_dir: str = "./cache",
) -> pd.DataFrame:
    """Load race laps for all drivers in a session."""
    enable_fastf1_cache(cache_dir)

    session = fastf1.get_session(year, grand_prix, "R")
    try:
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except TypeError:
        session.load(telemetry=False, weather=True, messages=False)

    try:
        _ = session.laps
    except DataNotLoadedError:
        session.load()

    if session.laps.empty:
        raise ValueError(
            f"No lap timing data is available for {year} {grand_prix} race."
        )

    laps = session.laps.copy().reset_index(drop=True)
    laps["EventName"] = str(session.event["EventName"])
    laps["RoundNumber"] = int(session.event["RoundNumber"])
    laps["EventDate"] = pd.to_datetime(session.event["EventDate"], errors="coerce")

    try:
        weather = laps.get_weather_data().reset_index(drop=True)
        laps = pd.concat([laps, weather], axis=1)
    except Exception:
        pass

    return laps


def get_events_split_by_date(year: int, cache_dir: str = "./cache") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (completed_events, future_events) for race sessions in a season."""
    enable_fastf1_cache(cache_dir)
    schedule = fastf1.get_event_schedule(year).copy()

    if "Session5" in schedule.columns:
        schedule = schedule[schedule["Session5"] == "Race"].copy()

    schedule["EventDate"] = pd.to_datetime(schedule.get("EventDate"), errors="coerce")
    today = pd.Timestamp(datetime.utcnow().date())

    completed = schedule[schedule["EventDate"].notna() & (schedule["EventDate"] <= today)].copy()
    future = schedule[schedule["EventDate"].notna() & (schedule["EventDate"] > today)].copy()
    return completed, future


def _safe_load_driver_laps_for_round(
    year: int,
    round_number: int,
    driver: str,
    cache_dir: str,
) -> pd.DataFrame:
    """Load a driver's race laps by season round and annotate season metadata."""
    laps = load_driver_laps(year, round_number, driver, cache_dir=cache_dir)
    laps["SeasonYear"] = int(year)
    return laps


def preprocess_laps(raw_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw laps and construct core modeling columns.

    Reasoning behind selected features:
    - Compound: captures tire material characteristics and durability.
    - TyreLife: direct proxy for wear progression over the stint.
    - TrackTemp: environmental factor affecting grip and degradation rate.
    """
    df = raw_laps.copy()

    required_columns = ["LapTime", "Compound", "TyreLife", "LapNumber"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in FastF1 data: {missing}")

    # Remove pit-related laps when those indicators are available.
    # We apply each filter with a floor to avoid deleting nearly all rows for
    # sessions where upstream timing/weather data is partially missing.
    if "PitOutTime" in df.columns:
        df = _apply_filter_with_floor(df, df["PitOutTime"].isna(), MIN_REQUIRED_LAPS)
    if "PitInTime" in df.columns:
        df = _apply_filter_with_floor(df, df["PitInTime"].isna(), MIN_REQUIRED_LAPS)

    # Remove laps flagged as inaccurate if available (traffic/incidents often included).
    if "IsAccurate" in df.columns:
        df = _apply_filter_with_floor(df, df["IsAccurate"] == True, MIN_REQUIRED_LAPS)  # noqa: E712

    # Remove Safety Car / VSC laps based on TrackStatus encodings.
    if "TrackStatus" in df.columns:
        status_mask = df["TrackStatus"].apply(_is_valid_track_status)
        df = _apply_filter_with_floor(df, status_mask, MIN_REQUIRED_LAPS)

    # Keep only dry compounds for a cleaner degradation signal.
    valid_compounds = ["SOFT", "MEDIUM", "HARD"]
    df["Compound"] = df["Compound"].astype(str).str.upper()
    dry_df = df[df["Compound"].isin(valid_compounds)].copy()
    if len(dry_df) >= MIN_REQUIRED_LAPS:
        df = dry_df
    else:
        # Fall back to all known compounds when dry-only rows are too sparse.
        df["Compound"] = df["Compound"].replace({"NAN": "UNKNOWN", "NONE": "UNKNOWN"})
        df.loc[df["Compound"].str.strip() == "", "Compound"] = "UNKNOWN"

    # Convert lap time target to numeric seconds for regression.
    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()

    # Ensure temperature exists; fallback to AirTemp if TrackTemp is missing.
    if "TrackTemp" not in df.columns:
        if "AirTemp" in df.columns:
            df["TrackTemp"] = df["AirTemp"]
        else:
            df["TrackTemp"] = pd.NA

    # If TrackTemp exists but has holes, use AirTemp as secondary source.
    if "AirTemp" in df.columns:
        df["TrackTemp"] = df["TrackTemp"].fillna(df["AirTemp"])

    # Convert key fields to numeric before imputation/fallback logic.
    df["TrackTemp"] = pd.to_numeric(df["TrackTemp"], errors="coerce")
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")

    # Robust temperature fallback to avoid dropping all rows when weather API is unavailable.
    if df["TrackTemp"].isna().all():
        df["TrackTemp"] = 30.0
    else:
        df["TrackTemp"] = df["TrackTemp"].fillna(df["TrackTemp"].median())

    # Tyre life fallback: if unavailable, use lap progression proxy.
    if df["TyreLife"].isna().all():
        min_lap = df["LapNumber"].dropna().min()
        if pd.isna(min_lap):
            min_lap = 1.0
        df["TyreLife"] = (df["LapNumber"] - min_lap + 1).clip(lower=1)
    else:
        df["TyreLife"] = df["TyreLife"].ffill().bfill()

    # Remove rows where key fields are missing.
    df = df.dropna(subset=["LapTimeSeconds", "TyreLife", "TrackTemp", "LapNumber"])

    # Keep lap-wise race position if available in timing payload.
    if "Position" in df.columns:
        df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
        # Position can be temporarily missing on some laps due timing gaps.
        df["Position"] = df["Position"].ffill().bfill()

    df = df.dropna(subset=["TyreLife", "TrackTemp", "LapNumber"])
    df = df.sort_values("LapNumber").reset_index(drop=True)

    if len(df) < MIN_REQUIRED_LAPS:
        raise ValueError(
            "Not enough laps after preprocessing to build a reliable model."
        )

    return df


def add_rolling_features(df: pd.DataFrame, rolling_window: int = 3) -> pd.DataFrame:
    """
    Add rolling-average lap-time feature.

    This smooths one-off traffic disturbances and highlights the underlying
    degradation curve.
    """
    if rolling_window < 1:
        raise ValueError("rolling_window must be >= 1")

    out = df.copy()
    out["RollingLapTime"] = (
        out["LapTimeSeconds"].rolling(window=rolling_window, min_periods=1).mean()
    )
    return out


def add_historical_features(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """
    Add historical context features using prior seasons at the same round/circuit.

    These features help the model anchor expected pace by tyre type and tyre age,
    while still adapting to current-race conditions.
    """
    out = df.copy()
    out["SeasonYear"] = pd.to_numeric(out.get("SeasonYear", target_year), errors="coerce").fillna(target_year)
    out["YearOffset"] = (target_year - out["SeasonYear"]).clip(lower=0)
    out["TyreLifeRounded"] = out["TyreLife"].round().astype(int)

    hist = out[out["SeasonYear"] < target_year].copy()

    if hist.empty:
        # No historical races available; fallback keeps model shape stable.
        out["HistMedianByCompoundTyreLife"] = out["RollingLapTime"]
        out["HistMedianByCompound"] = out["RollingLapTime"]
        return out

    by_compound_tyre = (
        hist.groupby(["Compound", "TyreLifeRounded"], as_index=False)["LapTimeSeconds"]
        .median()
        .rename(columns={"LapTimeSeconds": "HistMedianByCompoundTyreLife"})
    )
    by_compound = (
        hist.groupby("Compound", as_index=False)["LapTimeSeconds"]
        .median()
        .rename(columns={"LapTimeSeconds": "HistMedianByCompound"})
    )

    out = out.merge(by_compound_tyre, on=["Compound", "TyreLifeRounded"], how="left")
    out = out.merge(by_compound, on=["Compound"], how="left")

    # Fill historical priors with progressively broader defaults.
    out["HistMedianByCompound"] = out["HistMedianByCompound"].fillna(out["RollingLapTime"])
    out["HistMedianByCompoundTyreLife"] = out["HistMedianByCompoundTyreLife"].fillna(
        out["HistMedianByCompound"]
    )

    return out


def get_event_names(
    year: int,
    cache_dir: str = "./cache",
    completed_only: bool = True,
) -> list[str]:
    """Return race event names for a given season year.

    By default, only races whose event date has passed are returned so the UI
    does not default to future rounds that have no timing data yet.
    """
    enable_fastf1_cache(cache_dir)
    schedule = fastf1.get_event_schedule(year)
    if "EventName" not in schedule.columns:
        return []

    race_schedule = schedule.copy()
    if "Session5" in race_schedule.columns:
        race_schedule = race_schedule[race_schedule["Session5"] == "Race"]

    if completed_only and "EventDate" in race_schedule.columns:
        today = pd.Timestamp(datetime.utcnow().date())
        event_dates = pd.to_datetime(race_schedule["EventDate"], errors="coerce")
        race_schedule = race_schedule[event_dates.notna() & (event_dates <= today)]

    return race_schedule["EventName"].dropna().astype(str).drop_duplicates().tolist()


def _resolve_equivalent_event_name(
    source_event: pd.Series,
    target_year: int,
    cache_dir: str,
) -> str | None:
    """Resolve equivalent race event in another season by name/location fallback."""
    enable_fastf1_cache(cache_dir)
    schedule = fastf1.get_event_schedule(target_year).copy()
    if schedule.empty or "EventName" not in schedule.columns:
        return None

    if "Session5" in schedule.columns:
        schedule = schedule[schedule["Session5"] == "Race"].copy()
    if schedule.empty:
        return None

    source_name = str(source_event.get("EventName", "")).strip()
    source_location = str(source_event.get("Location", "")).strip().lower()
    source_country = str(source_event.get("Country", "")).strip().lower()

    # 1) Exact event name match.
    exact = schedule[schedule["EventName"].astype(str).str.lower() == source_name.lower()]
    if not exact.empty:
        return str(exact.iloc[0]["EventName"])

    # 2) Location match (track/city-level) when available.
    if "Location" in schedule.columns and source_location:
        by_loc = schedule[
            schedule["Location"].astype(str).str.strip().str.lower() == source_location
        ]
        if not by_loc.empty:
            return str(by_loc.iloc[0]["EventName"])

    # 3) Country fallback.
    if "Country" in schedule.columns and source_country:
        by_country = schedule[
            schedule["Country"].astype(str).str.strip().str.lower() == source_country
        ]
        if not by_country.empty:
            return str(by_country.iloc[0]["EventName"])

    # 4) Token-overlap fallback for naming changes.
    source_tokens = [t for t in source_name.lower().replace("grand prix", "").split() if t]
    if source_tokens:
        scores = []
        for _, row in schedule.iterrows():
            name = str(row["EventName"]).lower()
            score = sum(1 for tok in source_tokens if tok in name)
            scores.append(score)
        schedule = schedule.assign(_match_score=scores)
        schedule = schedule.sort_values("_match_score", ascending=False)
        if not schedule.empty and int(schedule.iloc[0]["_match_score"]) > 0:
            return str(schedule.iloc[0]["EventName"])

    return None


def build_modeling_frame(
    year: int,
    grand_prix: str,
    driver: str,
    config: PipelineConfig,
) -> pd.DataFrame:
    """End-to-end data assembly for downstream modeling."""
    enable_fastf1_cache(config.cache_dir)

    # Resolve the selected event, then backfill previous seasons using equivalent
    # event/track matching. Round numbers vary by season, so round-only matching
    # can miss valid historical races.
    event = fastf1.get_event(year, grand_prix)
    round_number = int(event["RoundNumber"])

    frames = []

    current_raw = _safe_load_driver_laps_for_round(
        year=year,
        round_number=round_number,
        driver=driver,
        cache_dir=config.cache_dir,
    )
    frames.append(current_raw)

    for offset in range(1, max(0, int(config.history_years)) + 1):
        prev_year = year - offset
        if prev_year < 2018:
            continue
        try:
            prev_event_name = _resolve_equivalent_event_name(
                source_event=event,
                target_year=prev_year,
                cache_dir=config.cache_dir,
            )
            if not prev_event_name:
                continue

            prev_raw = load_driver_laps(
                year=prev_year,
                grand_prix=prev_event_name,
                driver=driver,
                cache_dir=config.cache_dir,
            )
            prev_raw["SeasonYear"] = int(prev_year)
            frames.append(prev_raw)
        except Exception:
            # Historical coverage may be incomplete; skip unavailable seasons.
            continue

    merged_raw = pd.concat(frames, ignore_index=True)
    clean = preprocess_laps(merged_raw)
    featured = add_rolling_features(clean, rolling_window=config.rolling_window)
    featured = add_historical_features(featured, target_year=year)

    # Keep target season laps in front for easier plotting/readability.
    featured = featured.sort_values(["SeasonYear", "LapNumber"], ascending=[False, True]).reset_index(drop=True)
    return featured
