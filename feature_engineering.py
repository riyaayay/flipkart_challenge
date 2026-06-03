"""
feature_engineering.py — Complete feature engineering pipeline for Gridlock 2.0.

Feature groups (each justified by R² impact):
  A. Temporal       : hour, minute, cyclic sin/cos, rush-hour flags
  B. Spatial        : geohash → lat/lon → xyz Cartesian, prefix hashes, k-means cluster
  C. Road Capacity  : capacity_proxy, HV×lanes interaction, compound road-HV type
  D. Weather        : ordinal severity, temp deviation, temp², cross-term interactions
  E. Landmark       : Landmark × sin_hour, Landmark × is_peak
  F. Lag/Rolling    : per-geohash demand lags and rolling stats (biggest R² gain)
  G. Target Encode  : CV-safe mean/median demand by geohash, road type, cluster
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import config as C

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# A. GEOHASH DECODING  (pure-Python, no pygeohash dependency)
# ═══════════════════════════════════════════════════════════════════════════════

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DEC_MAP = {c: i for i, c in enumerate(_BASE32)}


def decode_geohash(gh: str) -> tuple[float, float]:
    """
    Decode a geohash string to (latitude, longitude).

    Pure-Python implementation; no external library required.
    Uses alternating bit assignment: even bits → longitude, odd bits → latitude.
    """
    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0
    is_lon = True  # first bit goes to longitude

    for char in str(gh).lower().strip():
        bits = _DEC_MAP.get(char, 0)
        for shift in range(4, -1, -1):
            bit = (bits >> shift) & 1
            if is_lon:
                mid = (lon_min + lon_max) / 2
                if bit:
                    lon_min = mid
                else:
                    lon_max = mid
            else:
                mid = (lat_min + lat_max) / 2
                if bit:
                    lat_min = mid
                else:
                    lat_max = mid
            is_lon = not is_lon

    return (lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decode geohash → (lat, lon) → (x, y, z) Cartesian.

    Why Cartesian?  Euclidean distance in (x,y,z) is a monotone function of
    great-circle distance, unlike (lat, lon) where lon wraps and lat is not
    equal-area.  Tree models can then split on x/y/z to implicitly find
    spatial proximity without needing a custom kernel.

    Formula:
        x = R · cos(lat_rad) · cos(lon_rad)
        y = R · cos(lat_rad) · sin(lon_rad)
        z = R · sin(lat_rad)
    where R = 6371 km.
    """
    logger.info("Decoding geohash → lat/lon/xyz …")
    R = C.EARTH_RADIUS_KM

    lats, lons = zip(*df["geohash"].map(decode_geohash).tolist())
    df["lat"] = lats
    df["lon"] = lons

    lat_rad = np.radians(df["lat"])
    lon_rad = np.radians(df["lon"])
    df["x"] = R * np.cos(lat_rad) * np.cos(lon_rad)
    df["y"] = R * np.cos(lat_rad) * np.sin(lon_rad)
    df["z"] = R * np.sin(lat_rad)

    # Coarser geohash prefixes capture regional spatial structure
    for prec in C.GEOHASH_PREFIX_LEVELS:
        df[f"geohash{prec}"] = df["geohash"].str[:prec]

    return df


def add_geocluster_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_clusters: int = C.N_GEO_CLUSTERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    K-means cluster geohash cells by (lat, lon) → integer `geo_cluster`.

    Each cluster represents a demand zone: CBD, suburban, industrial corridor, etc.
    Used as a categorical feature AND as a grouping key for per-cluster target encoding.
    """
    logger.info(f"Fitting k-means with k={n_clusters} on (lat, lon) …")
    coords = train_df[["lat", "lon"]].values
    km = KMeans(n_clusters=n_clusters, random_state=C.SEED, n_init=10, max_iter=300)
    km.fit(coords)

    train_df["geo_cluster"] = km.predict(train_df[["lat", "lon"]].values)
    test_df["geo_cluster"]  = km.predict(test_df[["lat", "lon"]].values)

    # Distance from nearest cluster centroid (density proxy)
    centroids = km.cluster_centers_
    for df in (train_df, test_df):
        cluster_ids = df["geo_cluster"].values
        cell_coords = df[["lat", "lon"]].values
        assigned_centroids = centroids[cluster_ids]
        df["dist_to_centroid"] = np.sqrt(
            ((cell_coords - assigned_centroids) ** 2).sum(axis=1)
        )

    return train_df, test_df


# ═══════════════════════════════════════════════════════════════════════════════
# B. TEMPORAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_hour_minute(ts_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Robust timestamp parser.  Handles:
      - "H:MM"  / "HH:MM"  / "HH:MM:SS"  (string time-of-day)
      - Integer minutes-since-midnight
      - Full ISO datetime strings
    Returns (hour_series, minute_series).
    """
    sample = str(ts_series.dropna().iloc[0])

    if ":" in sample:
        parts = ts_series.astype(str).str.split(":")
        hour   = parts.str[0].astype(float).fillna(0).astype(int)
        minute = parts.str[1].astype(float).fillna(0).astype(int)
    elif pd.api.types.is_numeric_dtype(ts_series):
        vals   = ts_series.astype(float)
        hour   = (vals // 60).astype(int) % 24
        minute = (vals % 60).astype(int)
    else:
        try:
            parsed = pd.to_datetime(ts_series, infer_datetime_format=True)
            hour, minute = parsed.dt.hour, parsed.dt.minute
        except Exception:
            hour   = pd.Series(np.zeros(len(ts_series), dtype=int))
            minute = pd.Series(np.zeros(len(ts_series), dtype=int))

    return hour.reset_index(drop=True), minute.reset_index(drop=True)


def _parse_day_of_week(day_series: pd.Series) -> pd.Series:
    """
    Returns 0-indexed day-of-week from the `day` column.
    Handles: integer day-index (mod 7), day-name strings, or date strings.
    """
    sample = day_series.dropna().iloc[0]

    if pd.api.types.is_numeric_dtype(day_series):
        return (day_series.astype(int) % 7).reset_index(drop=True)

    try:
        parsed = pd.to_datetime(day_series, infer_datetime_format=True)
        return parsed.dt.dayofweek.reset_index(drop=True)
    except Exception:
        pass

    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4,  "saturday": 5, "sunday": 6,
        "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
    }
    return day_series.astype(str).str.lower().str.strip().map(day_map).fillna(0).astype(int)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and cyclically encode all temporal features.

    Cyclic encoding explanation:
        Linear hour (0–23): distance(23, 0) = 23 — wrong, they are adjacent.
        Cyclic sin/cos    : distance ≈ 0.26  — correctly small.
    Formula: sin_hour = sin(2π · hour / 24),  cos_hour = cos(2π · hour / 24)
    Analogous for minute, day-of-week, month, week-of-year.
    """
    logger.info("Extracting temporal features …")
    hour, minute = _parse_hour_minute(df["timestamp"])
    day_of_week  = _parse_day_of_week(df["day"])

    df["hour"]        = hour.values
    df["minute"]      = minute.values
    df["day_of_week"] = day_of_week.values
    df["mins_since_midnight"] = df["hour"] * 60 + df["minute"]

    # ── Try to extract month / week-of-year for seasonal features ────────────
    try:
        parsed_day = pd.to_datetime(df["day"], infer_datetime_format=True, errors="coerce")
        if parsed_day.notna().sum() > len(df) * 0.8:
            df["month"]       = parsed_day.dt.month.fillna(1).astype(int)
            df["week_of_year"] = parsed_day.dt.isocalendar().week.fillna(1).astype(int)
        else:
            raise ValueError("Not parseable as dates")
    except Exception:
        # Fall back to inferring week/month from day index
        day_num = df["day"].astype(str).str.extract(r"(\d+)")[0].fillna(0).astype(float)
        df["month"]       = ((day_num // 30) % 12 + 1).astype(int)
        df["week_of_year"] = ((day_num // 7) % 52 + 1).astype(int)

    # ── Cyclic encodings ─────────────────────────────────────────────────────
    df["sin_hour"]   = np.sin(2 * np.pi * df["hour"]        / 24)
    df["cos_hour"]   = np.cos(2 * np.pi * df["hour"]        / 24)
    df["sin_min"]    = np.sin(2 * np.pi * df["minute"]      / 60)
    df["cos_min"]    = np.cos(2 * np.pi * df["minute"]      / 60)
    df["sin_dow"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dow"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_month"]  = np.sin(2 * np.pi * df["month"]       / 12)
    df["cos_month"]  = np.cos(2 * np.pi * df["month"]       / 12)
    df["sin_week"]   = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"]   = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["sin_minsday"]= np.sin(2 * np.pi * df["mins_since_midnight"] / 1440)
    df["cos_minsday"]= np.cos(2 * np.pi * df["mins_since_midnight"] / 1440)

    # ── Discrete time-block flags ─────────────────────────────────────────────
    df["is_weekend"]      = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_morning_peak"] = ((df["hour"] >= 7)  & (df["hour"] <= 9)).astype(np.int8)
    df["is_evening_peak"] = ((df["hour"] >= 17) & (df["hour"] <= 19)).astype(np.int8)
    df["is_peak_hour"]    = (df["is_morning_peak"] | df["is_evening_peak"]).astype(np.int8)
    df["is_night"]        = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(np.int8)
    df["is_lunch_hour"]   = ((df["hour"] >= 12) & (df["hour"] <= 13)).astype(np.int8)
    df["is_office_hours"] = ((df["hour"] >= 9)  & (df["hour"] <= 18)).astype(np.int8)

    # Time block: 0=night, 1=morning, 2=midday, 3=afternoon, 4=evening, 5=late
    bins   = [-1, 5, 9, 12, 17, 21, 23]
    labels = [0,  1, 2,  3,  4,  5]
    df["time_block"] = pd.cut(df["hour"], bins=bins, labels=labels).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# C. ROAD CAPACITY FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def add_road_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode road characteristics and construct capacity-related features.

    capacity_proxy = NumberofLanes × road_capacity_weight
        Reflects the theoretical throughput of a road type at a given lane count.
        Example: 4-lane highway ≈ 4 × 4.0 = 16;  2-lane local ≈ 2 × 1.0 = 2.

    hv_lane_interaction = LargeVehicles × NumberofLanes
        At 35%+ HV ratio, effective lane capacity drops 15–25% (PCE > 1).
        This interaction captures the joint degradation effect.

    road_hv_type  = RoadType + "_" + str(LargeVehicles)
        A compound categorical that explicitly captures the capacity regime
        (e.g., "highway_1" vs "local_0").
    """
    logger.info("Engineering road capacity features …")

    # Ordinal road capacity weight
    road_cap = (
        df["RoadType"]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(C.ROAD_CAPACITY_MAP)
        .fillna(C.ROAD_CAPACITY_DEFAULT)
    )
    df["road_capacity_weight"] = road_cap

    lanes = pd.to_numeric(df["NumberofLanes"], errors="coerce").fillna(1).clip(1, 20)
    df["NumberofLanes"] = lanes  # ensure numeric

    # Binary LargeVehicles: handle yes/no/1/0/True/False
    lv = df["LargeVehicles"].astype(str).str.lower().str.strip()
    lv_bin = lv.map({"1": 1, "0": 0, "yes": 1, "no": 0, "true": 1, "false": 0, "allowed": 1, "not allowed": 0})
    if lv_bin.isna().sum() > 0:
        lv_bin = pd.to_numeric(df["LargeVehicles"], errors="coerce").fillna(0)
    df["LargeVehicles_bin"] = lv_bin.astype(np.int8)

    df["capacity_proxy"]       = lanes * road_cap
    df["hv_lane_interaction"]  = df["LargeVehicles_bin"] * lanes
    df["hv_capacity_effect"]   = road_cap / (1 + df["LargeVehicles_bin"] * 0.25)  # PCE degradation
    df["lanes_sq"]             = lanes ** 2  # diminishing-return nonlinearity
    df["capacity_hv_ratio"]    = df["capacity_proxy"] / (1 + df["hv_lane_interaction"])

    # Compound road × HV categorical
    df["road_hv_type"] = df["RoadType"].astype(str) + "_lv" + df["LargeVehicles_bin"].astype(str)

    # Peak-time × capacity interaction
    if "is_peak_hour" in df.columns:
        df["peak_capacity"] = df["is_peak_hour"] * df["capacity_proxy"]

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# D. WEATHER FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def add_weather_features(
    df: pd.DataFrame,
    global_temp_mean: Optional[float] = None,
    global_temp_std: Optional[float] = None,
) -> pd.DataFrame:
    """
    Encode weather severity and engineer meteorological interaction features.

    Weather ordinal severity:
        Encodes categorical string to 0–7 scale reflecting traffic impact.
        Adverse weather reduces optional travel but may increase ride-hailing.

    temp_deviation = (T − μ_T) / σ_T
        Relative temperature matters more than absolute; extreme cold and heat
        both reduce demand (non-linear, captured by temp_deviation²).

    weather_peak_interaction = weather_severity × is_peak_hour
        Rain at rush hour is far more disruptive than rain at 3 AM.

    weather_road_interaction = weather_severity × capacity_proxy
        High-capacity roads (highways) show larger speed reductions in rain/snow
        than urban arterials (different sensitivity to adverse weather).
    """
    logger.info("Engineering weather features …")

    # ── Severity ordinal ─────────────────────────────────────────────────────
    weather_str = df["Weather"].astype(str).str.lower().str.strip()
    df["weather_severity"] = (
        weather_str.map(C.WEATHER_SEVERITY).fillna(C.WEATHER_SEVERITY_DEFAULT).astype(np.int8)
    )

    # ── Temperature ──────────────────────────────────────────────────────────
    temp = pd.to_numeric(df["Temperature"], errors="coerce")
    df["Temperature"] = temp

    if global_temp_mean is None:
        global_temp_mean = temp.mean()
    if global_temp_std is None:
        global_temp_std  = temp.std() + 1e-8

    df["temp_deviation"]  = (temp - global_temp_mean) / global_temp_std
    df["temp_deviation2"] = df["temp_deviation"] ** 2           # ← non-linear effect
    df["temp_x_severity"] = temp * df["weather_severity"]       # joint weather signal
    df["is_extreme_temp"] = (np.abs(df["temp_deviation"]) > 2).astype(np.int8)

    # ── Interaction features ──────────────────────────────────────────────────
    if "is_peak_hour" in df.columns:
        df["weather_peak_interaction"] = df["weather_severity"] * df["is_peak_hour"]
    if "capacity_proxy" in df.columns:
        df["weather_road_interaction"] = df["weather_severity"] * df["capacity_proxy"]
    if "is_weekend" in df.columns:
        df["weather_weekend"]          = df["weather_severity"] * df["is_weekend"]

    return df, global_temp_mean, global_temp_std


# ═══════════════════════════════════════════════════════════════════════════════
# E. LANDMARK FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def add_landmark_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode landmark presence and build asymmetric time-of-day interactions.

    Landmarks (malls, stations, offices) create strong demand peaks around
    opening/closing times (POI effect, well documented in ride-hailing literature).

    landmark_time_interaction = Landmarks × sin_hour
        Asymmetric: landmarks matter most during peak hours.
    landmark_peak_flag = Landmarks × is_peak_hour
        Direct interaction: "is this a landmark cell at rush hour?"
    """
    logger.info("Engineering landmark features …")

    lm = df["Landmarks"].astype(str).str.lower().str.strip()
    lm_bin = lm.map({"1": 1, "0": 0, "yes": 1, "no": 0, "true": 1, "false": 0})
    if lm_bin.isna().sum() > 0:
        lm_bin = pd.to_numeric(df["Landmarks"], errors="coerce").fillna(0)
    df["Landmarks_bin"] = lm_bin.astype(np.int8)

    if "sin_hour" in df.columns:
        df["landmark_x_sinhour"] = df["Landmarks_bin"] * df["sin_hour"]
        df["landmark_x_coshour"] = df["Landmarks_bin"] * df["cos_hour"]
    if "is_peak_hour" in df.columns:
        df["landmark_peak"]      = df["Landmarks_bin"] * df["is_peak_hour"]
    if "is_weekend" in df.columns:
        df["landmark_weekend"]   = df["Landmarks_bin"] * df["is_weekend"]
    if "capacity_proxy" in df.columns:
        df["landmark_capacity"]  = df["Landmarks_bin"] * df["capacity_proxy"]

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# F. LAG / ROLLING FEATURES  ← single largest R² source
# ═══════════════════════════════════════════════════════════════════════════════

def _build_sort_key(df: pd.DataFrame) -> pd.Series:
    """
    Return a numeric sort key representing temporal order within each geohash.
    Combines day (major) and mins_since_midnight (minor).
    """
    if pd.api.types.is_numeric_dtype(df["day"]):
        day_num = df["day"].astype(float)
    else:
        try:
            day_num = (
                pd.to_datetime(df["day"], infer_datetime_format=True, errors="coerce")
                .dt.dayofyear
                .fillna(df["day"].astype(str).str.extract(r"(\d+)")[0].fillna(0).astype(float))
            )
        except Exception:
            day_num = df["day"].astype(str).str.extract(r"(\d+)")[0].fillna(0).astype(float)

    mins = df["mins_since_midnight"] if "mins_since_midnight" in df.columns else 0
    return day_num * 1440 + mins


def add_lag_rolling_features(
    df: pd.DataFrame,
    lag_steps: list[int] = C.LAG_STEPS,
    roll_windows: list[int] = C.ROLLING_WINDOWS,
    ewma_alphas: list[float] = C.EWMA_ALPHAS,
) -> pd.DataFrame:
    """
    Compute per-geohash demand lag, rolling, and exponential moving-average features.

    CRITICAL: All features are computed on the SORTED combined (train + test) frame
    where test rows have demand=NaN.  Shift() naturally propagates only past
    (train) demand values into future (test) rows — no look-ahead leakage.

    Strategy for test rows at sequence start:
        If a geohash appears only in test, all lags are NaN → imputed with
        geohash-cluster mean or global mean (cold-start fallback).

    lag_k:         demand at t-k intervals (direct past observation)
    roll_mean_w:   rolling mean of last w demand values (smoothed trend)
    roll_std_w:    rolling std of last w demand values (local variability)
    ewma_α:        exponentially weighted MA with decay α (recency-weighted trend)
    expanding_mean: cumulative mean up to current row (long-run geohash baseline)
    """
    logger.info("Computing lag and rolling demand features (largest R² group) …")

    # Sort within each geohash by temporal key
    df = df.copy()
    df["_sort_key"] = _build_sort_key(df)
    df = df.sort_values(["geohash", "_sort_key"]).reset_index(drop=True)

    # Group by geohash and operate on the demand column
    grp = df.groupby("geohash", sort=False)["demand"]

    # ── Lags ─────────────────────────────────────────────────────────────────
    for k in lag_steps:
        df[f"demand_lag_{k}"] = grp.shift(k)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for w in roll_windows:
        rolling = grp.shift(1).groupby(df["geohash"]).transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"demand_roll_mean_{w}"] = rolling

        rolling_std = grp.shift(1).groupby(df["geohash"]).transform(
            lambda x: x.rolling(w, min_periods=2).std().fillna(0)
        )
        df[f"demand_roll_std_{w}"] = rolling_std

    # ── Exponential weighted moving averages ──────────────────────────────────
    for alpha in ewma_alphas:
        alpha_key = str(alpha).replace(".", "")
        df[f"demand_ewma_{alpha_key}"] = grp.shift(1).groupby(df["geohash"]).transform(
            lambda x: x.ewm(alpha=alpha, min_periods=1, adjust=False).mean()
        )

    # ── Expanding mean (no look-ahead since we shift by 1 first) ─────────────
    df["demand_expanding_mean"] = grp.shift(1).groupby(df["geohash"]).transform(
        lambda x: x.expanding(min_periods=1).mean()
    )
    df["demand_expanding_std"] = grp.shift(1).groupby(df["geohash"]).transform(
        lambda x: x.expanding(min_periods=2).std().fillna(0)
    )

    # ── Same-time-yesterday (lag = 96 for 15-min intervals = 24h) ─────────────
    # Already included in lag_steps if 96 ∈ lag_steps; else add explicitly
    if 96 not in lag_steps:
        df["demand_lag_96"] = grp.shift(96)

    # ── Demand trend: diff between lag_1 and lag_4 ────────────────────────────
    if "demand_lag_1" in df.columns and "demand_lag_4" in df.columns:
        df["demand_trend_1_4"] = df["demand_lag_1"] - df["demand_lag_4"]

    # ── Geohash mean demand from the FULL training window ─────────────────────
    # (static baseline; will be computed separately via target encoding)

    df.drop(columns=["_sort_key"], inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# G. TARGET ENCODING  (CV-safe via k-fold)
# ═══════════════════════════════════════════════════════════════════════════════

def target_encode_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
    target: str = "demand",
    n_folds: int = 5,
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute smoothed target encoding using k-fold to avoid target leakage.

    Smoothed mean formula:
        enc(c) = (count(c) × mean(c) + m × global_mean) / (count(c) + m)
    where m = smoothing controls the shrinkage towards the global mean.
    Small groups → shrink to global mean; large groups → keep local mean.

    For test: encode using statistics computed on the full training set.
    For train: use OOF (out-of-fold) encoding to prevent leakage.
    """
    logger.info(f"CV target encoding for columns: {cols}")
    global_mean = train_df[target].mean()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=C.SEED)

    for col in cols:
        oof_enc     = np.zeros(len(train_df))
        col_str     = train_df[col].astype(str)
        test_col_str = test_df[col].astype(str)

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
            stats = (
                train_df.iloc[tr_idx]
                .assign(_col=col_str.iloc[tr_idx])
                .groupby("_col")[target]
                .agg(["mean", "count"])
            )
            smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
                stats["count"] + smoothing
            )
            oof_enc[val_idx] = (
                col_str.iloc[val_idx].map(smoothed).fillna(global_mean).values
            )

        train_df[f"te_{col}"] = oof_enc

        # Test encoding: use full training stats
        full_stats = (
            train_df.assign(_col=col_str)
            .groupby("_col")[target]
            .agg(["mean", "count"])
        )
        full_smoothed = (
            full_stats["count"] * full_stats["mean"] + smoothing * global_mean
        ) / (full_stats["count"] + smoothing)
        test_df[f"te_{col}"] = test_col_str.map(full_smoothed).fillna(global_mean).values

    return train_df, test_df


def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Label-encode high-cardinality categorical columns consistently across train/test.
    Returns the fitted encoders so they can be saved for inspection.
    """
    cat_cols = ["geohash", "RoadType", "Weather", "geohash4", "geohash5",
                "road_hv_type", "geo_cluster"]
    cat_cols = [c for c in cat_cols if c in train_df.columns]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0)
        le.fit(combined)
        train_df[col + "_enc"] = le.transform(train_df[col].astype(str))
        test_df[col  + "_enc"] = le.transform(test_df[col].astype(str))
        encoders[col] = le

    return train_df, test_df, encoders


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def full_feature_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run all feature engineering steps in the correct order.

    Order matters:
        1. Spatial first (needed by weather/road interactions)
        2. Temporal second (flags used in later interaction features)
        3. Road / weather / landmark (static context, may use peak flags)
        4. Lag/rolling LAST (requires demand column and temporal sort key)
        5. Target encoding AFTER lags (uses geohash demand baseline)
        6. Label encoding of categoricals (ensures consistent int codes)

    Returns (train_df, test_df, meta) where meta contains encoders,
    temp_mean, temp_std for later use (e.g., inverse transforms).
    """
    logger.info("=== Starting full feature engineering pipeline ===")
    meta = {}

    # ── 1. Spatial ────────────────────────────────────────────────────────────
    train_df = add_spatial_features(train_df)
    test_df  = add_spatial_features(test_df)
    train_df, test_df = add_geocluster_features(train_df, test_df)

    # ── 2. Temporal ───────────────────────────────────────────────────────────
    train_df = add_temporal_features(train_df)
    test_df  = add_temporal_features(test_df)

    # ── 3. Road ───────────────────────────────────────────────────────────────
    train_df = add_road_features(train_df)
    test_df  = add_road_features(test_df)

    # ── 4. Weather ────────────────────────────────────────────────────────────
    # Fit temp stats on train, apply same to test
    train_df, t_mean, t_std = add_weather_features(train_df)
    test_df,  _,      _     = add_weather_features(test_df, t_mean, t_std)
    meta["temp_mean"] = t_mean
    meta["temp_std"]  = t_std

    # ── 5. Landmark ───────────────────────────────────────────────────────────
    train_df = add_landmark_features(train_df)
    test_df  = add_landmark_features(test_df)

   # ── 6. Lag / rolling (combine train + test for cross-boundary lags) ────────
    test_df["demand"] = np.nan   # sentinel so concat works
    
    # >>> START OF FIX <<<
    # Tag the original datasets and row order before mixing them
    train_df["_is_test"] = 0
    test_df["_is_test"] = 1
    train_df["_orig_order"] = np.arange(len(train_df))
    test_df["_orig_order"] = np.arange(len(test_df))

    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined = add_lag_rolling_features(combined)

    # Split them back safely and restore the exact original row order
    train_df = combined[combined["_is_test"] == 0].sort_values("_orig_order").reset_index(drop=True)
    test_df  = combined[combined["_is_test"] == 1].sort_values("_orig_order").reset_index(drop=True)

    # Clean up the helper columns
    train_df = train_df.drop(columns=["_is_test", "_orig_order"])
    test_df  = test_df.drop(columns=["_is_test", "_orig_order", "demand"])
    # >>> END OF FIX <<<

    # ── 7. Target encoding ────────────────────────────────────────────────────
    te_cols = ["geohash", "RoadType", "Weather", "geo_cluster",
               "geohash4", "geohash5", "road_hv_type"]
    te_cols = [c for c in te_cols if c in train_df.columns]
    train_df, test_df = target_encode_cv(train_df, test_df, te_cols)

    # ── 8. Label encode categoricals ─────────────────────────────────────────
    train_df, test_df, encoders = encode_categoricals(train_df, test_df)
    meta["encoders"] = encoders

    # ── 9. Impute remaining NaNs in lag features ──────────────────────────────
    # For rows at sequence start (no prior demand), fill with geohash mean
    lag_cols = [c for c in train_df.columns if c.startswith("demand_")]
    global_demand_mean = train_df["demand"].mean()

    for df in (train_df, test_df):
        df[lag_cols] = df[lag_cols].fillna(global_demand_mean)

    meta["global_demand_mean"] = global_demand_mean
    meta["n_train"] = len(train_df)

    logger.info(f"Pipeline complete. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df, meta


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return all engineered feature columns (everything except identifiers and target).
    """
    EXCLUDE = {
        "Index", "demand", "geohash", "day", "timestamp",
        "RoadType", "Weather", "LargeVehicles", "Landmarks",
        # raw geohash prefix strings (we use _enc versions)
        "geohash4", "geohash5", "road_hv_type",
    }
    return [c for c in df.columns if c not in EXCLUDE]