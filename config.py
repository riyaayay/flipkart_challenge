"""
config.py — Global configuration for Gridlock 2.0 Traffic Demand Solution.
Architecture: LightGBM + XGBoost + CatBoost + TCN(FiLM) ensemble on A100.

DATA-VERIFIED VALUES (from EDA on actual train.csv / test.csv):
  - demand          : already normalised [0, 1], skewness = 3.73
  - days            : train = {48, 49}, test = {49}  (only 2 training days)
  - timestamp       : "H:M" format, 15-min intervals (96 slots per full day)
  - RoadType        : Highway | Street | Residential | NaN (3 values only)
  - Weather         : Sunny | Rainy | Foggy | Snowy | NaN  (4 values only)
  - LargeVehicles   : Allowed | Not Allowed
  - Landmarks       : Yes | No
  - Temperature     : float, [−15, 48] °C, 2 495 NaN in train / 1 349 in test
  - train geohashes : 1 249 unique
  - test geohashes  : 1 190 unique (10 cold-start not in train)
  - test is a strict temporal continuation of train (2:15 follows train's 2:00)
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Paths  (data lives in dataset/ subdirectory after zip extraction)
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

TRAIN_PATH      = DATA_DIR / "train.csv"
TEST_PATH       = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════
SEED = 42

# ═══════════════════════════════════════════════════════════════════════════════
# Spatial / Geohash
# ═══════════════════════════════════════════════════════════════════════════════
EARTH_RADIUS_KM       = 6371.0
N_GEO_CLUSTERS        = 15          # k-means over (lat, lon); 1 249 geohashes → 15 zones
GEOHASH_PREFIX_LEVELS = [4, 5]      # coarser spatial granularities

# ═══════════════════════════════════════════════════════════════════════════════
# Temporal
# Intervals = 15 min.  lag_96 = exactly 24 h ago (same time day 48 → day 49).
# With only 2 training days, lags > 96 will produce mostly NaN — skip them.
# ═══════════════════════════════════════════════════════════════════════════════
LAG_STEPS       = [1, 2, 3, 4, 6, 8, 12, 24, 48, 96]   # 96 = same time yesterday
ROLLING_WINDOWS = [3, 6, 12, 24]
EWMA_ALPHAS     = [0.1, 0.3, 0.6]

# ═══════════════════════════════════════════════════════════════════════════════
# Weather severity  —  VERIFIED against actual Weather column values
# Sunny=0, Foggy=1, Rainy=2, Snowy=3  (traffic-impact ordering)
# ═══════════════════════════════════════════════════════════════════════════════
WEATHER_SEVERITY: dict[str, int] = {
    # ── Present in dataset ───────────────────────────────────────────────────
    "sunny":   0,
    "foggy":   1,   # reduces visibility; major impact on speed
    "rainy":   2,
    "snowy":   3,   # worst driving conditions
    # ── Common aliases (keep for robustness) ─────────────────────────────────
    "clear":   0, "fair": 0, "fine": 0,
    "fog":     1, "mist": 1, "haze": 1,
    "rain":    2, "rainy": 2, "drizzle": 2, "showers": 2,
    "snow":    3, "blizzard": 3, "sleet": 3,
}
WEATHER_SEVERITY_DEFAULT = 0   # treat unknown as clear

# ═══════════════════════════════════════════════════════════════════════════════
# Road capacity weights  —  VERIFIED against actual RoadType values
# ═══════════════════════════════════════════════════════════════════════════════
ROAD_CAPACITY_MAP: dict[str, float] = {
    "highway":     4.0,   # confirmed in dataset
    "street":      1.5,   # confirmed in dataset (urban arterial-like)
    "residential": 0.8,   # confirmed in dataset
    # extra aliases for robustness
    "motorway":    4.0, "expressway": 3.8,
    "trunk":       3.5, "primary": 3.0, "arterial": 2.5,
    "secondary":   2.0, "local": 1.0, "service": 0.5,
}
ROAD_CAPACITY_DEFAULT = 1.5   # fallback for NaN RoadType

# ═══════════════════════════════════════════════════════════════════════════════
# Cross-validation
# CRITICAL: only 2 training days (48 & 49).
# We split on the combined temporal index (day * 96 + interval_idx), giving ~105
# unique time points.  Use 3 folds so each val fold has ~35 time points ≈ 8-9 h.
# ═══════════════════════════════════════════════════════════════════════════════
N_FOLDS     = 3
CV_GAP_DAYS = 0

# ═══════════════════════════════════════════════════════════════════════════════
# LightGBM  (GPU via device_type='gpu')
# ═══════════════════════════════════════════════════════════════════════════════
LGBM_PARAMS: dict = {
    "objective":               "regression",
    "metric":                  "rmse",
    "boosting_type":           "gbdt",
    "learning_rate":           0.02,
    "num_leaves":              255,
    "max_depth":               -1,
    "min_child_samples":       10,
    "feature_fraction":        0.75,
    "bagging_fraction":        0.8,
    "bagging_freq":            1,
    "reg_alpha":               0.05,
    "reg_lambda":              0.1,
    "n_estimators":            5000,
    "early_stopping_rounds":   200,
    "verbose":                 -1,
    "n_jobs":                  -1,
    "device_type":             "cpu",
    "gpu_platform_id":         0,
    "gpu_device_id":           0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# XGBoost
# ═══════════════════════════════════════════════════════════════════════════════
XGB_PARAMS: dict = {
    "objective":              "reg:squarederror",
    "n_estimators":           5000,
    "learning_rate":          0.02,
    "max_depth":              8,
    "min_child_weight":       5,
    "subsample":              0.8,
    "colsample_bytree":       0.75,
    "reg_alpha":              0.05,
    "reg_lambda":             0.1,
    "tree_method":            "hist",
    "device":                 "cuda",
    "early_stopping_rounds":  200,
    "eval_metric":            "rmse",
    "verbosity":              0,
    "seed":                   SEED,
}

# ═══════════════════════════════════════════════════════════════════════════════
# CatBoost
# ═══════════════════════════════════════════════════════════════════════════════
CATBOOST_PARAMS: dict = {
    "iterations":          5000,
    "learning_rate":       0.02,
    "depth":               8,
    "l2_leaf_reg":         3.0,
    "task_type":           "GPU",
    "eval_metric":         "RMSE",
    "od_type":             "Iter",
    "od_wait":             200,
    "verbose":             500,
    "random_seed":         SEED,
    "border_count":        254,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Deep model  (TCN + FiLM, PyTorch, A100)
# ═══════════════════════════════════════════════════════════════════════════════
DEEP_PARAMS: dict = {
    "seq_len":         12,           # 12 × 15 min = 3 h lookback
    "tcn_channels":   [64, 128, 256, 256, 128, 64],
    "kernel_size":     3,
    "dilations":      [1, 2, 4, 8, 16, 32],
    "dropout":         0.15,
    "film_hidden":     256,
    "head_dims":       [256, 128, 64],
    "lr":              3e-4,
    "weight_decay":    1e-5,
    "batch_size":      4096,
    "epochs":          120,
    "patience":        20,
    "grad_clip":       1.0,
    "huber_delta":     1.0,
    "scheduler":       "cosine",
    "mixed_precision": True,
    "num_workers":     4,
    "pin_memory":      True,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble
# ═══════════════════════════════════════════════════════════════════════════════
ENSEMBLE_USE_NNLS = True
ENSEMBLE_WEIGHTS_DEFAULT: dict[str, float] = {
    "lgbm":     0.40,
    "xgb":      0.25,
    "catboost": 0.20,
    "tcn":      0.15,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Post-processing
# demand already in [0,1] but right-skewed (skew=3.73) → log1p still stabilises
# ═══════════════════════════════════════════════════════════════════════════════
APPLY_LOG_TRANSFORM  = True    # log1p(demand) during training; expm1 on predict
CLIP_LOWER           = 0.0
CLIP_UPPER           = 1.0     # demand is bounded [0, 1] by normalisation
APPLY_ISOTONIC_RECAL = True