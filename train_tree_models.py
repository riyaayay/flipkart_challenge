"""
train_tree_models.py — LightGBM / XGBoost / CatBoost training with time-series CV.

Key design decisions:
  • GroupTimeSeriesSplit (NOT random k-fold) — prevents lag-feature leakage.
  • All three tree models run; OOF predictions saved for ensemble weight search.
  • SHAP analysis reveals which features drive demand and guides further FE.
  • R² is the primary reported metric (mirrors the competition score × 100).
"""

import gc
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import config as C

logger = logging.getLogger(__name__)

# ─── Optional imports (graceful fallback if not installed) ────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("LightGBM not installed — skipping LGBM model.")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not installed — skipping XGB model.")

try:
    from catboost import CatBoostRegressor, Pool
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    logger.warning("CatBoost not installed — skipping CatBoost model.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ═══════════════════════════════════════════════════════════════════════════════
# Validation Strategy
# ═══════════════════════════════════════════════════════════════════════════════

class GroupTimeSeriesSplit:
    """
    Expanding-window cross-validator that respects temporal ordering.

    Unlike random k-fold, this ensures the validation set always comes AFTER
    the training set in time, preventing future-to-past information leakage
    in lag and rolling features.

    Typical use:
        groups = day_numeric_column (one integer per row = which day it belongs to)
        Each fold: train on days [0 .. cutoff], validate on [cutoff+1 .. cutoff+fold_width]
    """

    def __init__(self, n_splits: int = 5, gap: int = 0):
        self.n_splits = n_splits
        self.gap      = gap  # number of days to drop between train and val

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Must provide `groups` (day indices) for temporal CV.")

        unique_days = np.sort(np.unique(groups))
        n_days      = len(unique_days)
        fold_width  = max(1, n_days // (self.n_splits + 1))

        for fold in range(self.n_splits):
            cutoff_idx  = (fold + 1) * fold_width
            val_start   = cutoff_idx + self.gap
            val_end     = min(val_start + fold_width, n_days)

            if val_start >= n_days:
                break

            train_days = set(unique_days[:cutoff_idx])
            val_days   = set(unique_days[val_start:val_end])

            groups_s   = pd.Series(groups)
            train_idx  = np.where(groups_s.isin(train_days))[0]
            val_idx    = np.where(groups_s.isin(val_days))[0]

            if len(val_idx) == 0:
                continue

            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clipped R² matching the competition metric: max(0, R²)."""
    return max(0.0, r2_score(y_true, y_pred))


def competition_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact competition score: max(0, 100 × R²)."""
    return max(0.0, 100.0 * r2_score(y_true, y_pred))


# ═══════════════════════════════════════════════════════════════════════════════
# LightGBM
# ═══════════════════════════════════════════════════════════════════════════════

def train_lgbm_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    day_groups: np.ndarray,
    params: dict = C.LGBM_PARAMS,
    n_folds: int = C.N_FOLDS,
    use_log_target: bool = C.APPLY_LOG_TRANSFORM,
    save_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Train LightGBM with GroupTimeSeriesSplit CV.

    Returns:
        oof_preds  : Out-of-fold predictions on train set (for ensemble weight search)
        test_preds : Mean of fold test predictions
        models     : List of trained LGB booster objects (one per fold)

    Log-target transform:
        If use_log_target=True, fit on log1p(y), predict then expm1 back.
        Stabilises variance across the demand range; prevents high-demand rows
        from dominating the RMSE loss.
    """
    assert HAS_LGB, "LightGBM not installed."

    cv = GroupTimeSeriesSplit(n_splits=n_folds, gap=C.CV_GAP_DAYS)
    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    models     = []

    y_fit = np.log1p(y_train.values) if use_log_target else y_train.values

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_fit, groups=day_groups)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_fit[tr_idx],        y_fit[val_idx]

        lgb_train = lgb.Dataset(X_tr,  y_tr)
        lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train)

        # Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=params.get("early_stopping_rounds", 150),
                               verbose=False),
            lgb.log_evaluation(period=500),
        ]

        p = {k: v for k, v in params.items()
             if k not in ("n_estimators", "early_stopping_rounds")}

        model = lgb.train(
            p,
            lgb_train,
            num_boost_round=params.get("n_estimators", 5000),
            valid_sets=[lgb_val],
            callbacks=callbacks,
        )

        fold_val_pred  = model.predict(X_val)
        fold_test_pred = model.predict(X_test)

        if use_log_target:
            fold_val_pred  = np.expm1(fold_val_pred)
            fold_test_pred = np.expm1(fold_test_pred)

        oof_preds[val_idx] = fold_val_pred
        test_preds        += fold_test_pred / n_folds
        models.append(model)

        y_val_actual = np.expm1(y_val) if use_log_target else y_val
        score = competition_score(y_val_actual, fold_val_pred)
        logger.info(f"  [LGBM] Fold {fold+1}/{n_folds}  score={score:.2f}  "
                    f"best_iter={model.best_iteration}")

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(save_dir / f"lgbm_fold{fold}.txt"))

    overall = competition_score(y_train.values, oof_preds)
    logger.info(f"[LGBM] OOF competition score: {overall:.4f}")

    gc.collect()
    return oof_preds, test_preds, models


# ═══════════════════════════════════════════════════════════════════════════════
# XGBoost
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgb_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    day_groups: np.ndarray,
    params: dict = C.XGB_PARAMS,
    n_folds: int = C.N_FOLDS,
    use_log_target: bool = C.APPLY_LOG_TRANSFORM,
    save_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Train XGBoost with GroupTimeSeriesSplit CV. Same API as train_lgbm_cv."""
    assert HAS_XGB, "XGBoost not installed."

    cv = GroupTimeSeriesSplit(n_splits=n_folds, gap=C.CV_GAP_DAYS)
    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    models     = []

    y_fit = np.log1p(y_train.values) if use_log_target else y_train.values

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_fit, groups=day_groups)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_fit[tr_idx],        y_fit[val_idx]

        p = {k: v for k, v in params.items()
             if k not in ("n_estimators", "early_stopping_rounds", "eval_metric")}

        model = xgb.XGBRegressor(
            **p,
            n_estimators=params.get("n_estimators", 5000),
            early_stopping_rounds=params.get("early_stopping_rounds", 150),
            eval_metric=params.get("eval_metric", "rmse"),
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        fold_val_pred  = model.predict(X_val)
        fold_test_pred = model.predict(X_test)

        if use_log_target:
            fold_val_pred  = np.expm1(fold_val_pred)
            fold_test_pred = np.expm1(fold_test_pred)

        oof_preds[val_idx] = fold_val_pred
        test_preds        += fold_test_pred / n_folds
        models.append(model)

        y_val_actual = np.expm1(y_val) if use_log_target else y_val
        score = competition_score(y_val_actual, fold_val_pred)
        logger.info(f"  [XGB] Fold {fold+1}/{n_folds}  score={score:.2f}  "
                    f"best_iter={model.best_iteration}")

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(save_dir / f"xgb_fold{fold}.json"))

    overall = competition_score(y_train.values, oof_preds)
    logger.info(f"[XGB] OOF competition score: {overall:.4f}")

    gc.collect()
    return oof_preds, test_preds, models


# ═══════════════════════════════════════════════════════════════════════════════
# CatBoost
# ═══════════════════════════════════════════════════════════════════════════════

def train_catboost_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    day_groups: np.ndarray,
    cat_feature_names: list[str],
    params: dict = C.CATBOOST_PARAMS,
    n_folds: int = C.N_FOLDS,
    use_log_target: bool = C.APPLY_LOG_TRANSFORM,
    save_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Train CatBoost with GroupTimeSeriesSplit CV.

    cat_feature_names : list of column names that are integer-encoded categoricals.
    CatBoost natively handles categorical columns (no target encoding leakage risk),
    but we still pass the pre-encoded integers for API consistency.
    """
    assert HAS_CAT, "CatBoost not installed."

    # CatBoost wants column indices, not names
    cat_indices = [X_train.columns.tolist().index(c)
                   for c in cat_feature_names if c in X_train.columns]

    cv = GroupTimeSeriesSplit(n_splits=n_folds, gap=C.CV_GAP_DAYS)
    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    models     = []

    y_fit = np.log1p(y_train.values) if use_log_target else y_train.values

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_fit, groups=day_groups)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_fit[tr_idx],        y_fit[val_idx]

        train_pool = Pool(X_tr,  y_tr,  cat_features=cat_indices)
        val_pool   = Pool(X_val, y_val, cat_features=cat_indices)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        fold_val_pred  = model.predict(X_val)
        fold_test_pred = model.predict(X_test)

        if use_log_target:
            fold_val_pred  = np.expm1(fold_val_pred)
            fold_test_pred = np.expm1(fold_test_pred)

        oof_preds[val_idx] = fold_val_pred
        test_preds        += fold_test_pred / n_folds
        models.append(model)

        y_val_actual = np.expm1(y_val) if use_log_target else y_val
        score = competition_score(y_val_actual, fold_val_pred)
        logger.info(f"  [CAT] Fold {fold+1}/{n_folds}  score={score:.2f}  "
                    f"best_iter={model.best_iteration_}")

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(save_dir / f"catboost_fold{fold}.cbm"))

    overall = competition_score(y_train.values, oof_preds)
    logger.info(f"[CatBoost] OOF competition score: {overall:.4f}")

    gc.collect()
    return oof_preds, test_preds, models


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_shap(
    model,
    X: pd.DataFrame,
    model_type: str = "lgbm",
    n_samples: int = 5000,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute and display SHAP feature importances.

    Why SHAP (not gain/split importance)?
        SHAP values are additive and model-agnostic; they show the marginal
        contribution of each feature to each prediction.  This reveals:
        - Which features truly drive demand (not just split the tree)
        - Interaction effects (e.g., lag_1 importance under different weather)
        - Features to prune (near-zero SHAP across all samples)

    Returns a DataFrame of mean absolute SHAP value per feature (importance).
    """
    if not HAS_SHAP:
        logger.warning("shap package not installed; skipping SHAP analysis.")
        return pd.DataFrame()

    sample_idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sample   = X.iloc[sample_idx]

    if model_type == "lgbm":
        explainer = shap.TreeExplainer(model)
    elif model_type in ("xgb", "catboost"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model)

    shap_values = explainer.shap_values(X_sample)
    mean_abs    = np.abs(shap_values).mean(axis=0)

    importance_df = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("\n=== Top 30 features by mean |SHAP| ===")
    logger.info(importance_df.head(30).to_string())

    if save_path:
        importance_df.to_csv(save_path, index=False)
        logger.info(f"SHAP importances saved to {save_path}")

    return importance_df


# ═══════════════════════════════════════════════════════════════════════════════
# EDA validation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def validate_key_correlations(train_df: pd.DataFrame, target: str = "demand") -> None:
    """
    Print the key mathematical relationships that should hold if FE is correct.
    Compare against expected R² thresholds from the master prompt.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OrdinalEncoder

    logger.info("\n=== Validating key variance relationships ===")

    # 1. Time explains variance
    time_feats = [c for c in ["sin_hour", "cos_hour", "sin_dow", "cos_dow"]
                  if c in train_df.columns]
    if time_feats:
        X_ = train_df[time_feats].fillna(0)
        y_ = train_df[target].values
        r2 = r2_score(y_, LinearRegression().fit(X_, y_).predict(X_))
        flag = "✓" if r2 > 0.3 else "✗ (expected > 0.30)"
        logger.info(f"  R²(demand ~ time_cyclic)  = {r2:.4f}  {flag}")

    # 2. Lag_1 dominates
    if "demand_lag_1" in train_df.columns:
        notna = train_df["demand_lag_1"].notna() & train_df[target].notna()
        X_ = train_df.loc[notna, "demand_lag_1"].values.reshape(-1, 1)
        y_ = train_df.loc[notna, target].values
        r2 = r2_score(y_, LinearRegression().fit(X_, y_).predict(X_))
        flag = "✓" if r2 > 0.5 else "✗ (expected > 0.50)"
        logger.info(f"  R²(demand ~ lag_1)         = {r2:.4f}  {flag}")

    # 3. Geohash residual variance
    if "te_geohash" in train_df.columns:
        X_ = train_df[["te_geohash"]].fillna(0)
        y_ = train_df[target].values
        r2 = r2_score(y_, LinearRegression().fit(X_, y_).predict(X_))
        logger.info(f"  R²(demand ~ geohash_te)   = {r2:.4f}  (expected > 0.20 residual)")

    logger.info("=== Validation complete ===\n")