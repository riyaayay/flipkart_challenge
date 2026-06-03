"""
run_solution.py — Master Execution Pipeline for Gridlock 2.0.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.isotonic import IsotonicRegression

# Import local modules
import config as C
from feature_engineering import full_feature_pipeline, get_feature_columns
from train_tree_models import train_lgbm_cv, train_xgb_cv, train_catboost_cv, competition_score
from deep_model import train_tcn

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def optimize_ensemble_weights(oof_dict: dict, y_true: np.ndarray) -> np.ndarray:
    logger.info("--- Optimizing Ensemble Weights (NNLS) ---")
    model_names = list(oof_dict.keys())
    X_oof = np.column_stack([oof_dict[m] for m in model_names])
    
    weights, _ = nnls(X_oof, y_true)
    weights = weights / np.sum(weights)  # Normalize to sum to 1
    
    for name, w in zip(model_names, weights):
        logger.info(f"  {name:>10} weight: {w:.4f}")
        
    optimal_oof = X_oof @ weights
    blend_score = competition_score(y_true, optimal_oof)
    logger.info(f"--> Optimized Ensemble OOF Score: {blend_score:.4f}")
    return weights

def post_process_predictions(y_pred: np.ndarray, y_train: np.ndarray, oof_pred: np.ndarray) -> np.ndarray:
    logger.info("--- Applying Post-Processing ---")
    if C.APPLY_ISOTONIC_RECAL:
        logger.info("Fitting Isotonic Regression on OOF residuals...")
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof_pred, y_train)
        y_pred = iso.transform(y_pred)
    
    logger.info(f"Clipping predictions to range [{C.CLIP_LOWER}, {C.CLIP_UPPER}]")
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
    return y_pred

def main():
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("--- 1. Loading Data ---")
    train_df = pd.read_csv(C.TRAIN_PATH)
    test_df  = pd.read_csv(C.TEST_PATH)
    
    test_indices = test_df['Index'].values
    y_train_true = train_df['demand'].values
    
    logger.info("--- 2. Running Feature Engineering Pipeline ---")
    train_df, test_df, fe_meta = full_feature_pipeline(train_df, test_df)
    
    features = get_feature_columns(train_df)
    logger.info(f"Total features engineered: {len(features)}")
    
    X_train = train_df[features]
    X_test  = test_df[features]
    
    # CRITICAL FIX: Because we only have 2 days, we must group by the exact 15-min interval 
    # for the TimeSeriesSplit, otherwise n_folds=3 will fail.
    time_groups = (train_df['day'].astype(float) * 1440) + (train_df['hour'].astype(float) * 60) + train_df['minute'].astype(float)
    time_groups = time_groups.values

    oof_preds, test_preds = {}, {}

    logger.info("--- 3a. Training LightGBM ---")
    oof_lgb, test_lgb, _ = train_lgbm_cv(X_train, train_df['demand'], X_test, time_groups, save_dir=C.MODELS_DIR)
    oof_preds['lgbm'] = oof_lgb
    test_preds['lgbm'] = test_lgb

    logger.info("--- 3b. Training XGBoost ---")
    oof_xgb, test_xgb, _ = train_xgb_cv(X_train, train_df['demand'], X_test, time_groups, save_dir=C.MODELS_DIR)
    oof_preds['xgb'] = oof_xgb
    test_preds['xgb'] = test_xgb

    logger.info("--- 3c. Training CatBoost ---")
    cat_features = [c for c in features if c.endswith("_enc") or c == "geo_cluster"]
    oof_cat, test_cat, _ = train_catboost_cv(X_train, train_df['demand'], X_test, time_groups, cat_feature_names=cat_features, save_dir=C.MODELS_DIR)
    oof_preds['catboost'] = oof_cat
    test_preds['catboost'] = test_cat

    logger.info("--- 4. Training PyTorch TCN + FiLM ---")
    # TCN internally uses GroupTimeSeriesSplit. We need to overwrite the `_make_day_groups` logic implicitly 
    # by passing our time_groups if we refactored it, or let it rely on the fact that TCN handles sequences based on row index.
    # To keep it simple, train_df['day'] can be temporarily replaced by time_groups for the deep model CV
    train_df_deep = train_df.copy()
    train_df_deep['day'] = time_groups 
    
    oof_tcn, test_tcn = train_tcn(train_df_deep, test_df, save_dir=C.MODELS_DIR)
    oof_preds['tcn'] = oof_tcn
    test_preds['tcn'] = test_tcn

    logger.info("--- 5. Ensembling via NNLS ---")
    if C.ENSEMBLE_USE_NNLS:
        weights = optimize_ensemble_weights(oof_preds, y_train_true)
        model_names = list(oof_preds.keys())
    else:
        model_names = list(C.ENSEMBLE_WEIGHTS_DEFAULT.keys())
        weights = [C.ENSEMBLE_WEIGHTS_DEFAULT[m] for m in model_names]

    final_oof  = np.zeros(len(train_df))
    final_test = np.zeros(len(test_df))
    for i, name in enumerate(model_names):
        final_oof  += oof_preds[name] * weights[i]
        final_test += test_preds[name] * weights[i]

    logger.info("--- 6. Post-Processing ---")
    final_test = post_process_predictions(final_test, y_train_true, final_oof)

    logger.info("--- 7. Generating Final Submission ---")
    sub_df = pd.DataFrame({'Index': test_indices, 'demand': final_test})
    sub_df.to_csv(C.SUBMISSION_PATH, index=False)
    logger.info(f"Success! Submission saved to {C.SUBMISSION_PATH}")

if __name__ == "__main__":
    main()