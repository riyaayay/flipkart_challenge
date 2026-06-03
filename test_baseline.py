import logging
import pandas as pd
import config as C
from feature_engineering import full_feature_pipeline, get_feature_columns
from train_tree_models import train_lgbm_cv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

print("1. Loading Data...")
train_df = pd.read_csv(C.TRAIN_PATH)
test_df  = pd.read_csv(C.TEST_PATH)

# Using only the first 20,000 rows for a fast local test!
train_df = train_df.iloc[:20000].reset_index(drop=True)

print("2. Running Feature Engineering...")
train_df, test_df, meta = full_feature_pipeline(train_df, test_df)

features = get_feature_columns(train_df)
print(f"Features created: {len(features)}")

day_groups = train_df['day'].astype(str).str.extract(r"(\d+)")[0].fillna(0).astype(float).values

print("3. Training LightGBM Baseline...")
# Setting epochs low just to verify the pipeline works
C.LGBM_PARAMS['n_estimators'] = 100 

oof_preds, test_preds, models = train_lgbm_cv(
    train_df[features], 
    train_df['demand'], 
    test_df[features], 
    day_groups
)

print("\n✅ Baseline Test Complete! The pipeline is solid.")