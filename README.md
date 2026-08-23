### **Gridlock 2.0: Spatio-Temporal Traffic Demand Forecasting** | *Flipkart GRIDs Challenge*
**Tech Stack:** PyTorch, LightGBM, XGBoost, CatBoost, GCP (NVIDIA A100), Python, Pandas, Scikit-learn

* **Hybrid Architecture:** Engineered a forecasting engine combining a gradient-boosted tree ensemble (LightGBM, XGBoost, CatBoost) with a PyTorch Temporal Convolutional Network (TCN) to predict 15-minute interval traffic demand across 1,249 spatial geohash regions.
* **Deep Spatio-Temporal Modeling:** Designed the TCN with Feature-wise Linear Modulation (FiLM) to dynamically scale and shift temporal feature representations based on static road capacity and Cartesian spatial coordinates.
* **Residual Learning Strategy:** Configured the TCN to learn non-linear temporal residuals from the tree baseline (y_true - y_tree_preds), blending outputs via Ordinary Least Squares (OLS) scaling to capture both macro-trends and micro-volatility.
* **Advanced Feature Engineering:** Developed spatial Cartesian mappings, k-means demand clustering (k=15), cyclic time encodings, weather severity interactions, and leak-free lag/rolling statistics spanning up to 96 steps (24 hours).
* **Cloud Infrastructure & Optimization:** Provisioned a GCP Compute Engine instance with an NVIDIA Tesla A100 GPU (40GB VRAM); optimized throughput using mixed-precision training (bfloat16), TF32 execution, and multi-worker data loaders to complete 3-fold cross-validation and deep residual training in 10.3 minutes.
* **Outcomes:** Processed 119,000+ records across 99 engineered features, utilizing a GroupTimeSeriesSplit validation scheme to achieve an out-of-fold ensemble score of 60.9238 and successfully generated full end-to-end inference for the competition leaderboard.
