"""
deep_model.py — TCN + FiLM conditioning for Gridlock 2.0 traffic demand.

Architecture overview:
    ┌──────────────────────────────────────────────────────────────────┐
    │  Static context (per geohash × road):                           │
    │   [RoadType_emb, NumLanes, LargeVehicles, Landmarks,            │
    │    lat, lon, geo_cluster_emb]                                   │
    │         │                                                        │
    │         ▼ Linear → ReLU → Linear → (γ, β)  [FiLM params]       │
    │                                                                  │
    │  Dynamic sequence (T × d_dyn per geohash):                      │
    │   [demand_t, sin_hour, cos_hour, weather_sev, temp_dev]          │
    │         │                                                        │
    │         ▼ TCN(dilations=[1,2,4,8,16,32], kernel=3)              │
    │         │                                                        │
    │  FiLM:  h = γ ⊙ tcn_out + β  ← Road/space modulates dynamics   │
    │         │                                                        │
    │         ▼ MLP head → scalar demand prediction                   │
    └──────────────────────────────────────────────────────────────────┘

Why TCN over LSTM?
  - Dilated causal convolutions: receptive field = kernel × 2^(n_layers).
    With dilations [1,2,4,8,16,32] and kernel=3: RF = 3 × 63 = 189 timesteps.
  - Fully parallelisable over the time dimension on A100 (vs sequential LSTM).
  - No vanishing-gradient problem for long sequences.
  - Stable gradients via residual connections in each TemporalBlock.

Why FiLM conditioning?
  - Road type and lane count modulate HOW demand evolves, not just what it is.
  - FiLM learns a per-context γ (scale) and β (shift) that rescales the TCN's
    internal activations:  h = γ(static) ⊙ TCN(dynamic) + β(static).
  - This is equivalent to giving each road-type / lane configuration its own
    learned temporal dynamics without duplicating model weights.
"""

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import config as C

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Building Blocks
# ═══════════════════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    """
    1-D causal dilated convolution.

    Causal: output at time t only depends on inputs at t, t-1, t-2, …
    Dilation d: effective receptive field = kernel_size × d per layer.
    Left-padding = (kernel_size − 1) × d ensures output length == input length.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=dilation,
        )
        self.norm    = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        x_pad = F.pad(x, (self.pad, 0))       # left-pad only → causal
        out   = self.conv(x_pad)               # (B, C_out, T)
        # LayerNorm over the channel dim: transpose → norm → transpose
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.relu(out))


class TemporalBlock(nn.Module):
    """
    TCN residual block: two causal convolutions + residual skip connection.

    Residual connection lets gradients bypass the block during backprop,
    preventing vanishing gradients over long dilations.
    A 1×1 conv adapts the channel dimension for the residual if necessary.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels,  out_channels, kernel_size, dilation, dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, dropout)

        # Residual adapter (only if channel dims differ)
        self.residual_adapt = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.residual_adapt is None else self.residual_adapt(x)
        out = self.conv2(self.conv1(x))
        return F.relu(out + residual, inplace=True)


class TCN(nn.Module):
    """
    Temporal Convolutional Network: stack of TemporalBlocks with exponential dilations.

    input  : (B, d_in, T)   — batch × feature_dim × seq_len
    output : (B, d_out, T)  — features at every time step (for FiLM conditioning)
    """
    def __init__(self, d_in: int, channels: list[int],
                 kernel_size: int, dilations: list[int], dropout: float):
        super().__init__()
        assert len(channels) == len(dilations), \
            "Must provide one channel count per dilation level."

        layers = []
        in_ch = d_in
        for out_ch, dil in zip(channels, dilations):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dil, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)          # (B, C_last, T)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.

    γ, β = MLP(static_context)
    h    = γ ⊙ tcn_output + β

    γ rescales the per-channel activations (gain);
    β shifts them (bias).  Together they allow the static road/geohash context
    to modulate every internal timestep of the TCN without adding parameters
    per timestep.
    """
    def __init__(self, static_dim: int, n_channels: int, hidden: int):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(static_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_channels),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(static_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_channels),
        )
        # Initialise close to identity so initial conditioning is neutral
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, tcn_out: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        # tcn_out: (B, C, T);  static: (B, d_static)
        gamma = self.gamma_net(static).unsqueeze(-1)   # (B, C, 1) → broadcast over T
        beta  = self.beta_net(static).unsqueeze(-1)
        return gamma * tcn_out + beta


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Full Model
# ═══════════════════════════════════════════════════════════════════════════════

class GridlockTCNModel(nn.Module):
    """
    Complete TCN + FiLM model for Gridlock 2.0 traffic demand prediction.

    Two input streams:
      - static_x  : per-geohash×road context (shape: B × d_static)
      - dynamic_x : temporal sequence of demand + exogenous features
                    (shape: B × d_dynamic × seq_len)

    The static context is processed once, producing (γ, β) that modulate
    every layer of the TCN operating on the dynamic sequence.
    The last-timestep output is fed into a small MLP head → scalar demand.
    """

    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        tcn_channels: list[int],
        kernel_size: int,
        dilations: list[int],
        dropout: float,
        film_hidden: int,
        head_dims: list[int],
    ):
        super().__init__()
        # ── TCN on dynamic sequence ─────────────────────────────────────────
        self.tcn = TCN(dynamic_dim, tcn_channels, kernel_size, dilations, dropout)
        tcn_out_dim = self.tcn.out_channels

        # ── FiLM conditioning ────────────────────────────────────────────────
        self.film = FiLMLayer(static_dim, tcn_out_dim, film_hidden)

        # ── Prediction head (MLP on last-step features) ──────────────────────
        head_layers = []
        in_dim = tcn_out_dim + static_dim   # concat static into head
        for out_dim in head_dims:
            head_layers += [nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True),
                            nn.Dropout(dropout)]
            in_dim = out_dim
        head_layers.append(nn.Linear(in_dim, 1))   # → scalar demand
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        static_x: torch.Tensor,   # (B, d_static)
        dynamic_x: torch.Tensor,  # (B, d_dynamic, T)
    ) -> torch.Tensor:            # (B,)
        tcn_out    = self.tcn(dynamic_x)           # (B, C, T)
        film_out   = self.film(tcn_out, static_x)  # (B, C, T) — modulated
        last_step  = film_out[:, :, -1]            # (B, C) — last timestep
        combined   = torch.cat([last_step, static_x], dim=-1)   # (B, C + d_static)
        return self.head(combined).squeeze(-1)     # (B,)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Dataset & Sequence Builder
# ═══════════════════════════════════════════════════════════════════════════════

# Static features: road + spatial context
STATIC_FEATURES = [
    "road_capacity_weight", "NumberofLanes", "LargeVehicles_bin", "Landmarks_bin",
    "lat", "lon", "x", "y", "z", "dist_to_centroid",
    "geo_cluster",          # will be embedded / or used as float
]

# Dynamic features: per-timestep demand + exogenous
DYNAMIC_FEATURES = [
    "demand",               # the actual target at previous steps
    "sin_hour", "cos_hour",
    "sin_dow",  "cos_dow",
    "weather_severity",     # ordinal 0–7
    "temp_deviation",       # (T − μ) / σ
    "is_peak_hour",
    "is_weekend",
]


class TrafficSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        static_cols: list[str],
        dynamic_cols: list[str],
        is_train: bool = True,
        global_demand_mean: float = 0.0,
    ):
        self.seq_len   = seq_len
        self.is_train  = is_train

        # >>> ADD THIS LINE: Save original index before sorting <<<
        df["_orig_idx"] = np.arange(len(df))

        # Sort by geohash + time before building index
        df = df.sort_values(["geohash", "day", "mins_since_midnight"]).reset_index(drop=True)
        
        # >>> ADD THIS LINE: Create the unsort mapping <<<
        self.inverse_order = np.argsort(df["_orig_idx"].values)

        # Impute missing static / dynamic columns
        for col in static_cols + dynamic_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
                global_demand_mean if col == "demand" else 0.0
            )

        self.static_data  = df[static_cols].values.astype(np.float32)   # (N, d_static)
        self.dynamic_data = df[dynamic_cols].values.astype(np.float32)  # (N, d_dyn)
        self.targets      = df["demand"].values.astype(np.float32) if is_train else None
        self.geohashes    = df["geohash"].values
        self.n            = len(df)

        # For each row, find the start of its geohash block
        gh_change    = np.concatenate([[True], self.geohashes[1:] != self.geohashes[:-1]])
        self.gh_starts = np.where(gh_change)[0]   # first row index for each geohash

        # Map each row → its geohash block start
        self.row_to_block_start = np.zeros(self.n, dtype=np.int64)
        for start_idx, end_idx in zip(self.gh_starts, np.append(self.gh_starts[1:], self.n)):
            self.row_to_block_start[start_idx:end_idx] = start_idx

        # Geohash-level mean demand for left-padding (cold-start)
        self.gh_mean = np.zeros(self.n, dtype=np.float32)
        dyn_demand_col = dynamic_cols.index("demand") if "demand" in dynamic_cols else 0
        for start_idx, end_idx in zip(self.gh_starts, np.append(self.gh_starts[1:], self.n)):
            m = np.nanmean(self.dynamic_data[start_idx:end_idx, dyn_demand_col])
            self.gh_mean[start_idx:end_idx] = m if np.isfinite(m) else global_demand_mean

        self.dyn_demand_col  = dyn_demand_col
        self.static_cols     = static_cols
        self.dynamic_cols    = dynamic_cols

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        block_start = self.row_to_block_start[idx]

        # Gather up to seq_len prior rows within the same geohash
        start      = max(block_start, idx - self.seq_len)
        seq_dyn    = self.dynamic_data[start:idx, :]   # (≤seq_len, d_dyn)
        pad_len    = self.seq_len - len(seq_dyn)

        if pad_len > 0:
            pad = np.full((pad_len, seq_dyn.shape[1]), 0.0, dtype=np.float32)
            pad[:, self.dyn_demand_col] = self.gh_mean[idx]
            seq_dyn = np.vstack([pad, seq_dyn])        # (seq_len, d_dyn)

        seq_dyn   = seq_dyn.T                           # (d_dyn, seq_len) for TCN
        static_x  = self.static_data[idx]              # (d_static,)

        if self.is_train:
            target = self.targets[idx]
            return (torch.tensor(static_x), torch.tensor(seq_dyn), torch.tensor(target))
        else:
            return (torch.tensor(static_x), torch.tensor(seq_dyn))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    Huber loss: MSE for |r| ≤ δ, MAE-like for |r| > δ.
    Robust to demand spikes while still penalising large errors.
    L(r) = 0.5 r²                if |r| ≤ δ
           δ(|r| − 0.5δ)         otherwise
    """
    return F.huber_loss(pred, target, delta=delta, reduction="mean")


def _make_day_groups(df: pd.DataFrame) -> np.ndarray:
    """Numeric day index for GroupTimeSeriesSplit inside deep model CV."""
    if pd.api.types.is_numeric_dtype(df["day"]):
        return df["day"].astype(float).values
    day_enc = pd.factorize(df["day"])[0].astype(float)
    return day_enc


def train_tcn(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict = C.DEEP_PARAMS,
    use_log_target: bool = C.APPLY_LOG_TRANSFORM,
    save_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train TCN + FiLM model on train_df, predict on test_df.

    Returns:
        oof_preds  : Out-of-fold predictions aligned to train_df row order
        test_preds : Predictions for test_df
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[TCN] Training on device: {device}")
    if device.type == "cuda":
        logger.info(f"[TCN] GPU: {torch.cuda.get_device_name(0)} "
                    f"| Mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Static / dynamic column availability ─────────────────────────────────
    static_cols  = [c for c in STATIC_FEATURES  if c in train_df.columns]
    dynamic_cols = [c for c in DYNAMIC_FEATURES if c in train_df.columns]
    if "demand" not in dynamic_cols:
        dynamic_cols.insert(0, "demand")

    d_static  = len(static_cols)
    d_dynamic = len(dynamic_cols)
    logger.info(f"[TCN] d_static={d_static}, d_dynamic={d_dynamic}")

    global_demand_mean = train_df["demand"].mean()

    # ── Cross-validation loop ─────────────────────────────────────────────────
    from train_tree_models import GroupTimeSeriesSplit, competition_score
    cv        = GroupTimeSeriesSplit(n_splits=C.N_FOLDS, gap=C.CV_GAP_DAYS)
    day_groups = _make_day_groups(train_df)
    oof_preds  = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    for fold, (tr_idx, val_idx) in enumerate(
        cv.split(train_df, groups=day_groups)
    ):
        logger.info(f"[TCN] ── Fold {fold+1}/{C.N_FOLDS} ──")

        tr_df  = train_df.iloc[tr_idx].copy()
        val_df = train_df.iloc[val_idx].copy()

        y_tr  = tr_df["demand"].values
        y_val = val_df["demand"].values
        if use_log_target:
            tr_df  = tr_df.copy(); tr_df["demand"]  = np.log1p(y_tr)
            val_df = val_df.copy(); val_df["demand"] = np.log1p(y_val)

        tr_ds  = TrafficSequenceDataset(tr_df,  params["seq_len"],
                                        static_cols, dynamic_cols,
                                        is_train=True,
                                        global_demand_mean=global_demand_mean)
        val_ds = TrafficSequenceDataset(val_df, params["seq_len"],
                                        static_cols, dynamic_cols,
                                        is_train=True,
                                        global_demand_mean=global_demand_mean)

        tr_loader  = DataLoader(tr_ds,  batch_size=params["batch_size"], shuffle=True,
                                num_workers=params["num_workers"],
                                pin_memory=params["pin_memory"])
        val_loader = DataLoader(val_ds, batch_size=params["batch_size"] * 2, shuffle=False,
                                num_workers=params["num_workers"],
                                pin_memory=params["pin_memory"])

        model = GridlockTCNModel(
            static_dim   = d_static,
            dynamic_dim  = d_dynamic,
            tcn_channels = params["tcn_channels"],
            kernel_size  = params["kernel_size"],
            dilations    = params["dilations"],
            dropout      = params["dropout"],
            film_hidden  = params["film_hidden"],
            head_dims    = params["head_dims"],
        ).to(device)

        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

        if params["scheduler"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=params["epochs"], eta_min=1e-6
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser, mode="min", factor=0.5, patience=5
            )

        scaler     = GradScaler(enabled=params["mixed_precision"])
        best_val   = float("inf")
        patience_c = 0
        best_state = None

        for epoch in range(params["epochs"]):
            # ── Train ─────────────────────────────────────────────────────
            model.train()
            tr_loss = 0.0
            for batch in tr_loader:
                static_b, dyn_b, target_b = [b.to(device, non_blocking=True) for b in batch]
                optimiser.zero_grad(set_to_none=True)

                with autocast(enabled=params["mixed_precision"]):
                    pred = model(static_b, dyn_b)
                    loss = huber_loss(pred, target_b, params["huber_delta"])

                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])
                scaler.step(optimiser)
                scaler.update()
                tr_loss += loss.item()

            # ── Validate ──────────────────────────────────────────────────
            model.eval()
            val_preds_fold = []
            with torch.no_grad():
                for batch in val_loader:
                    static_b, dyn_b, target_b = [b.to(device, non_blocking=True) for b in batch]
                    with autocast(enabled=params["mixed_precision"]):
                        pred = model(static_b, dyn_b)
                    val_preds_fold.append(pred.float().cpu().numpy())

            val_pred_arr = np.concatenate(val_preds_fold)
            if use_log_target:
                val_pred_arr = np.expm1(val_pred_arr)
            val_pred_arr = val_pred_arr[val_ds.inverse_order]
            val_score = competition_score(y_val, val_pred_arr)
            avg_tr_loss = tr_loss / len(tr_loader)

            if params["scheduler"] == "cosine":
                scheduler.step()
            else:
                scheduler.step(avg_tr_loss)

            # Early stopping on RMSE
            val_rmse = np.sqrt(np.mean((y_val - val_pred_arr)**2))
            if val_rmse < best_val:
                best_val  = val_rmse
                patience_c = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_c += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"    ep {epoch+1:3d}  tr_loss={avg_tr_loss:.4f}  "
                            f"val_score={val_score:.2f}  val_rmse={val_rmse:.4f}")

            if patience_c >= params["patience"]:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

        # ── Restore best model, predict ───────────────────────────────────
        model.load_state_dict(best_state)
        model.eval()

        oof_fold_preds = []
        for batch in val_loader:
            static_b, dyn_b, _ = [b.to(device, non_blocking=True) for b in batch]
            with torch.no_grad(), autocast(enabled=params["mixed_precision"]):
                pred = model(static_b, dyn_b)
            oof_fold_preds.append(pred.float().cpu().numpy())
        
        oof_fold_arr = np.concatenate(oof_fold_preds)
        if use_log_target:
            oof_fold_arr = np.expm1(oof_fold_arr)
        
        # >>> ADD THIS LINE RIGHT HERE <<<
        oof_fold_arr = oof_fold_arr[val_ds.inverse_order]

        oof_preds[val_idx] = oof_fold_arr

        # ── Test predictions ──────────────────────────────────────────────
        test_ds  = TrafficSequenceDataset(test_df, params["seq_len"],
                                          static_cols, dynamic_cols,
                                          is_train=False,
                                          global_demand_mean=global_demand_mean)
        test_loader = DataLoader(test_ds, batch_size=params["batch_size"] * 2, shuffle=False,
                                 num_workers=params["num_workers"],
                                 pin_memory=params["pin_memory"])
        test_fold_preds = []
        with torch.no_grad():
            for batch in test_loader:
                static_b, dyn_b = [b.to(device, non_blocking=True) for b in batch]
                with autocast(enabled=params["mixed_precision"]):
                    pred = model(static_b, dyn_b)
                test_fold_preds.append(pred.float().cpu().numpy())
        test_fold_arr = np.concatenate(test_fold_preds)
        if use_log_target:
            test_fold_arr = np.expm1(test_fold_arr)

        test_fold_arr = test_fold_arr[test_ds.inverse_order]
        test_preds += test_fold_arr / C.N_FOLDS

        fold_score = competition_score(train_df.iloc[val_idx]["demand"].values, oof_fold_arr)
        logger.info(f"[TCN] Fold {fold+1} competition score: {fold_score:.4f}")

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, save_dir / f"tcn_fold{fold}.pt")

        del model, tr_ds, val_ds, test_ds
        torch.cuda.empty_cache()

    overall = competition_score(train_df["demand"].values, oof_preds)
    logger.info(f"[TCN] OOF competition score: {overall:.4f}")

    return oof_preds, test_preds