"""Anomaly detection utilities for chess game analysis.

This module provides several anomaly detection methods that can be
applied to per-move time series extracted from a single game. The
current implementation exposes:

* :func:`stl_residual_zscores` – uses Seasonal-Trend decomposition
  (STL) to isolate residuals and returns their z-scores. This helps
  highlighting abrupt changes once trend/seasonality have been
  removed.
* :func:`isolation_forest_scores` – runs an Isolation Forest model on
  rolling window features to detect unusual behaviour patterns.
* :func:`aggregate_anomaly_features` – convenience wrapper used by the
  analysis pipeline. It combines the previous methods and returns a
  single ``anomaly_score`` for the game.

An optional ``lstm_autoencoder_scores`` stub is provided in case a
sequence autoencoder is implemented in the future.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:  # statsmodels is only needed for the STL decomposition
    from statsmodels.tsa.seasonal import STL
except Exception:  # pragma: no cover - fallback when statsmodels unavailable
    STL = None  # type: ignore

logger = logging.getLogger(__name__)


def stl_residual_zscores(series: pd.Series, period: int = 10) -> pd.Series:
    """Return z-scores of the residuals after STL decomposition.

    Parameters
    ----------
    series:
        Time series to analyse (e.g. per-move times).
    period:
        Seasonal period passed to :class:`~statsmodels.tsa.seasonal.STL`.

    Returns
    -------
    pandas.Series
        Z-scores of the residual component. Empty if decomposition is
        not possible (e.g. not enough data or statsmodels missing).
    """

    series = series.dropna()
    if STL is None or len(series) < period * 2:
        logger.debug("STL decomposition skipped – insufficient data or library")
        return pd.Series(dtype=float)

    stl = STL(series, period=period, robust=True)
    res = stl.fit()
    resid = res.resid
    if resid.std(ddof=0) == 0:
        return pd.Series(np.zeros_like(resid), index=resid.index)
    zscores = (resid - resid.mean()) / resid.std(ddof=0)
    return pd.Series(zscores, index=resid.index)


def isolation_forest_scores(features: pd.DataFrame, contamination: float = 0.1) -> np.ndarray:
    """Compute Isolation Forest anomaly scores for the given features.

    Parameters
    ----------
    features:
        DataFrame containing rolling window features.
    contamination:
        Expected proportion of anomalies in the data.

    Returns
    -------
    numpy.ndarray
        Array of anomaly scores where higher values indicate more
        anomalous observations. Returns an empty array if there are not
        enough samples for training.
    """

    if features.empty or len(features) < 10:
        logger.debug("Isolation Forest skipped – insufficient data")
        return np.array([])

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    model.fit(features)
    scores = -model.decision_function(features)
    return scores


def lstm_autoencoder_scores(series: pd.Series) -> np.ndarray:
    """Return reconstruction errors from a pre-trained LSTM autoencoder.

    The function expects a model checkpoint at ``models/lstm_autoencoder.pt``
    containing ``state_dict``, ``mean`` and ``std`` entries. If PyTorch is not
    available, the weights are missing, or the series is too short, an empty
    array is returned.
    """

    series = series.dropna().astype(float)
    if series.empty:
        return np.array([])

    try:  # Import torch lazily so that the dependency remains optional
        import torch
        from torch import nn
    except Exception:  # pragma: no cover - torch is optional
        logger.debug("PyTorch not available – skipping autoencoder")
        return np.array([])

    class LSTMAutoencoder(nn.Module):
        def __init__(self, seq_len: int, n_features: int = 1, embedding_dim: int = 8):
            super().__init__()
            self.seq_len = seq_len
            self.encoder = nn.LSTM(n_features, embedding_dim, batch_first=True)
            self.decoder = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
            self.output_layer = nn.Linear(embedding_dim, n_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple wrapper
            enc, _ = self.encoder(x)
            context = enc[:, -1, :].unsqueeze(1).repeat(1, self.seq_len, 1)
            dec, _ = self.decoder(context)
            return self.output_layer(dec)

    model_path = Path(__file__).resolve().parents[2] / "models" / "lstm_autoencoder.pt"
    if not model_path.exists():
        logger.debug("LSTM autoencoder weights not found at %s", model_path)
        return np.array([])

    checkpoint = torch.load(model_path, map_location="cpu")
    seq_len = int(checkpoint.get("seq_len", 10))
    if len(series) < seq_len:
        logger.debug("LSTM autoencoder skipped – insufficient data")
        return np.array([])

    model = LSTMAutoencoder(seq_len)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    mean = float(checkpoint.get("mean", 0.0))
    std = float(checkpoint.get("std", 1.0)) or 1.0
    data = series.values.astype(np.float32)
    norm = (data - mean) / std

    scores = []
    with torch.no_grad():
        for i in range(len(norm) - seq_len + 1):
            window = torch.tensor(norm[i : i + seq_len], dtype=torch.float32).view(1, seq_len, 1)
            recon = model(window).view(seq_len, 1).numpy().squeeze(1)
            mse = float(np.mean((window.numpy().squeeze(0).squeeze(1) - recon) ** 2))
            scores.append(mse)

    return np.array(scores)


def aggregate_anomaly_features(moves_df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate anomaly related features for a game.

    Parameters
    ----------
    moves_df:
        DataFrame with per-move information. The function expects at
        least ``move_time`` and ``cp_loss`` columns to be present.

    Returns
    -------
    dict
        ``{"anomaly_score": float, "reconstruction_error": float}``
        combining different detection methods. A score of ``0.0`` denotes
        no detected anomalies.
    """

    if moves_df.empty:
        return {"anomaly_score": 0.0}

    # --- STL based z-scores -------------------------------------------------
    zscores = stl_residual_zscores(moves_df.get("move_time", pd.Series(dtype=float)))
    max_abs_z = float(np.abs(zscores).max()) if not zscores.empty else 0.0

    # --- Isolation Forest on rolling window features -----------------------
    window = 5
    features = pd.DataFrame({
        "cp_loss_mean": moves_df["cp_loss"].rolling(window).mean(),
        "move_time_std": moves_df["move_time"].rolling(window).std(),
    }).dropna()
    iso_scores = isolation_forest_scores(features)
    iso_max = float(iso_scores.max()) if iso_scores.size else 0.0

    # --- LSTM autoencoder reconstruction error ----------------------------
    ae_scores = lstm_autoencoder_scores(moves_df.get("move_time", pd.Series(dtype=float)))
    ae_max = float(ae_scores.max()) if ae_scores.size else 0.0

    anomaly_score = max_abs_z + iso_max + ae_max
    logger.debug(
        "Anomaly detection – max_abs_z: %s, iso_max: %s, ae_max: %s, final score: %s",
        max_abs_z,
        iso_max,
        ae_max,
        anomaly_score,
    )
    return {"anomaly_score": anomaly_score, "reconstruction_error": ae_max}
