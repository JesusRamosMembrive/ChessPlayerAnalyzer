from __future__ import annotations

"""Utilities for unsupervised clustering of player profiles.

This module builds feature vectors from aggregated metrics and fits both
KMeans and Gaussian Mixture Models (GMM).  It also provides helpers to assign
an individual player to the nearest cluster and measure the distance to its
centroid.  Trained models are persisted under ``models/clustering`` so they can
be reused by subsequent analyses.
"""

from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sqlmodel import Session, select

from app.database import engine
from app.models import PlayerAnalysisDetailed

# Directory where clustering models are stored
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "clustering"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _to_frame(rows: Iterable[Tuple[float, float, float]]) -> pd.DataFrame:
    """Convert an iterable of raw tuples to a DataFrame.

    Each tuple is ``(mean_move_time, avg_acpl, mean_entropy)``.
    """

    return pd.DataFrame(rows, columns=["mean_move_time", "avg_acpl", "mean_entropy"])


def fit_models(df: pd.DataFrame, k: int = 3) -> Tuple[KMeans, GaussianMixture]:
    """Fit KMeans and GMM models over ``df`` and return them."""

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)

    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(df)

    return kmeans, gmm


def assign_cluster_from_values(
    *, avg_acpl: float, mean_move_time: float, mean_entropy: float
) -> Tuple[int, float]:
    """Assign a player to a cluster using stored KMeans model.

    Returns a tuple ``(cluster_id, distance_to_center)``.  If no trained model is
    available, ``(-1, 0.0)`` is returned.
    """

    model_path = MODEL_DIR / "kmeans.joblib"
    if not model_path.exists():
        return -1, 0.0

    kmeans: KMeans = joblib.load(model_path)
    features = _to_frame([(mean_move_time, avg_acpl, mean_entropy)])
    cluster_id = int(kmeans.predict(features)[0])
    center = kmeans.cluster_centers_[cluster_id]
    distance = float(np.linalg.norm(features.values[0] - center))
    return cluster_id, distance


def recompute_and_update_clusters(k: int = 3) -> None:
    """Re-train clustering models with all players and update assignments."""

    with Session(engine) as session:
        players = session.exec(select(PlayerAnalysisDetailed)).all()

        rows: list[Tuple[float, float, float]] = []
        usernames: list[str] = []
        for p in players:
            mean_time = (p.time_management or {}).get("mean_move_time")
            mean_entropy = (p.opening_patterns or {}).get("mean_entropy")
            if mean_time is None or mean_entropy is None or p.avg_acpl is None:
                continue
            rows.append((mean_time, p.avg_acpl, mean_entropy))
            usernames.append(p.username)

        if not rows:
            return

        df = _to_frame(rows)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        kmeans, gmm = fit_models(df, k=k)
        joblib.dump(kmeans, MODEL_DIR / "kmeans.joblib")
        joblib.dump(gmm, MODEL_DIR / "gmm.joblib")

        clusters = kmeans.predict(df)
        centers = kmeans.cluster_centers_
        distances = np.linalg.norm(df.values - centers[clusters], axis=1)

        for username, cid, dist in zip(usernames, clusters, distances):
            player = session.get(PlayerAnalysisDetailed, username)
            if player:
                player.cluster_id = int(cid)
                player.cluster_distance = float(dist)
        session.commit()
