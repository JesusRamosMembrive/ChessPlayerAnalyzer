from __future__ import annotations

"""Supervised classifier pipeline for suspicion analysis.

This module provides a thin wrapper around a scikit-learn model that
uses the existing game metrics as features.  It supports training with
cross-validation, prediction of probabilities, and persistence of the
trained model under ``models/``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - fallback for standalone usage
    from .causal_fairness import check_fairness_thresholds
except Exception:  # ImportError in non-package execution
    from causal_fairness import check_fairness_thresholds  # type: ignore


MODEL_FILENAME = "ml_suspicion_model.pkl"


@dataclass
class MLSuspicionClassifier:
    """Wrapper around a scikit-learn classifier for suspicious behaviour.

    The classifier expects a set of numerical features describing a
    game.  Features correspond to the metrics computed by the analysis
    engine (ACPL, match_rate, timing correlations, etc.).

    The model is persisted under ``models/`` relative to the repository
    root so it can be reused across sessions.
    """

    model_path: Optional[Path] = None
    model: Optional[Pipeline] = None
    feature_names: Optional[Iterable[str]] = None

    def __post_init__(self) -> None:
        if self.model_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            self.model_path = repo_root / "models" / MODEL_FILENAME
        if self.model_path.exists():
            data = joblib.load(self.model_path)
            self.model = data["model"]
            self.feature_names = data.get("features")

    # ------------------------------------------------------------------
    def train(
        self,
        X: pd.DataFrame,
        y: Iterable[int],
        model_type: str = "random_forest",
        cv: int = 5,
        sensitive_features: Optional[Iterable[int]] = None,
        fairness_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Train the classifier and persist it to disk.

        Parameters
        ----------
        X:
            DataFrame containing feature columns.
        y:
            Iterable with binary labels (1 suspicious, 0 clean).
        model_type:
            Either ``"random_forest"`` or ``"gradient_boosting"``.
        cv:
            Number of folds for cross-validation.
        sensitive_features:
            Optional iterable with sensitive group labels aligned with ``y``.
        fairness_thresholds:
            Thresholds for fairness metrics. If provided, demographic
            parity and equalized odds will be computed and validated.

        Returns
        -------
        Dict with cross-validation and (optionally) fairness statistics.
        """

        if model_type == "gradient_boosting":
            estimator = GradientBoostingClassifier(random_state=42)
        else:
            estimator = RandomForestClassifier(
                n_estimators=200, random_state=42
            )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ])

        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipeline, X, y, cv=cv_strategy, scoring="roc_auc"
        )

        pipeline.fit(X, y)
        self.model = pipeline
        self.feature_names = list(X.columns)

        # Persist model and feature ordering
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names}, self.model_path)

        result: Dict[str, float] = {
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
        }

        if sensitive_features is not None:
            preds = pipeline.predict(X)
            metrics = check_fairness_thresholds(
                y,
                preds,
                sensitive_features,
                fairness_thresholds or {},
            )
            result.update(metrics)

        return result

    # ------------------------------------------------------------------
    def predict_proba(self, features: Dict[str, float]) -> float:
        """Predict probability of being suspicious for a single set of features."""
        if self.model is None:
            if not self.model_path or not self.model_path.exists():
                raise FileNotFoundError(
                    "Trained model not found. Train the classifier first."
                )
            data = joblib.load(self.model_path)
            self.model = data["model"]
            self.feature_names = data.get("features")

        assert self.feature_names is not None
        row = [features.get(name, 0.0) for name in self.feature_names]
        X = pd.DataFrame([row], columns=self.feature_names)
        proba = self.model.predict_proba(X)[0, 1]
        return float(proba)
