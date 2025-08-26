from __future__ import annotations
"""Simple Bayesian model for computing suspicion probabilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Callable, Optional
import json


@dataclass
class BayesianSuspicionModel:
    """Tiny Naive Bayes model for `P(suspicious | evidence)`.

    The model is intentionally lightweight.  It provides a deterministic way
    of combining a prior – based on rating and playing experience – with a set
    of likelihood ratios derived from observed evidence (ACPL, timing metrics
    …).  The goal is not to be statistically perfect but to have an extensible
    place where new evidence can be plugged in easily.
    """

    # Base prior probability that a random game is suspicious
    base_prior: float = 0.10
    priors: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = Path(__file__).resolve().parents[2] / "data" / "suspicion_priors.json"
        if path.exists():
            self.priors = json.loads(path.read_text())
        else:
            self.priors = {}

    def _rating_bucket(self, rating: Optional[int]) -> str:
        if rating is None:
            return "unknown"
        start = (rating // 200) * 200
        return f"{start}-{start + 199}"

    def _exp_bucket(self, experience: int) -> str:
        start = (experience // 50) * 50
        return f"{start}-{start + 49}"

    def _clamp(self, p: float) -> float:
        return max(0.001, min(0.999, p))

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------
    def compute_prior(self, rating: Optional[int], experience: int) -> float:
        """Compute prior :math:`P(\text{suspicious})` using a Beta-Binomial model.

        Historical counts of suspicious games are stored in
        ``data/suspicion_priors.json`` grouped by rating and experience buckets.
        For a given player we look up the corresponding hyper-parameters
        (:math:`\alpha`, :math:`\beta`).  Player experience adds extra
        (assumed non-suspicious) trials to the beta distribution which shrinks
        the prior as more clean games are observed.
        """
        r_bucket = self._rating_bucket(rating)
        e_bucket = self._exp_bucket(experience)
        params = self.priors.get(r_bucket, {}).get(e_bucket)
        if params:
            alpha = params["alpha"]
            beta = params["beta"]
            start = int(e_bucket.split('-')[0])
            extra = max(0, experience - start)
            p = alpha / (alpha + beta + extra)
        else:
            p = self.base_prior
        return self._clamp(p)

    # ------------------------------------------------------------------
    # Likelihoods
    # ------------------------------------------------------------------
    def _likelihood_ratio(self, feature: str, value: float) -> float:
        """Return likelihood ratio P(evidence|suspicious)/P(evidence|not)."""
        rules: Dict[str, Callable[[float], float]] = {
            "acpl": lambda v: 5.0 if v < 20 else 1.0,
            "match_rate": lambda v: 4.0 if v > 0.7 else 1.0,
            "time_complexity_corr": lambda v: 3.0 if v < 0.1 else 1.0,
            "lag_spike_count": lambda v: 2.0 if v > 2 else 1.0,
            "opening_entropy": lambda v: 2.0 if v < 1.0 else 1.0,
            "second_choice_rate": lambda v: 3.0 if v > 0.8 else 1.0,
        }
        func = rules.get(feature)
        return func(value) if func else 1.0

    # ------------------------------------------------------------------
    def update(self, rating: Optional[int], experience: int, evidence: Dict[str, float]) -> float:
        """Compute posterior probability given evidence."""
        prior = self.compute_prior(rating, experience)
        odds = prior / (1 - prior)
        for name, value in evidence.items():
            odds *= self._likelihood_ratio(name, value)
        posterior = odds / (1 + odds)
        return self._clamp(posterior)
