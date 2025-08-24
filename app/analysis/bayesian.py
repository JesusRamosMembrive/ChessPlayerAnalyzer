from __future__ import annotations
"""Simple Bayesian model for computing suspicion probabilities."""

from dataclasses import dataclass
from typing import Dict, Callable, Optional


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

    def _clamp(self, p: float) -> float:
        return max(0.001, min(0.999, p))

    # ------------------------------------------------------------------
    # Priors
    # ------------------------------------------------------------------
    def compute_prior(self, rating: Optional[int], experience: int) -> float:
        """Compute prior P(suspicious) from rating and experience.

        Low rated or very inexperienced players get a slightly higher prior,
        whereas experienced players get a lower prior.
        """
        p = self.base_prior
        if rating:
            # Players well above 2000 have a bit lower prior; beginners slightly higher
            p += (1500 - rating) / 10000  # rating 2500 -> -0.1, rating 1000 -> +0.05
        if experience:
            p -= min(experience / 1000, 0.05)  # up to -0.05 for very experienced players
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
