import numpy as np
import pandas as pd
from scipy.stats import t
from typing import Iterable, List


def cusum_change_points(series: Iterable[float], threshold: float | None = None, drift: float = 0.0) -> List[int]:
    """Detect change points using a simple two-sided CUSUM algorithm.

    Parameters
    ----------
    series : iterable of float
        Time series data.
    threshold : float, optional
        Threshold for the cumulative sum. If ``None`` it is set to ``5 * sd``.
    drift : float, optional
        Expected drift between observations.

    Returns
    -------
    list[int]
        Indices where a change point is detected.
    """
    values = pd.Series(series).dropna().to_numpy()
    if len(values) < 2:
        return []
    if threshold is None:
        threshold = 5 * np.std(values)
    pos_sum = 0.0
    neg_sum = 0.0
    change_points: List[int] = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1] - drift
        pos_sum = max(0.0, pos_sum + diff)
        neg_sum = min(0.0, neg_sum + diff)
        if pos_sum > threshold or neg_sum < -threshold:
            change_points.append(i)
            pos_sum = 0.0
            neg_sum = 0.0
    return change_points


def bayesian_online_change_points(series: Iterable[float], hazard_lambda: float = 250) -> List[int]:
    """Bayesian online change point detection (Adams & MacKay, 2007).

    Parameters
    ----------
    series : iterable of float
        Time series data.
    hazard_lambda : float, optional
        Parameter for the constant hazard function (larger → less frequent changes).

    Returns
    -------
    list[int]
        Indices of probable change points.
    """
    data = pd.Series(series).dropna().to_numpy()
    n = len(data)
    if n == 0:
        return []

    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1

    mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 1.0
    mu = np.full(n + 1, mu0)
    kappa = np.full(n + 1, kappa0)
    alpha = np.full(n + 1, alpha0)
    beta = np.full(n + 1, beta0)

    hazard = 1.0 / hazard_lambda
    change_points: List[int] = []

    for t_idx, x in enumerate(data, start=1):
        pred_mean = mu[:t_idx]
        pred_kappa = kappa[:t_idx]
        pred_alpha = alpha[:t_idx]
        pred_beta = beta[:t_idx]

        pred_var = pred_beta * (pred_kappa + 1) / (pred_alpha * pred_kappa)
        pred_probs = t.pdf(x, df=2 * pred_alpha, loc=pred_mean, scale=np.sqrt(pred_var))

        growth_probs = pred_probs * R[:t_idx, t_idx - 1] * (1 - hazard)
        cp_prob = np.sum(pred_probs * R[:t_idx, t_idx - 1] * hazard)

        R[1:t_idx + 1, t_idx] = growth_probs
        R[0, t_idx] = cp_prob
        R[:t_idx + 1, t_idx] /= np.sum(R[:t_idx + 1, t_idx])

        mu[1:t_idx + 1] = (pred_kappa * pred_mean + x) / (pred_kappa + 1)
        kappa[1:t_idx + 1] = pred_kappa + 1
        alpha[1:t_idx + 1] = pred_alpha + 0.5
        beta[1:t_idx + 1] = pred_beta + (pred_kappa * (x - pred_mean) ** 2) / (2 * (pred_kappa + 1))

        if R[0, t_idx] > 0.5:
            change_points.append(t_idx - 1)

    return change_points


def detect_change_points(series: Iterable[float], threshold: float | None = None, hazard_lambda: float = 250) -> List[int]:
    """Combine CUSUM and Bayesian detectors for a robust set of change points."""
    cusum_cps = cusum_change_points(series, threshold=threshold)
    bayes_cps = bayesian_online_change_points(series, hazard_lambda=hazard_lambda)
    return sorted(set(cusum_cps + bayes_cps))
