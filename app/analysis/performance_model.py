import numpy as np
import pandas as pd
from typing import Dict, Tuple


def fit_garch_model(series: pd.Series) -> Dict[str, float]:
    """Fit a rudimentary GARCH(1,1) model.

    This implementation is intentionally lightweight and avoids external
    dependencies.  Parameters are estimated using basic moment
    approximations rather than full maximum likelihood estimation.
    """

    s = pd.Series(series).dropna().astype(float)
    if len(s) < 2:
        raise ValueError("series must contain at least two observations")

    resid = s - s.mean()
    var = float(np.var(resid, ddof=1)) or 1e-6

    # Simple moment-based parameter guesses
    alpha = 0.1
    beta = 0.8
    omega = max(var * (1 - alpha - beta), 1e-6)

    sigma2 = var
    for r in resid:
        sigma2 = omega + alpha * r ** 2 + beta * sigma2

    return {
        "model": "garch",
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "last": float(s.iloc[-1]),
        "last_var": float(sigma2),
        "last_resid_sq": float(resid.iloc[-1] ** 2),
    }


def fit_performance_model(series: pd.Series, model: str = "auto") -> Dict[str, float]:
    """Fit a simple time-series model for performance metrics.

    Parameters
    ----------
    series : pd.Series
        Series of historical metric values (ACPL, ROI, etc.).
    model : str
        Type of model to fit. Options: ``"arima"``, ``"kalman"``, ``"garch"`` or
        ``"auto"`` for automatic selection based on volatility.

    Returns
    -------
    dict
        Dictionary containing fitted parameters and model type.
    """
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        raise ValueError("series must contain data")

    if model == "auto":
        vol = float(np.std(s.diff().dropna())) if len(s) > 1 else 0.0
        if vol > 1.0:
            model = "garch"
        elif vol > 0.1:
            model = "arima"
        else:
            model = "kalman"

    if model == "arima":
        if len(s) < 2:
            phi = 0.0
            c = float(s.iloc[-1])
        else:
            y = s.iloc[1:].values
            X = np.vstack([np.ones(len(y)), s.iloc[:-1].values]).T
            c, phi = np.linalg.lstsq(X, y, rcond=None)[0]
        return {"model": "arima", "phi": float(phi), "c": float(c), "last": float(s.iloc[-1])}

    if model == "kalman":
        x = float(s.iloc[0])
        p = 1.0
        q = float(np.var(s.diff().dropna(), ddof=1) or 1.0)
        r = float((np.var(s, ddof=1) * 0.1) or 1.0)
        for z in s:
            p = p + q
            k = p / (p + r)
            x = x + k * (z - x)
            p = (1 - k) * p
        return {"model": "kalman", "state": float(x), "p": float(p), "q": float(q), "r": float(r)}

    if model == "garch":
        return fit_garch_model(s)

    raise ValueError("model must be 'arima', 'kalman', 'garch' or 'auto'")


def predict_performance(
    model: Dict[str, float],
    steps: int = 1,
    return_conf_int: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate forward predictions from a fitted model.

    For GARCH models, if ``return_conf_int`` is True a tuple of
    ``(preds, lower, upper)`` arrays is returned representing 95%
    confidence bands.
    """

    if model.get("model") == "arima":
        if return_conf_int:
            raise ValueError("confidence intervals only available for garch models")
        last = model["last"]
        phi = model["phi"]
        c = model["c"]
        preds = []
        for _ in range(steps):
            last = c + phi * last
            preds.append(last)
        return np.array(preds)

    if model.get("model") == "kalman":
        if return_conf_int:
            raise ValueError("confidence intervals only available for garch models")
        state = model["state"]
        return np.full(steps, state)

    if model.get("model") == "garch":
        last = model["last"]
        omega = model["omega"]
        alpha = model["alpha"]
        beta = model["beta"]
        var = model["last_var"]
        resid_sq = model.get("last_resid_sq", 0.0)
        preds, lower, upper = [], [], []
        for _ in range(steps):
            var = omega + alpha * resid_sq + beta * var
            resid_sq = 0.0
            preds.append(last)
            ci = 1.96 * np.sqrt(var)
            lower.append(last - ci)
            upper.append(last + ci)
        preds = np.array(preds)
        if return_conf_int:
            return preds, np.array(lower), np.array(upper)
        return preds

    raise ValueError("unknown model type")
