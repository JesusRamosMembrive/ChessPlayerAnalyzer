import numpy as np
import pandas as pd
from typing import Dict


def fit_performance_model(series: pd.Series, model: str = "arima") -> Dict[str, float]:
    """Fit a simple time-series model for performance metrics.

    Parameters
    ----------
    series : pd.Series
        Series of historical metric values (ACPL, ROI, etc.).
    model : str
        Type of model to fit. Options: ``"arima"`` or ``"kalman"``.

    Returns
    -------
    dict
        Dictionary containing fitted parameters and model type.
    """
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        raise ValueError("series must contain data")

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

    raise ValueError("model must be 'arima' or 'kalman'")


def predict_performance(model: Dict[str, float], steps: int = 1) -> np.ndarray:
    """Generate forward predictions from a fitted model."""
    if model.get("model") == "arima":
        last = model["last"]
        phi = model["phi"]
        c = model["c"]
        preds = []
        for _ in range(steps):
            last = c + phi * last
            preds.append(last)
        return np.array(preds)

    if model.get("model") == "kalman":
        state = model["state"]
        return np.full(steps, state)

    raise ValueError("unknown model type")
