from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Sequence, Dict, List
import logging
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from app.utils_debugging.tracer import trace

# Constantes para gráficos X-bar y R (n = 2..10)
A2 = {2: 1.88, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483,
      7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
D3 = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0,
      7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004,
      7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}

###############################################################################
# 1. X-BAR Y R ################################################################
###############################################################################
@trace

def xbar_r_chart(series: Sequence[float], subgroup_size: int = 5) -> Dict[str, object]:
    """Calcula medias (X-bar) y rangos (R) por subgrupos."""
    data = pd.Series(series).dropna().astype(float)
    if len(data) < subgroup_size:
        return {}

    groups = data.groupby(np.arange(len(data)) // subgroup_size)
    means = groups.mean()
    ranges = groups.max() - groups.min()
    xbar_bar = means.mean()
    r_bar = ranges.mean()

    A2_const = A2.get(subgroup_size, 0)
    D3_const = D3.get(subgroup_size, 0)
    D4_const = D4.get(subgroup_size, 0)

    ucl_x = xbar_bar + A2_const * r_bar
    lcl_x = xbar_bar - A2_const * r_bar
    ucl_r = D4_const * r_bar
    lcl_r = D3_const * r_bar

    alert_x = bool(((means > ucl_x) | (means < lcl_x)).any())
    alert_r = bool(((ranges > ucl_r) | (ranges < lcl_r)).any())

    return {
        "xbar_values": means.tolist(),
        "r_values": ranges.tolist(),
        "xbar_center": float(xbar_bar),
        "xbar_ucl": float(ucl_x),
        "xbar_lcl": float(lcl_x),
        "r_center": float(r_bar),
        "r_ucl": float(ucl_r),
        "r_lcl": float(lcl_r),
        "alerts": {"xbar": alert_x, "r": alert_r},
    }

###############################################################################
# 2. CUSUM ####################################################################
###############################################################################
@trace
def cusum_chart(series: Sequence[float], target: float | None = None,
               k: float = 0.5, h: float = 5.0) -> Dict[str, object]:
    """CUSUM estándar (acumulativo positivo y negativo)."""
    data = pd.Series(series).dropna().astype(float)
    if data.empty:
        return {}
    target = float(target) if target is not None else float(data.mean())
    c_plus, c_minus = 0.0, 0.0
    pos_vals: List[float] = []
    neg_vals: List[float] = []
    alert = False
    for x in data:
        c_plus = max(0.0, c_plus + x - target - k)
        c_minus = min(0.0, c_minus + x - target + k)
        pos_vals.append(c_plus)
        neg_vals.append(c_minus)
        if c_plus > h or abs(c_minus) > h:
            alert = True
    return {
        "cusum_pos": pos_vals,
        "cusum_neg": neg_vals,
        "cusum_target": target,
        "cusum_alert": alert,
    }

###############################################################################
# 3. EWMA #####################################################################
###############################################################################
@trace
def ewma_chart(series: Sequence[float], lambda_: float = 0.3,
              L: float = 3.0, target: float | None = None) -> Dict[str, object]:
    """EWMA con límites de control clásicos."""
    data = pd.Series(series).dropna().astype(float)
    if data.empty:
        return {}
    target = float(target) if target is not None else float(data.mean())
    z = target
    ewma_vals: List[float] = []
    for x in data:
        z = lambda_ * x + (1 - lambda_) * z
        ewma_vals.append(z)
    sigma = float(data.std(ddof=1))
    sigma_z = sigma * np.sqrt(lambda_ / (2 - lambda_))
    ucl = target + L * sigma_z
    lcl = target - L * sigma_z
    arr = np.array(ewma_vals)
    alert = bool(((arr > ucl) | (arr < lcl)).any())
    return {
        "ewma_values": ewma_vals,
        "ewma_center": target,
        "ewma_ucl": float(ucl),
        "ewma_lcl": float(lcl),
        "ewma_alert": alert,
    }

###############################################################################
# 4. Agregador general #########################################################
###############################################################################
@trace
def compute_spc(series: Sequence[float], subgroup_size: int = 5) -> Dict[str, object]:
    """Calcula X-bar/R, CUSUM y EWMA para una serie temporal."""
    if series is None:
        return {}
    ser = pd.Series(series).dropna().astype(float)
    if ser.empty:
        return {}
    xr = xbar_r_chart(ser, subgroup_size)
    cu = cusum_chart(ser)
    ew = ewma_chart(ser)
    alerts = {
        "xbar": xr.get("alerts", {}).get("xbar", False),
        "r": xr.get("alerts", {}).get("r", False),
        "cusum": cu.get("cusum_alert", False),
        "ewma": ew.get("ewma_alert", False),
    }
    return {
        "xbar": {
            "values": xr.get("xbar_values"),
            "center": xr.get("xbar_center"),
            "ucl": xr.get("xbar_ucl"),
            "lcl": xr.get("xbar_lcl"),
        },
        "r": {
            "values": xr.get("r_values"),
            "center": xr.get("r_center"),
            "ucl": xr.get("r_ucl"),
            "lcl": xr.get("r_lcl"),
        },
        "cusum": {
            "pos": cu.get("cusum_pos"),
            "neg": cu.get("cusum_neg"),
            "target": cu.get("cusum_target"),
        },
        "ewma": {
            "values": ew.get("ewma_values"),
            "center": ew.get("ewma_center"),
            "ucl": ew.get("ewma_ucl"),
            "lcl": ew.get("ewma_lcl"),
        },
        "alerts": alerts,
    }
