# timing_metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple, Any
from scipy.stats import spearmanr, lognorm, kstest
import logging
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure repository root is on the Python path so imports like ``app.*`` work
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from app.utils_debugging.tracer import trace


###############################################################################
# EXPECTED COLUMNS PER MOVE
# ─────────────────────────────────────────────────────────────────────────────
# 'move_time'            float   – segundos consumidos en el movimiento
# 'legal_moves'          int     – nº de lances legales en la posición
# 'is_engine_best'       bool    – True si coincide con 1ª línea Stockfish
# 'player_clock_before'  float   – segundos en el reloj antes de mover
# 'eval_cp_before'       int     – evaluación (centipawns) antes de mover  (opcional)
# 'eval_cp_after'        int     – evaluación tras mover                  (opcional)
###############################################################################


# --------------------------------------------------------------------------- #
# 1.  Estadística básica de tiempos                                           #
# --------------------------------------------------------------------------- #
@trace
def time_stats(game_df: pd.DataFrame) -> Tuple[float, float, float]:
    """Devuelve media, desviación típica y coef. de variación del tiempo por jugada."""
    times_series = pd.to_numeric(game_df.get("move_time", pd.Series(dtype=float)), errors="coerce")
    times = times_series[np.isfinite(times_series)].values  # Extract NumPy array
    if len(times) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(times))
    std = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
    cv = (std / mean) if mean else np.nan
    return mean, std, cv

@trace
def low_variance_flag(game_df: pd.DataFrame, threshold_std: float = 1.5) -> bool:
    """
    Señal ‘varianza baja’: std < threshold_std segundos (p.ej. 1,5 s en blitz).
    Un std tan reducido suele indicar automatización o “copiar‑pegar”.
    """
    _, std, _ = time_stats(game_df)
    return std < threshold_std


# --------------------------------------------------------------------------- #
# 2.  Correlación tiempo-complejidad                                          #
# --------------------------------------------------------------------------- #
@trace
def time_complexity_correlation(game_df: pd.DataFrame,
                                method: str = "spearman") -> float | None | Any:
    """
    Correlación entre tiempo invertido y complejidad (# legal_moves).
    """
    if "move_time" not in game_df or "legal_moves" not in game_df:
        return 0.0

    mt = pd.to_numeric(game_df["move_time"], errors="coerce").values
    lm = pd.to_numeric(game_df["legal_moves"], errors="coerce").values
    mask = ~np.isnan(mt) & ~np.isnan(lm) & np.isfinite(mt) & np.isfinite(lm)
    if not mask.any():
        return 0.0

    mt = mt[mask]
    lm = lm[mask]

    if method == "spearman":
        if np.std(mt, ddof=0) == 0 or np.std(lm, ddof=0) == 0:
            return 0.0
        try:
            corr, _ = spearmanr(mt, lm)
            return float(corr) if np.isfinite(corr) else 0.0
        except Exception:
            return 0.0

    try:
        # For non-spearman methods, convert back to pandas for .corr()
        mt_series = pd.Series(mt)
        lm_series = pd.Series(lm)
        corr = mt_series.corr(lm_series, method=method)
        return float(corr) if corr is not None and np.isfinite(corr) else 0.0
    except Exception:
        return 0.0

# --------------------------------------------------------------------------- #
# 3.  ‘Lag spikes’ (pausa + ráfaga perfecta)                                  #
# --------------------------------------------------------------------------- #
@trace
def detect_lag_spikes(game_df: pd.DataFrame,
                      pause_sec: Tuple[float, float] = (5.0, 12.0),
                      rapid_thresh: float      = 2.0,
                      rapid_window: int        = 2,
                      accuracy_required: bool  = False
                     ) -> List[int]:
    """
    Devuelve los índices de movimiento donde hay pausa seguida de jugadas rápidas y precisas.
    """
    logger.info(f"DEBUG TIMING: detect_lag_spikes params pause_sec={pause_sec}, rapid_thresh={rapid_thresh}, rapid_window={rapid_window}, accuracy_required={accuracy_required}")
    idx: List[int] = []
    t_series = pd.to_numeric(game_df.get("move_time", pd.Series(dtype=float)), errors="coerce")
    t = t_series.fillna(np.inf).to_numpy()
    if accuracy_required and "is_engine_best" in game_df:
        best = game_df["is_engine_best"].fillna(False).to_numpy(dtype=bool)
    else:
        best = np.ones_like(t, dtype=bool)

    n = len(t)
    if n == 0 or rapid_window <= 0:
        return idx

    for i in range(n - rapid_window):
        if pause_sec[0] <= t[i] <= pause_sec[1]:
            window_times = t[i+1 : i+1+rapid_window]
            window_best  = best[i+1 : i+1+rapid_window]
            if np.all(window_times < rapid_thresh) and np.all(window_best):
                idx.append(i)
    return idx


# --------------------------------------------------------------------------- #
# 4.  Exactitud bajo presión (“clutch accuracy”)                              #
# --------------------------------------------------------------------------- #
@trace
def clutch_accuracy(game_df: pd.DataFrame,
                    clutch_threshold: float = 30.0
                   ) -> float:
    """
    Diferencia de precisión entre fase con < clutch_threshold s en reloj y el resto.
    Usa delta_eval si existe; si no, swing de evaluación; si no, % de coincidencia.
    """
    logger.info(f"DEBUG TIMING: clutch_accuracy threshold={clutch_threshold}")
    if 'player_clock_before' not in game_df.columns:
        return 0.0

    pcb = pd.to_numeric(game_df["player_clock_before"], errors="coerce")
    clutch_mask = pcb < clutch_threshold
    non_mask = ~clutch_mask

    if "delta_eval" in game_df.columns:
        d = pd.to_numeric(game_df["delta_eval"], errors="coerce").abs().values
        clutch = np.nanmean(d[clutch_mask]) if clutch_mask.any() else np.nan
        normal = np.nanmean(d[non_mask]) if non_mask.any() else np.nan
        diff = (normal - clutch) if np.isfinite(normal) and np.isfinite(clutch) else 0.0
        logger.info(f"DEBUG TIMING: clutch_accuracy using delta_eval: normal={normal}, clutch={clutch}, diff={diff}")
        return float(diff)

    if {'eval_cp_before', 'eval_cp_after'} <= set(game_df.columns):
        before = pd.to_numeric(game_df["eval_cp_before"], errors="coerce").values
        after  = pd.to_numeric(game_df["eval_cp_after"], errors="coerce").values
        diffs = np.abs(after - before)
        clutch = np.nanmean(diffs[clutch_mask]) if clutch_mask.any() else np.nan
        normal = np.nanmean(diffs[non_mask]) if non_mask.any() else np.nan
        diff = (normal - clutch) if np.isfinite(normal) and np.isfinite(clutch) else 0.0
        logger.info(f"DEBUG TIMING: clutch_accuracy using eval swing: normal={normal}, clutch={clutch}, diff={diff}")
        return float(diff)

    if "is_engine_best" in game_df.columns:
        engine_best = game_df["is_engine_best"].values.astype(float)
        clutch = np.nanmean(engine_best[clutch_mask]) if clutch_mask.any() else np.nan
        normal = np.nanmean(engine_best[non_mask]) if non_mask.any() else np.nan
        if not (np.isnan(clutch) or np.isnan(normal)):
            diff = float(clutch - normal)
            logger.info(f"DEBUG TIMING: clutch_accuracy using is_engine_best: normal={normal}, clutch={clutch}, diff={diff}")
            return diff
    return 0.0


# --------------------------------------------------------------------------- #
# 5.  Forma de la distribución de tiempos                                     #
# --------------------------------------------------------------------------- #
@trace
def uniformity_score(game_df: pd.DataFrame) -> float:
    """
    Kolmogorov–Smirnov contra log‑normal ajustada.
    """
    times = pd.to_numeric(game_df.get("move_time", pd.Series(dtype=float)), errors="coerce").dropna()
    times = times.clip(lower=1e-3)
    if len(times) < 5:
        return 0.0
    try:
        shape, loc, scale = lognorm.fit(times, floc=0)
        cdf = lambda x: lognorm.cdf(x, shape, loc=loc, scale=scale)
        ks_stat, _ = kstest(times, cdf)
        return float(ks_stat)
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# 6.  Agregador cómodo para ML / scoring                                      #
# --------------------------------------------------------------------------- #
@trace
def aggregate_time_features(game_df: pd.DataFrame) -> dict:
    logger.info("DEBUG TIMING: Starting timing features calculation")
    logger.info(f"DEBUG TIMING: Input DataFrame shape: {game_df.shape}")
    logger.info(f"DEBUG TIMING: Input DataFrame columns: {list(game_df.columns)}")
    
    if game_df.empty or "move_time" not in game_df:
        logger.info("DEBUG TIMING: No timing data available, returning default values")
        return {
            "mean_move_time"      : np.nan,
            "time_variance"       : np.nan,
            "time_complexity_corr": np.nan,
            "lag_spike_count"     : 0,
            "uniformity_score"    : np.nan,
            "clutch_accuracy_diff": None,
            "timing_score"        : 0,
            "timing_rows_total"   : int(game_df.shape[0]),
            "timing_rows_valid_time": 0,
            "timing_rows_valid_corr": 0,
        }

    # Garantizar columnas requeridas
    df = game_df.copy()
    if "move_time" not in df:
        df = df.assign(move_time=0)
        logger.info("DEBUG TIMING: Added default move_time column")
    if "legal_moves" not in df:
        df = df.assign(legal_moves=0)
        logger.info("DEBUG TIMING: Added default legal_moves column")

    mt = pd.to_numeric(df["move_time"], errors="coerce").values
    mean_t = float(np.nanmean(mt))
    var_t  = float(np.nanvar(mt, ddof=1))
    valid_time = int(np.sum(~np.isnan(mt)))
    logger.info(f"DEBUG TIMING: Mean move time: {mean_t}, Variance: {var_t}, valid_time_rows={valid_time}")

    try:
        q10 = float(np.nanquantile(mt, 0.10))
        q50 = float(np.nanquantile(mt, 0.50))
        q90 = float(np.nanquantile(mt, 0.90))
        logger.info(f"DEBUG TIMING: move_time quantiles p10={q10}, p50={q50}, p90={q90}")
    except Exception:
        logger.info("DEBUG TIMING: move_time quantiles unavailable")

    if "delta_eval" in df.columns:
        de = np.abs(pd.to_numeric(df["delta_eval"], errors="coerce").values)
        if np.any(~np.isnan(de)):
            try:
                de_q10 = float(np.nanquantile(de, 0.10))
                de_q50 = float(np.nanquantile(de, 0.50))
                de_q90 = float(np.nanquantile(de, 0.90))
                logger.info(f"DEBUG TIMING: delta_eval(abs) quantiles p10={de_q10}, p50={de_q50}, p90={de_q90}")
                CAP = 1500
                extremes = int(np.sum(de > CAP))
                logger.info(f"DEBUG TIMING SANITY: suspected mate-driven extremes (> {CAP}cp): {extremes}")
            except Exception:
                logger.info("DEBUG TIMING: delta_eval quantiles unavailable")

    corr = time_complexity_correlation(df)
    logger.info(f"DEBUG TIMING: Time-complexity correlation: {corr}")

    lm = pd.to_numeric(df["legal_moves"], errors="coerce").values
    valid_corr = int(np.sum(~np.isnan(mt) & ~np.isnan(lm)))
    total_rows = int(df.shape[0]) if hasattr(df, "shape") else len(df)
    corr_pct = 100.0 * valid_corr / max(total_rows, 1)
    logger.info(f"DEBUG TIMING SANITY: Valid rows for correlation: {valid_corr}/{total_rows} ({corr_pct:.1f}%)")

    lag_spikes = len(detect_lag_spikes(df))
    logger.info(f"DEBUG TIMING: Lag spikes detected: {lag_spikes}")

    uniform = uniformity_score(df) if len(df) >= 5 else 0.0
    uniform = max(0.0, min(1.0, uniform)) if not np.isnan(uniform) else 0.0
    corr_safe = corr if (corr is not None and not np.isnan(corr)) else 0.0
    score   = 50 * uniform + 50 * max(corr_safe, 0)

    clutch_acc = clutch_accuracy(df) if 'player_clock_before' in df.columns else None
    logger.info(f"DEBUG TIMING: Clutch accuracy: {clutch_acc}")

    result = {
        "mean_move_time"        : mean_t,
        "time_variance"         : var_t,
        "time_complexity_corr"  : corr,
        "lag_spike_count"       : lag_spikes,
        "uniformity_score"      : uniform,
        "clutch_accuracy_diff"  : clutch_acc,
        "timing_score"          : score,
        "timing_rows_total"     : int(df.shape[0]),
        "timing_rows_valid_time": valid_time,
        "timing_rows_valid_corr": valid_corr,
    }
    
    logger.info(f"DEBUG TIMING: Final timing features: {result}")
    return result
@trace
def aggregate_time_management(moves_dfs):
    if not moves_dfs:
        return {}

    mt = np.concatenate([df["move_time"].values for df in moves_dfs if "move_time" in df])
    mean_t  = float(np.mean(mt))
    var_t   = float(np.var(mt, ddof=1))
    spikes  = int(np.sum(mt > 5 * mean_t))

    uniformity = 1 - (np.std(mt) / mean_t) if mean_t else 0

    return {
        "mean_move_time": mean_t,
        "time_variance": var_t,
        "uniformity_score": round(uniformity, 3),
        "lag_spike_count": spikes,
    }
@trace
def aggregate_time_complexity_corr(games_df: pd.DataFrame) -> dict:
    if games_df.empty or "time_complexity_corr" not in games_df:
        return {}
    corr_values = games_df["time_complexity_corr"].values
    mean_corr = np.nanmean(corr_values)
    return {"time_complexity_corr": float(mean_corr)}
# --------------------------------------------------------------------------- #
# 7.  Ejemplo mínimo (ejecución directa)                                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    n_moves = 40
    demo = pd.DataFrame({
        'move_time'           : np.random.exponential(scale=2.5, size=n_moves),
        'legal_moves'         : np.random.randint(5, 40, n_moves),
        'is_engine_best'      : np.random.rand(n_moves) < 0.35,
        'player_clock_before' : np.linspace(300, 0, n_moves),
        'eval_cp_before'      : np.random.randint(-200, 200, n_moves),
        'eval_cp_after'       : np.random.randint(-200, 200, n_moves),
    })

    logger.info("-" * 60)
    logger.info("Time‑feature snapshot:\n%s", aggregate_time_features(demo))


# ─────────────────────────────────────────────────────────────────────────



# How to implement
# Fusion of features

# import quality_metrics as qm
# import timing_metrics  as tm
# feats_q  = qm.aggregate_quality_features(df_moves)     # función que defines en tu código
# feats_t  = tm.aggregate_time_features(df_moves)
# features = {**feats_q, **feats_t}timing
