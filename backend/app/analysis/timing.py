# timing_metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple, Any
from scipy.stats import spearmanr, lognorm, kstest, skew
import logging

logger = logging.getLogger(__name__)

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
def time_stats(game_df: pd.DataFrame) -> Tuple[float, float, float]:
    """Devuelve media, desviación típica y coef. de variación del tiempo por jugada."""
    times = game_df.move_time.to_numpy()
    mean = times.mean()
    std  = times.std(ddof=1)
    cv   = std / mean if mean else np.nan
    return mean, std, cv


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
def time_complexity_correlation(game_df: pd.DataFrame,
                                method: str = "spearman") -> float | None | Any:
    """
    Correlación entre tiempo invertido y complejidad (# legal_moves).
    Esperamos un coeficiente **positivo** en juego humano; ≈ 0 o negativo es sospechoso.
    Si alguna de las dos series es constante, devolvemos 0 para evitar el
    ConstantInputWarning de SciPy.
    """
    if "move_time" not in game_df or "legal_moves" not in game_df:
        return 0.0

    if method == "spearman":
        if np.std(game_df.move_time) == 0 or np.std(game_df.legal_moves) == 0:
            return 0.0
        corr, _ = spearmanr(game_df.move_time, game_df.legal_moves)
        return corr

    return game_df.move_time.corr(game_df.legal_moves, method=method)

# --------------------------------------------------------------------------- #
# 3.  ‘Lag spikes’ (pausa + ráfaga perfecta)                                  #
# --------------------------------------------------------------------------- #
def detect_lag_spikes(game_df: pd.DataFrame,
                      pause_sec: Tuple[float, float] = (8.0, 15.0),
                      rapid_thresh: float      = 1.0,
                      rapid_window: int        = 3,
                      accuracy_required: bool  = True
                     ) -> List[int]:
    """
    Devuelve los índices de movimiento donde:
      • El jugador se detiene 'pause_sec' segundos,
      • Seguidos de 'rapid_window' jugadas < rapid_thresh s,
      • (Opcional) todas las jugadas son 1ª línea de Stockfish.
    """
    idx = []
    t   = game_df.move_time.to_numpy()
    if accuracy_required:
        best = game_df.is_engine_best.to_numpy()
    else:
        best = np.ones_like(t, dtype=bool)

    for i in range(len(t) - rapid_window):
        if pause_sec[0] <= t[i] <= pause_sec[1]:
            window_times = t[i+1 : i+1+rapid_window]
            window_best  = best[i+1 : i+1+rapid_window]
            if np.all(window_times < rapid_thresh) and np.all(window_best):
                idx.append(i)
    return idx


# --------------------------------------------------------------------------- #
# 4.  Exactitud bajo presión (“clutch accuracy”)                              #
# --------------------------------------------------------------------------- #
def clutch_accuracy(game_df: pd.DataFrame,
                    clutch_threshold: float = 30.0
                   ) -> float:
    """
    Diferencia de precisión (ACPL o %match) entre:
      – Fase con < clutch_threshold s en el reloj y
      – Resto de la partida.
    Valor **positivo** ⇒ juega MEJOR con poco tiempo → atípico.
    """
    if 'player_clock_before' not in game_df.columns:
        return np.nan

    clutch_mask = game_df.player_clock_before < clutch_threshold
    non_mask    = ~clutch_mask

    # --- Usamos ACPL si las evaluaciones están disponibles
    if {'eval_cp_before', 'eval_cp_after'} <= set(game_df.columns):
        diffs = np.abs(game_df.eval_cp_after - game_df.eval_cp_before)
        clutch  = diffs[clutch_mask].mean() if clutch_mask.any() else np.nan
        normal  = diffs[non_mask].mean()    if non_mask.any()  else np.nan
        return normal - clutch          # ↑valor ⇒ clutch más preciso
    # --- Fallback a porcentaje de coincidencia
    clutch = game_df.is_engine_best[clutch_mask].mean() if clutch_mask.any() else np.nan
    normal = game_df.is_engine_best[non_mask].mean()    if non_mask.any()    else np.nan
    return clutch - normal


# --------------------------------------------------------------------------- #
# 5.  Forma de la distribución de tiempos                                     #
# --------------------------------------------------------------------------- #
def uniformity_score(game_df: pd.DataFrame) -> float:
    """
    Kolmogorov–Smirnov contra log‑normal ajustada.
    KS > 0.20 sugiere que la distribución NO es log‑normal (i.e. demasiado uniforme).
    """
    times = game_df.move_time.clip(lower=1e-3)       # evita ceros para log
    if len(times) < 5:
        return np.nan
    shape, loc, scale = lognorm.fit(times, floc=0)   # f‑loc=0   (≥scipy 1.12)
    cdf = lambda x: lognorm.cdf(x, shape, loc=loc, scale=scale)
    ks_stat, _ = kstest(times, cdf)
    return ks_stat


# --------------------------------------------------------------------------- #
# 6.  Agregador cómodo para ML / scoring                                      #
# --------------------------------------------------------------------------- #
def aggregate_time_features(game_df: pd.DataFrame) -> dict:
    if game_df.empty or "move_time" not in game_df:
        return {
            "mean_move_time"      : np.nan,
            "time_variance"       : np.nan,
            "time_complexity_corr": np.nan,
            "lag_spike_count"     : 0,
            "uniformity_score"    : np.nan,
            "timing_score"        : 0,
        }

    # Garantizar columnas requeridas
    if "move_time" not in game_df:
        game_df = game_df.assign(move_time=0)
    if "legal_moves" not in game_df:            #  ⇦  salvaguarda final
        game_df = game_df.assign(legal_moves=0)

    mean_t = game_df["move_time"].mean()
    var_t  = game_df["move_time"].var()
    corr   = time_complexity_correlation(game_df)

    # ejemplo simple de uniformidad y score
    uniform = 1 - (var_t / (mean_t**2 + 1e-6))
    score   = 50 * uniform + 50 * max(corr, 0)

    return {
        "mean_move_time"      : mean_t,
        "time_variance"       : var_t,
        "time_complexity_corr": corr,
        "lag_spike_count"     : (game_df["move_time"] > 20).sum(),
        "uniformity_score"    : uniform,
        "timing_score"        : score,
    }

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