# longitudinal_metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import zscore
import logging

logger = logging.getLogger(__name__)


###############################################################################
# 0.  CONSTANTES Y UTILIDADES ##################################################
###############################################################################

ACPL_SD_BY_ELO = {
    #   elo_mid : sd_acpl   (valores de referencia; ajusta a tu corpus limpio)
       1200: 95, 1600: 80, 2000: 65, 2400: 55, 2800: 45
}

MATCH_SD_BY_ELO = {
       1200: 7.5, 1600: 7.0, 2000: 6.5, 2400: 5.5, 2800: 5.0
}

def interp_sd(elo: float, sd_dict: dict) -> float:
    xs, ys = zip(*sorted(sd_dict.items()))
    return np.interp(elo, xs, ys)

###############################################################################
# 1.  ROI / OPI (por partida y global) ########################################
###############################################################################

def performance_rating(match_pct: float, acpl: float,
                       coef_m: float = 800, coef_a: float = -0.5) -> float:
    """
    Misma fórmula simplificada que en quality_metrics.intrinsic_performance_rating.
    """
    return coef_m * match_pct + coef_a * acpl + 2000

def roi_per_game(games_df: pd.DataFrame,
                 match_col: str | None = None,
                 acpl_col: str | None = None) -> pd.Series:
    """
    Devuelve el ROI (performance_rating) partida a partida.
    Si faltan las columnas explícitas se usan alias seguros.
    """
    # ── Resolver columnas por alias ──────────────────────────────
    match_aliases = ["match_pct", "match_rate", "weighted_match_rate"]
    acpl_aliases  = ["acl", "acpl"]

    match_col = match_col or next(
        (c for c in match_aliases if c in games_df.columns), None
    )
    acpl_col  = acpl_col  or next(
        (c for c in acpl_aliases  if c in games_df.columns), None
    )

    if match_col is None or acpl_col is None:
        # No data → devolvemos serie vacía para no romper el flujo
        return pd.Series(dtype=float)

    pr = performance_rating(games_df[match_col], games_df[acpl_col])
    return pr

def aggregate_roi(games_df: pd.DataFrame) -> Dict[str, float]:
    roi_series = roi_per_game(games_df)
    return {
        'roi_mean'   : roi_series.mean(),
        'roi_max'    : roi_series.max(),
        'roi_sd'     : roi_series.std(ddof=1),
        'roi_games>2': (roi_series > 2).sum()    # n partidas ROI > 2 σ
    }

###############################################################################
# 2.  STEP‑FUNCTION GAINS #####################################################
###############################################################################


def detect_step_function(
    games_df: pd.DataFrame,
    *,
    aliases: Tuple[str, ...] = ("acpl", "match_pct", "match_rate", "weighted_match_rate"),
    min_delta: float = 20,
    window: int = 10,
) -> Dict[str, float | int | bool]:
    """
    Busca un cambio súbito en la serie temporal de una métrica.

    • Para métricas de **error** (por defecto «acpl») detecta un *descenso* ≥ min_delta.
    • Para métricas de **calidad** (match_*) detecta un *ascenso* ≥ min_delta.
      La métrica se selecciona usando el primer alias presente en `games_df`.

    Parámetros
    ----------
    games_df : pd.DataFrame
        DataFrame con las partidas ordenadas cronológicamente.
    aliases : tuple[str, ...]
        Nombres alternativos que se aceptan para la métrica.
    min_delta : float
        Umbral mínimo que debe superar la diferencia entre medias pre/post.
    window : int
        Longitud de la ventana deslizante para las medias móviles.

    Retorna
    -------
    dict
        {'step_<metric>_delta': float,
         'step_<metric>_index': int,
         'step_<metric>_flag' : bool}
    """
    # ── 1. Elegir columna disponible ─────────────────────────────
    metric = next((col for col in aliases if col in games_df.columns), None)
    if metric is None:
        # No existe ninguna de las columnas requeridas → métrica vacía
        return {
            "step_unknown_delta": np.nan,
            "step_unknown_index": -1,
            "step_unknown_flag": False,
        }

    # ── 2. Serie y medias móviles ───────────────────────────────
    series = games_df[metric].reset_index(drop=True)
    pre  = series.rolling(window).mean().shift(1)
    post = series.rolling(window).mean()

    # Error (ACPL) = buscamos *descenso*; Calidad (match) = *ascenso*
    if metric == "acpl":
        delta = pre - post
    else:
        delta = post - pre

    best_idx    = delta.idxmax()
    best_delta  = delta.max()
    step_detect = best_delta >= min_delta

    # ── 3. Construir salida coherente ───────────────────────────
    key_base = metric
    return {
        f"step_{key_base}_delta": best_delta,
        f"step_{key_base}_index": int(best_idx) if step_detect else -1,
        f"step_{key_base}_flag": step_detect,
    }

###############################################################################
# 3.  SELECTIVITY SCORE (varianza intra‑jugador) ##############################
###############################################################################

def selectivity_score(
    games_df: pd.DataFrame,
    *,
    match_aliases: tuple[str, ...] = ("match_pct", "match_rate", "weighted_match_rate"),
) -> dict:
    """
    % de partidas cuya calidad (match-rate) está por encima de la mediana
    personal.  Devuelve un diccionario para que .update() funcione.
    """
    match_col = next((c for c in match_aliases if c in games_df.columns), None)
    if match_col is None:
        return {"selectivity_pct": np.nan}

    s = games_df[match_col]
    pct = (s > s.median()).mean() * 100.0
    return {"selectivity_pct": pct}

###############################################################################
# 4.  PEER‑GROUP DELTA ########################################################
###############################################################################

def peer_group_delta(games_df: pd.DataFrame,
                     reference_df: pd.DataFrame,
                     elo_col: str = "elo",
                     acpl_col: str = "acpl",
                     match_col: str = "match_pct",
                     k: int = 200
                    ) -> Dict[str, float]:
    """
    Compara cada partida con jugadores “vecinos” en ELO (±k).
    Devuelve diferencia media ACPL y Match%.
    """
    deltas_acpl  = []
    deltas_match = []

    for _, row in games_df.iterrows():
        elo = row[elo_col]
        peers = reference_df[(reference_df[elo_col] >= elo-k) & (reference_df[elo_col] <= elo+k)]
        if peers.empty:
            continue
        deltas_acpl.append(peers[acpl_col].mean() - row[acpl_col])
        deltas_match.append(row[match_col] - peers[match_col].mean())

    return {
        'peer_acpl_delta' : np.nanmean(deltas_acpl)  if deltas_acpl else np.nan,
        'peer_match_delta': np.nanmean(deltas_match) if deltas_match else np.nan,
    }

###############################################################################
# 5.  STREAKS DE ALTO RENDIMIENTO ############################################
###############################################################################

def longest_streak(roi_series: pd.Series,
                   threshold: float = 2.75
                  ) -> int:
    """
    Longest consecutive streak of ROI ≥ threshold.
    """
    mask = roi_series >= threshold
    # run-length encoding
    streaks = (mask != mask.shift()).cumsum()
    return mask.groupby(streaks).sum().max()

###############################################################################
# 6.  AGREGADOR PRINCIPAL #####################################################
###############################################################################

def aggregate_longitudinal_features(
        games_df: pd.DataFrame,
        reference_df: pd.DataFrame | None = None
    ) -> Dict[str, float | int]:

    features = {}

    # --- ROI --------------------------------------------------------------
    roi_series = roi_per_game(games_df)
    features.update(aggregate_roi(games_df))
    features['longest_roi_streak'] = longest_streak(roi_series)

    # --- Step‑function gains ---------------------------------------------
    features.update(detect_step_function(games_df, aliases=("match_pct", "match_rate", "weighted_match_rate"),  min_delta=20))
    features.update(detect_step_function(games_df,aliases=("match_pct", "match_rate", "weighted_match_rate"), min_delta=5))

    # --- Selectivity ------------------------------------------------------
    features.update(selectivity_score(games_df))

    # --- Peer comparison --------------------------------------------------
    if reference_df is not None:
        features.update(peer_group_delta(games_df, reference_df))

    return features


###############################################################################
# 7.  DEMO RÁPIDO #############################################################
###############################################################################
if __name__ == "__main__":
    # Simula 60 partidas con salto brusco en la mitad
    n = 60
    df_games = pd.DataFrame({
        'elo'       : 1800,
        'acpl'      : np.concatenate([np.random.normal(70, 8, n//2),
                                      np.random.normal(35, 5, n//2)]),
        'match_pct' : np.concatenate([np.random.normal(55, 3, n//2),
                                      np.random.normal(72, 2, n//2)]),
    })

    reference = pd.DataFrame({
        'elo': np.random.randint(1700, 1900, 200),
        'acpl': np.random.normal(65, 10, 200),
        'match_pct': np.random.normal(56, 4, 200)
    })

    feats_long = aggregate_longitudinal_features(df_games, reference)
    logger.info("Longitudinal‑feature snapshot:\n%s", feats_long)


# how to implement:
# import longitudinal_metrics  as lm

# feats_l = lm.aggregate_longitudinal_features(games_df, reference_df)
# player_features = {**feats_q, **feats_t, **feats_o, **feats_e, **feats_l}

"""Métrica	Indicador de alerta
roi_mean	> +2.0 σ durante ≥ 30 partidas
step_acpl_flag	True con step_acpl_delta ≥ 25 cp
selectivity_cv	> 0.20 y selectivity_kurtosis > 3
peer_acpl_delta	> 20 cp mejor que el grupo (
longest_roi_streak	≥ 8 partidas consecutivas ROI > 2.75"""