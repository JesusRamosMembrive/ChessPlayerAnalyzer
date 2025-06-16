# longitudinal_metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import zscore


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
                 elo_col: str = "elo",
                 acpl_col: str = "acpl",
                 match_col: str = "match_pct"
                ) -> pd.Series:
    """
    Devuelve el ROI (z‑score) de cada partida:
      (PerformanceRating − ELO) / 60    (σ≈60 ELO según Regan)
    """
    pr = performance_rating(games_df[match_col], games_df[acpl_col])
    return (pr - games_df[elo_col]) / 60.0

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

def detect_step_function(games_df: pd.DataFrame,
                         metric: str = "acpl",
                         min_delta: float = 20,
                         window: int = 10
                        ) -> Dict[str, float | int]:
    """
    Busca un descenso súbito ≥ min_delta (ACPL) o ascenso (Match) puertas
    deslizantes de longitud 'window'.
    """
    series = games_df[metric].reset_index(drop=True)
    pre  = series.rolling(window).mean().shift(1)
    post = series.rolling(window).mean()
    delta = pre - post if metric == "acpl" else post - pre

    best_idx = delta.idxmax()
    best_delta = delta.max()

    flag = best_delta >= min_delta
    return {
        f'step_{metric}_delta' : best_delta,
        f'step_{metric}_index' : int(best_idx) if flag else -1,
        f'step_{metric}_flag'  : flag
    }

###############################################################################
# 3.  SELECTIVITY SCORE (varianza intra‑jugador) ##############################
###############################################################################

def selectivity_score(games_df: pd.DataFrame,
                      match_col: str = "match_pct"
                     ) -> Dict[str, float]:
    """
    Coeficiente de variación y kurtosis de la precisión entre partidas.
    Alto CV + alta kurtosis ⇒ alterna partidas muy “buenas” y muy “malas”.
    """
    s = games_df[match_col]
    cv = s.std(ddof=1) / s.mean()
    k  = s.kurtosis()
    return {'selectivity_cv': cv, 'selectivity_kurtosis': k}

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
    features.update(detect_step_function(games_df, metric="acpl",  min_delta=20))
    features.update(detect_step_function(games_df, metric="match_pct", min_delta=5))

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
    print("Longitudinal‑feature snapshot:\n", feats_long)


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