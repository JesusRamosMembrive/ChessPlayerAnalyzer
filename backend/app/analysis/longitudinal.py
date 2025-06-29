# longitudinal_metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
from scipy.stats import linregress

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
    if roi_series.empty:
        return {
            'roi_mean': 0.0,
            'roi_max': 0.0,
            'roi_std': 0.0,
            'roi_games>2': 0
        }
    
    return {
        'roi_mean': roi_series.mean() if not roi_series.isna().all() else 0.0,
        'roi_max': roi_series.max() if not roi_series.isna().all() else 0.0,
        'roi_sd': roi_series.std(ddof=1) if len(roi_series) > 1 and not roi_series.isna().all() else 0.0,
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
            "step_unknown_delta": 0.0,
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

    if delta.isna().all() or delta.empty:
        best_idx = -1
        best_delta = 0.0
        step_detect = False
    else:
        best_idx = delta.idxmax()
        best_delta = delta.max()
        if pd.isna(best_idx) or pd.isna(best_delta):
            best_idx = -1
            best_delta = 0.0
            step_detect = False
        else:
            step_detect = best_delta >= min_delta

    # ── 3. Construir salida coherente ───────────────────────────
    key_base = metric
    return {
        f"step_{key_base}_delta": float(best_delta) if not pd.isna(best_delta) else 0.0,
        f"step_{key_base}_index": int(best_idx) if step_detect and not pd.isna(best_idx) else -1,
        f"step_{key_base}_flag": bool(step_detect),
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
        return {"selectivity_pct": 50.0}

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
        'peer_delta_acpl': np.nanmean(deltas_acpl) if deltas_acpl and not np.isnan(np.nanmean(deltas_acpl)) else 0.0,
        'peer_delta_match': np.nanmean(deltas_match) if deltas_match and not np.isnan(np.nanmean(deltas_match)) else 0.0,
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
    max_streak = mask.groupby(streaks).sum().max()
    return int(max_streak) if not pd.isna(max_streak) and not np.isnan(max_streak) else 0



def compute_trends(games_df: pd.DataFrame) -> dict:
    """
    Devuelve dict acorde a PerformanceTrendsOut:
        • trend_acpl (cp/100 games)
        • trend_match_rate (Δ/100 games)
        • roi_curve (media ROI por mes, máx 24 puntos)
    Requiere columnas: 'acpl', 'match_rate', 'roi', 'date'.
    """
    logger.info(f"DEBUG TRENDS: compute_trends called with DataFrame shape: {games_df.shape}")
    logger.info(f"DEBUG TRENDS: Available columns: {list(games_df.columns)}")

    if games_df.empty or "date" not in games_df:
        logger.warning(f"DEBUG TRENDS: Missing required data - empty: {games_df.empty}, has date: {'date' in games_df.columns}")
        return {}

    # --- 1. Ordenar por fecha
    df = games_df.sort_values("date").reset_index(drop=True)
    x = np.arange(len(df)) / 100          # 1 unidad = 100 partidas

    # --- 2. Tendencia lineal (polyfit grado 1)
    def slope(col: str) -> float | None:
        if df[col].isna().all():
            return 0.0
        m, _ = np.polyfit(x, df[col].fillna(df[col].median()), 1)
        return float(m) if not np.isnan(float(m)) else 0.0

    trend_acpl = slope("acpl")
    trend_match = slope("match_rate")

    # --- 3. ROI curve (últimos 24 meses)
    roi_curve = (
        df.set_index("date")
          .groupby(pd.Grouper(freq="M"))["roi"]
          .mean()
          .tail(24)
          .round(2)
          .tolist()
    )
    
    roi_curve_clean = [float(x) for x in roi_curve if not pd.isna(x) and not np.isnan(x)] if roi_curve else []

    return {
        "trend_acpl": trend_acpl if trend_acpl is not None and not np.isnan(trend_acpl) else 0.0,
        "trend_match_rate": trend_match if trend_match is not None and not np.isnan(trend_match) else 0.0,
        "roi_curve": roi_curve_clean if roi_curve_clean else None,
    }

###############################################################################
# 6.  AGREGADOR PRINCIPAL #####################################################
###############################################################################

def aggregate_longitudinal_features(
        games_df: pd.DataFrame,
        reference_df: pd.DataFrame | None = None
    ) -> Dict[str, float | int]:

    logger.info("DEBUG LONGITUDINAL: Starting longitudinal features calculation")
    logger.info(f"DEBUG LONGITUDINAL: Games DataFrame shape: {games_df.shape}")
    logger.info(f"DEBUG LONGITUDINAL: Games DataFrame columns: {list(games_df.columns)}")
    logger.info(f"DEBUG LONGITUDINAL: Reference DataFrame provided: {reference_df is not None}")

    features = {}



    # --- ROI --------------------------------------------------------------
    logger.info("DEBUG LONGITUDINAL: Calculating ROI features")
    roi_series = roi_per_game(games_df)
    logger.info(f"DEBUG LONGITUDINAL: ROI series length: {len(roi_series)}")
    if len(roi_series) > 0:
        logger.info(f"DEBUG LONGITUDINAL: ROI series stats - mean: {roi_series.mean():.2f}, max: {roi_series.max():.2f}")
    
    roi_features = aggregate_roi(games_df)
    features.update(roi_features)
    logger.info(f"DEBUG LONGITUDINAL: ROI features: {roi_features}")
    
    longest_roi_streak = longest_streak(roi_series)
    features['longest_streak'] = longest_roi_streak
    logger.info(f"DEBUG LONGITUDINAL: Longest ROI streak: {longest_roi_streak}")

    # --- Step‑function gains ---------------------------------------------
    logger.info("DEBUG LONGITUDINAL: Detecting step functions")
    step_features_20 = detect_step_function(games_df, aliases=("match_pct", "match_rate", "weighted_match_rate"), min_delta=20)
    features.update(step_features_20)
    logger.info(f"DEBUG LONGITUDINAL: Step function features (delta=20): {step_features_20}")
    
    step_features_5 = detect_step_function(games_df, aliases=("match_pct", "match_rate", "weighted_match_rate"), min_delta=5)
    features.update(step_features_5)
    logger.info(f"DEBUG LONGITUDINAL: Step function features (delta=5): {step_features_5}")

    # --- ACPL Step function detection (for engine compatibility) ---------------
    logger.info("DEBUG LONGITUDINAL: Detecting ACPL step functions")
    step_acpl_features = detect_step_function(games_df, aliases=("acpl",), min_delta=25)
    features.update(step_acpl_features)
    logger.info(f"DEBUG LONGITUDINAL: ACPL step function features: {step_acpl_features}")

    # --- Selectivity ------------------------------------------------------
    logger.info("DEBUG LONGITUDINAL: Calculating selectivity score")
    selectivity_features = selectivity_score(games_df)
    features.update(selectivity_features)
    logger.info(f"DEBUG LONGITUDINAL: Selectivity features: {selectivity_features}")

    # --- Peer comparison --------------------------------------------------
    if reference_df is not None:
        logger.info("DEBUG LONGITUDINAL: Calculating peer group comparison")
        peer_features = peer_group_delta(games_df, reference_df)
        features.update(peer_features)
        logger.info(f"DEBUG LONGITUDINAL: Peer comparison features: {peer_features}")
    else:
        logger.info("DEBUG LONGITUDINAL: Skipping peer comparison - no reference data")

    # --- Date range calculations ------------------------------------------
    if "created_at" in games_df.columns and not games_df.empty:
        logger.info("DEBUG LONGITUDINAL: Calculating date range")
        first_date = games_df["created_at"].min()
        last_date = games_df["created_at"].max()
        features.update({
            "first_game_date": first_date,
            "last_game_date": last_date,
        })
        logger.info(f"DEBUG LONGITUDINAL: Date range - first: {first_date}, last: {last_date}")
    else:
        logger.info("DEBUG LONGITUDINAL: Skipping date calculations - no created_at column or empty DataFrame")

    logger.info(f"DEBUG LONGITUDINAL: Final longitudinal features: {features}")
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
