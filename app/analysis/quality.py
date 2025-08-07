# quality_metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

###############################################################################
# 1.  Average Centipawn Loss (ACPL)   #########################################
###############################################################################

def acpl(game_df: pd.DataFrame, player_color: str = 'white') -> float:
    """
    Calcula el Average Centipawn Loss de un jugador en una partida.

    Espera un DataFrame con columnas:
    ─ 'eval_cp_before'  -> evaluación (centipawns) justo ANTES del movimiento,
    ─ 'eval_cp_after'   -> evaluación tras el movimiento,
    ─ 'player_color'    -> 'white'/'black' para ajustar perspectiva.

    ACPL = mean(|eval_cp_after - eval_cp_before|)  (solo lances del jugador)
    
    Args:
        game_df: DataFrame with move analysis
        player_color: 'white' or 'black' - perspective for evaluation adjustment
    """
    required_cols = {"eval_cp_before", "eval_cp_after"}
    if not required_cols.issubset(game_df.columns):
        return np.nan  # o 0.0 según prefieras

    eval_before = game_df["eval_cp_before"]
    eval_after = game_df["eval_cp_after"]
    
    if player_color == 'black':
        eval_before = -eval_before
        eval_after = -eval_after

    diffs = np.abs(eval_after - eval_before)
    return diffs.mean() if len(diffs) else np.nan


###############################################################################
# 2.  ACPL ajustado al rating  #################################################
###############################################################################

class ACPLModel:
    """
    Ajusta una curva de referencia ACPL_expected(ELO) con un robust regressor
    sobre un conjunto grande de partidas 'limpias' y produce z‑scores.
    """
    def __init__(self):
        self.model = HuberRegressor()        # menos sensible a outliers
        self.sigma_ = None                   # desviación típica residual

    def fit(self, df_stats: pd.DataFrame):
        """
        df_stats debe tener columnas: ['elo', 'acpl']
        """
        X = df_stats[['elo']].values
        y = df_stats['acpl'].values
        self.model.fit(X, y)
        residuals = y - self.model.predict(X)
        self.sigma_ = residuals.std(ddof=1)
        return self

    def z_score(self, elo: float, acpl_value: float) -> float:
        """
        z > +2,75 ≈ umbral FIDE de evidencia estadística.
        """
        mu = self.model.predict([[elo]])[0]
        return (mu - acpl_value) / self.sigma_

###############################################################################
# 3.  Intrinsic Performance Rating (IPR) ######################################
###############################################################################

def intrinsic_performance_rating(match_pct: float, acpl: float,
                                 coef_match: float = 800,
                                 coef_acpl: float = -0.5) -> float:
    """
    Aproximación lineal al modelo de Regan.
    – match_pct: % de jugadas que coinciden con la 1ª línea del motor (0‑1).
    – acpl: Average Centipawn Loss.
    
    Devuelve un rating ELO estimado que explicaría esa precisión.
    Los coeficientes se obtienen calibrando sobre tu base de datos de referencia.
    """
    return coef_match * match_pct + coef_acpl * acpl + 2000  # offset base


def ipr_z_score(ipr: float, elo: float, sigma: float = 60) -> float:
    """
    Desviación típica (~60 ELO) tomada de los papers de Regan.
    """
    return (ipr - elo) / sigma


###############################################################################
# 4.  Coincidencia ponderada por complejidad ##################################
###############################################################################

def complexity_weighted_match(game_df: pd.DataFrame,
                              max_moves_cap: int = 50) -> float:
    """
    % de coincidencia con Stockfish ponderado por complejidad.
    Si todos los pesos salen 0 (p.ej. legal_moves == max_moves_cap en
    todos los lances) se hace fallback a la media simple para evitar
    ZeroDivisionError.
    """
    if "is_engine_best" not in game_df.columns:
        return np.nan

    # Peso inverso a la complejidad: +difícil ⇒ +peso si acierta
    weights = np.log1p(max_moves_cap - game_df.legal_moves.clip(0, max_moves_cap)
                       if "legal_moves" in game_df.columns
                       else max_moves_cap)

    total_w = weights.sum()
    if total_w == 0 or np.isnan(total_w):
        # fallback seguro
        return game_df.is_engine_best.mean()

    return np.dot(game_df.is_engine_best, weights) / total_w
###############################################################################
# 5.  Detección de rachas de precisión ########################################
###############################################################################

def precision_bursts(game_df: pd.DataFrame,
                     threshold_cp: int = 25,
                     window_size: int = 5) -> List[Tuple[int, int]]:
    """
    Devuelve una lista de (move_index_start, move_index_end) donde
    el ACPL por jugada en la ventana < threshold_cp (p. ej. 10 cp).

    Ideal para encontrar momentos donde el jugador parece 'consultar' motor.
    """
    diffs = np.abs(game_df["eval_cp_after"] - game_df["eval_cp_before"]).values
    bursts = []
    for i in range(len(diffs) - window_size + 1):
        window = diffs[i:i + window_size]
        if window.mean() < threshold_cp:
            bursts.append((i, i + window_size - 1))
    return bursts


BLUNDER_THRESHOLD = 300  # cp

def compute_phase_quality(moves_df_list: list[pd.DataFrame]) -> dict:
    """
    Agrega calidad por fase a nivel jugador.
    Cada moves_df debe tener:
       • 'phase'  ('opening' | 'middlegame' | 'endgame')
       • 'delta_eval'  (abs cp vs best)
    """

    if not moves_df_list:
        return {}

    combined = pd.concat(moves_df_list, ignore_index=True)

    # ACPL por fase
    phase_acpl = (
        combined.groupby("phase")["delta_eval"]
        .mean()
        .to_dict()
    )

    # Blunder rate
    blunder_rate = (
        (combined["delta_eval"].abs() > BLUNDER_THRESHOLD).mean()
        if "delta_eval" in combined else None
    )

    return {
        "opening_acpl": float(phase_acpl.get("opening", np.nan)),
        "middlegame_acpl": float(phase_acpl.get("middlegame", np.nan)),
        "endgame_acpl": float(phase_acpl.get("endgame", np.nan)),
        "blunder_rate": float(blunder_rate) if blunder_rate is not None else None,
    }

def aggregate_clutch_accuracy(games_df):
    if "clutch_accuracy_diff" not in games_df:
        return {}

    diffs = games_df["clutch_accuracy_diff"].dropna().abs()
    if diffs.empty:
        return {}

    avg_diff = float(diffs.mean())
    pct_good = float((diffs < 100).mean())  # “bueno” si <100 cp

    return {
        "avg_clutch_diff": round(avg_diff, 1),
        "clutch_games_pct": round(pct_good, 3),
    }

def aggregate_tactical_trends(games_df: pd.DataFrame) -> dict:
    # Si no hay ninguna de las dos columnas, devolver dict vacío
    if all(col not in games_df for col in ["precision_burst_count", "second_choice_rate"]):
        return {}

    burst = (
        games_df["precision_burst_count"].sum(min_count=1)
        if "precision_burst_count" in games_df else np.nan
    )
    scr = (
        games_df["second_choice_rate"].mean(skipna=True)
        if "second_choice_rate" in games_df else np.nan
    )

    return {
        "precision_burst_count": int(burst) if not np.isnan(burst) else None,
        "second_choice_rate": float(scr) if not np.isnan(scr) else None,
    }

BLUNDER = 300  # cp

def aggregate_blunders_by_phase(moves_dfs: list[pd.DataFrame]) -> dict:

    if not moves_dfs:
        return {}

    df = pd.concat(moves_dfs, ignore_index=True)
    if "phase" not in df or "delta_eval" not in df:
        return {}

    df["is_blunder"] = df["delta_eval"].abs() > BLUNDER

    phase_rates = (
        df.groupby("phase")["is_blunder"]
          .mean()
          .to_dict()
    )

    return {
        "opening_blunder_rate":  float(phase_rates.get("opening", np.nan)),
        "middlegame_blunder_rate": float(phase_rates.get("middlegame", np.nan)),
        "endgame_blunder_rate":   float(phase_rates.get("endgame", np.nan)),
        "blunder_rate":           float(df["is_blunder"].mean()),
    }

###############################################################################
# 6.  Uso de ejemplo ##########################################################
###############################################################################

if __name__ == "__main__":
    # ── Ejemplo mínimo con un DataFrame ficticio ──────────────────────────
    df_moves = pd.DataFrame({
        'move_number': np.arange(1, 41),
        'eval_cp_before': np.random.randint(-200, 200, 40),
        'eval_cp_after': np.random.randint(-200, 200, 40),
        'legal_moves': np.random.randint(5, 40, 40),
        'is_engine_best': np.random.rand(40) < 0.35,
    })

    logger.info("ACPL partida: %s", acpl(df_moves))
    logger.info("Match ponderado: %s", complexity_weighted_match(df_moves))
    logger.info("Bursts: %s", precision_bursts(df_moves))


# ─────────────────────────────────────────────────────────────────────────
#  🔗  AGGREGATOR
# ------------------------------------------------------------------------
def aggregate_quality_features(game_df, elo: int | None = None, player_color: str = 'white') -> dict:
    logger.info("DEBUG QUALITY: Starting quality features calculation")
    logger.info(f"DEBUG QUALITY: Input DataFrame shape: {game_df.shape}")
    logger.info(f"DEBUG QUALITY: Input DataFrame columns: {list(game_df.columns)}")
    logger.info(f"DEBUG QUALITY: ELO parameter: {elo}")
    
    match_rate = (
        game_df["is_engine_best"].mean() if "is_engine_best" in game_df else 0.0
    )
    logger.info(f"DEBUG QUALITY: Match rate: {match_rate}")
    
    acpl_val = acpl(game_df, player_color)
    logger.info(f"DEBUG QUALITY: ACPL value: {acpl_val}")
    
    weighted_match = complexity_weighted_match(game_df)
    logger.info(f"DEBUG QUALITY: Weighted match rate: {weighted_match}")
    
    ipr_val = intrinsic_performance_rating(match_rate, acpl_val)
    logger.info(f"DEBUG QUALITY: IPR value: {ipr_val}")

    feats = {
        "acpl"               : acpl_val,
        "match_rate"         : match_rate,
        "weighted_match_rate": weighted_match,
        "ipr"                : ipr_val,
        # ipr_z_score con valor neutro por defecto
        "ipr_z_score"        : 0.0,
    }

    if elo is not None:
        ipr_z = ipr_z_score(feats["ipr"], elo)
        feats["ipr_z_score"] = ipr_z
        logger.info(f"DEBUG QUALITY: IPR Z-score: {ipr_z}")
    else:
        logger.info("DEBUG QUALITY: No ELO provided, IPR Z-score remains 0.0")

    # Nuevo score sintético: 40 % ACPL, 30 % match_rate, 30 % weighted_match_rate
    quality_score = (
            40 * (1 - acpl_val / 100) +  # menos ACPL ⇒ mejor
            30 * match_rate +  # jugadas exactas
            30 * weighted_match  # precisión ponderada por complejidad
    )
    feats["quality_score"] = quality_score
    logger.info(f"DEBUG QUALITY: Quality score: {quality_score}")

    burst_count = len(precision_bursts(game_df))
    feats["precision_burst_count"] = burst_count
    logger.info(f"DEBUG QUALITY: Precision burst count: {burst_count}")
    
    logger.info(f"DEBUG QUALITY: Final quality features: {feats}")
    return feats

# How to implement
# Group by player and calculate
# player_stats = df_moves.groupby('player').apply(
#     lambda g: pd.Series({
#         'acpl': acpl(g),
#         'match_w': complexity_weighted_match(g),
#         'games': g.game_id.nunique(),
#         # etc.
#     })
# )
# Adjusts the ACPL_expected(ELO)rinse_quality curve.
# model = ACPLModel().fit(reference_stats)    # reference_stats: elo, acpl
# player_stats['acpl_z'] = player_stats.apply(
#     lambda r: model.z_score(r['elo'], r['acpl']), axis=1
# )
# Calculate IPR and z-score:Group by player and calculate
# player_stats['ipr'] = intrinsic_performance_rating(
#     player_stats['match_w'], player_stats['acpl']
# )
# player_stats['ipr_z'] = ipr_z_score(player_stats['ipr'], player_stats['elo'])

