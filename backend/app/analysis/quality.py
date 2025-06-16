# quality_metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from typing import Tuple, List

###############################################################################
# 1.  Average Centipawn Loss (ACPL)   #########################################
###############################################################################

def acpl(game_df: pd.DataFrame) -> float:
    """
    Calcula el Average Centipawn Loss de un jugador en una partida.

    Espera un DataFrame con columnas:
    ─ 'eval_cp_before'  -> evaluación (centipawns) justo ANTES del movimiento,
    ─ 'eval_cp_after'   -> evaluación tras el movimiento,
    ─ 'player_to_move'  -> 'white'/'black' para filtrar.

    ACPL = mean(|eval_cp_after - eval_cp_before|)  (solo lances del jugador)
    """
    diffs = np.abs(game_df.eval_cp_after - game_df.eval_cp_before)
    return diffs.mean()


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
    Calcula un % de coincidencia con Stockfish ponderado por la
    complejidad (número de lances legales en la posición).
    
    Espera:
    ─ 'is_engine_best' (bool)   – si la jugada coincide con la 1ª línea,
    ─ 'legal_moves'    (int)    – nº de jugadas legales en la posición.
    """
    # Peso inverso: + complejidad  ⇒  mayor peso si acierta
    weights = np.log1p(max_moves_cap - game_df.legal_moves.clip(0, max_moves_cap))
    return np.average(game_df.is_engine_best, weights=weights)

###############################################################################
# 5.  Detección de rachas de precisión ########################################
###############################################################################

def precision_bursts(game_df: pd.DataFrame,
                     threshold_cp: int = 10,
                     window_size: int = 8) -> List[Tuple[int, int]]:
    """
    Devuelve una lista de (move_index_start, move_index_end) donde
    el ACPL por jugada en la ventana < threshold_cp (p. ej. 10 cp).

    Ideal para encontrar momentos donde el jugador parece 'consultar' motor.
    """
    diffs = np.abs(game_df.eval_cp_after - game_df.eval_cp_before).to_numpy()
    bursts = []
    for i in range(len(diffs) - window_size + 1):
        window = diffs[i:i + window_size]
        if window.mean() < threshold_cp:
            bursts.append((i, i + window_size - 1))
    return bursts

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

    print("ACPL partida:", acpl(df_moves))
    print("Match ponderado:", complexity_weighted_match(df_moves))
    print("Bursts:", precision_bursts(df_moves))


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

