# opening_metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np
import chess.pgn
import chess.polyglot
from typing import List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


###############################################################################
# 0.  Utilidades auxiliares ###################################################
###############################################################################

def shannon_entropy(series: pd.Series) -> float:
    """
    H = −Σ p_i·log2(p_i)   (bits)
    """
    clean_series = series.dropna()
    if clean_series.empty:
        return 0.0
    
    counts = clean_series.value_counts()
    probs  = counts / counts.sum()
    return -(probs * np.log2(probs)).sum()


def load_reference_book(path: str | Path) -> chess.polyglot.Reader:
    """
    Abre un libro Polyglot o PGN grande convertido a .bin (mucho más rápido).
    """
    return chess.polyglot.open_reader(path)


def novelty_ply(game: chess.pgn.Game, book: chess.polyglot.Reader) -> int:
    """
    Devuelve el número de ply (1‑based) en el que la partida se desvía del libro.
    Si no se desvía, devuelve el número total de jugadas del libro disponibles.
    """
    board = game.board()
    for ply, move in enumerate(game.mainline_moves(), start=1):
        # Si la posición previa NO está en el libro → la novedad es en ply‑1
        if len(list(book.find_all(board))) == 0:
            return ply - 1
        board.push(move)
    # No salió nunca del libro
    return ply


###############################################################################
# 1.  ENTROPÍA DE APERTURAS ###################################################
###############################################################################

def opening_entropy(games_df: pd.DataFrame,
                    eco_col: str = "eco_code",
                    elo: int | pd.Series | None = None,
                    reference_entropy_by_elo: pd.Series | None = None
                   ) -> dict:
    """
    Calcula H y, opcionalmente, un z‑score contra la distribución por tramo ELO.
    * games_df debe contener una fila por partida con columna 'eco_code' (A00‑E99).

    Devuelve { 'H_opening': float, 'H_z': float | None }
    """
    H = shannon_entropy(games_df[eco_col])
    if reference_entropy_by_elo is None or elo is None:
        return {'H_opening': H, 'H_z': None}

    # Tomamos referencia interpolando por ELO
    ref = reference_entropy_by_elo.sort_index()
    exp_H = np.interp(elo, ref.index, ref.values)
    sigma = ref.std(ddof=1)
    H_z   = (H - exp_H) / sigma
    return {'H_opening': H, 'H_z': H_z}


###############################################################################
# 2.  PROFUNDIDAD DE NOVEDAD (TN‑depth) ######################################
###############################################################################

def novelty_depth_stats(games: List[chess.pgn.Game],
                        book: chess.polyglot.Reader
                       ) -> dict:
    """
    Recorre una lista de objetos chess.pgn.Game y calcula:
      • media, mediana y sd de la profundidad de novedad
    """
    depths = [novelty_ply(g, book) for g in games]
    depths = np.array(depths, dtype=int)

    return {
        'mean_tn_depth' : depths.mean(),
        'median_tn_depth': np.median(depths),
        'sd_tn_depth'   : depths.std(ddof=1),
        'pct_late_nov'  : (depths >= 20).mean()     # % de novedades “tardías”
    }


###############################################################################
# 3.  COINCIDENCIA CON 2ª / 3ª LÍNEA DEL MOTOR ###############################
###############################################################################

def second_choice_rate(moves_df: pd.DataFrame,
                       rank_col: str = "bestmove_rank",
                       delta_eval_col: str = "delta_eval",
                       threshold_cp: int = 50
                      ) -> dict:
    """
    moves_df debe contener una fila por jugada del jugador con:
      • 'bestmove_rank'   = 1 (jugada 1ª línea), 2 (2ª) …
      • 'delta_eval'      = |eval(best) - eval(jugada)| en centipawns

    Calcula:
      • % jugadas que son rank 2 o 3 cuando la jugada 1 supera umbral Δ‑eval.
    """
    mask = moves_df[delta_eval_col] > threshold_cp
    if not mask.any():
        return {'second_choice_pct': np.nan}

    secondish = moves_df.loc[mask, rank_col].isin([2, 3]).mean()
    return {'second_choice_pct': secondish}


###############################################################################
# 4.  BREADTH‑/FOCUS‑INDEX ####################################################
###############################################################################

def repertoire_breadth_focus(games_df: pd.DataFrame,
                             eco_col: str = "eco_code",
                             min_occurrences: int = 3
                            ) -> dict:
    """
    Mide equilibrio variedad‑especialización:
      • breadth = nº de códigos ECO diferentes
      • focus   = % de partidas concentradas en tus 3 líneas más jugadas
    """
    eco_counts = games_df[eco_col].value_counts()
    breadth = eco_counts.size
    focus   = eco_counts.head(3).sum() / eco_counts.sum()

    # Optionally flag “hyper‑specialist”
    hyper_specialist = focus > 0.75 and breadth >= min_occurrences
    return {
        'breadth_openings' : breadth,
        'focus_top3_pct'   : focus,
        'hyper_specialist' : hyper_specialist
    }


###############################################################################
# 5.  AGREGADOR GENERAL #######################################################
###############################################################################

def aggregate_opening_features(opening_key: str,
                               eco_code: str | None,
                               moves_df: pd.DataFrame,
                               games_df: pd.DataFrame) -> dict:
    """
    Empaqueta todas las métricas de repertorio en un único diccionario.
    * games_df : 1 fila / partida, eco, etc.
    * moves_df : 1 fila / jugada (bestmove_rank, delta_eval, …)
    * games_pgn: listado de objetos chess.pgn.Game (para calcular TN‑depth)
    """
    if "eco_code" not in games_df:
        games_df = games_df.assign(eco_code=None)
    entropy = shannon_entropy(games_df["eco_code"])
    novelty_depth = (
        next((i for i, mv in enumerate(moves_df.played, 1)
              if mv not in opening_key.split()), len(moves_df))
    )

    return {
        "opening_entropy": entropy,
        "novelty_depth": novelty_depth,
        "second_choice_rate": (moves_df.best_rank == 1).mean(),
        "opening_breadth": moves_df.played[:8].nunique(),
        # peso simple en 0-100
        "opening_score": 100 * (1 - novelty_depth / 50),
    }

# ---------------------------------------------------------- #
# 7.  AGREGADOR -- NIVEL JUGADOR ---------------------------- #
# ---------------------------------------------------------- #
def aggregate_player_opening_patterns(games_df: pd.DataFrame,
                                      moves_dfs: list[pd.DataFrame]) -> dict:
    """
    Devuelve mean_entropy, novelty_depth, opening_breadth y second_choice_rate.
    Requiere 'eco_code' y no falla si falta 'delta_eval'.
    """
    # ── 0. Use eco_code directly ──────────────────────────────────────────
    if "eco_code" not in games_df.columns:
        return {"mean_entropy": np.nan, "novelty_depth": np.nan,
                "opening_breadth": 0, "second_choice_rate": np.nan}
    eco_series = games_df["eco_code"]

    # ── 1. Entropía y breadth ─────────────────────────────────────────
    mean_entropy   = shannon_entropy(eco_series)
    opening_breadth = eco_series.nunique()

    # ── 2. Profundidad media de novedad ───────────────────────────────
    nov_depths = []
    for game_row, mv_df in zip(games_df.itertuples(), moves_dfs):
        opening_key = getattr(game_row, "opening_key", "")
        nov = next((i for i, mv in enumerate(mv_df.played, 1)
                    if mv not in opening_key.split()), len(mv_df))
        nov_depths.append(nov)
    novelty_depth = float(np.mean(nov_depths))

    # ── 3. Second-choice-rate global (sólo si hay delta_eval) ─────────
    ranks = []
    for mv_df in moves_dfs:
        if "delta_eval" not in mv_df.columns:
            continue
        mask = mv_df["delta_eval"] > 50
        if mask.any():
            ranks.append(mv_df.loc[mask, "best_rank"].isin([2, 3]).mean())
    second_choice_rate = float(np.mean(ranks)) if ranks else np.nan

    return {
        "mean_entropy": mean_entropy,
        "novelty_depth": novelty_depth,
        "opening_breadth": opening_breadth,
        "second_choice_rate": second_choice_rate,
    }

###############################################################################
# 6.  DEMO RÁPIDO #############################################################
###############################################################################

if __name__ == "__main__":
    # --- Dummy games_df (10 partidas) ----
    games_df = pd.DataFrame({
        'eco_code': ['C54', 'B30', 'C54', 'C54', 'B30', 'A46', 'C54', 'C50', 'C50', 'B30']
    })

    # --- Dummy moves_df (50 jugadas) -----
    moves_df = pd.DataFrame({
        'bestmove_rank': np.random.choice([1, 2, 3], 50, p=[0.6, 0.3, 0.1]),
        'delta_eval'   : np.random.randint(10, 200, 50)
    })

    # --- Dummy PGN list (crear part. vacías con ECO random) ----
    dummy_game = chess.pgn.Game()
    games_pgn = [dummy_game] * 10

    # --- Cálculo de features ----
    feats = aggregate_opening_features(
        games_df,
        moves_df,
        games_pgn,
        elo=1800,
        reference_entropy_by_elo=pd.Series({1400:1.5, 1800:2.2, 2200:2.8}),
        book_path=None      # pon aquí 'reference_book.bin' cuando lo tengas
    )
    logger.info("Opening‑feature snapshot:\n%s", feats)



# How to implement
# Fusión de featuresfeats_q = quality_metrics.aggregate_quality_features(moves_df)
# feats_t = timing_metrics.aggregate_time_features(moves_df)
# feats_o = opening_metrics.aggregate_opening_features(
#              games_df, moves_df, games_pgn,
#              elo=player_elo,
#              reference_entropy_by_elo=entropy_curve,
#              book_path='reference_book.bin'
#          )

#player_features = {**feats_q, **feats_t, **feats_o}
