# endgame_metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np
import chess
import chess.pgn
from chess.syzygy import Tablebase
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

###############################################################################
# 0.  UTILIDADES GENERALES ####################################################
###############################################################################

def is_tb_position(board: chess.Board, max_pieces: int = 7) -> bool:
    """True si la posición puede estar en las tablebases N‑piece."""    
    return board.piece_count() <= max_pieces and not board.is_variant_end()

def result_sign(board: chess.Board) -> int:
    """
    WDL ganador/perdedor relativo al side‑to‑move:
      1  -> ganar,   0 -> tablas,   ‑1 -> perder
    """
    if board.is_checkmate():
        return -1       # side‑to‑move está en mate
    if board.is_stalemate():
        return 0
    return None         # no es final inmediato

###############################################################################
# 1.  TABLEBASE MATCH‑RATE ####################################################
###############################################################################

def tablebase_match_rate(game: chess.pgn.Game,
                         tb: Tablebase,
                         multipv: int = 1
                        ) -> Tuple[int, int, float]:
    """
    Recorre la partida y devuelve:
      • n_tb_positions  – nº de lances donde la posición tiene ≤7 piezas
      • n_matches       – jugadas del jugador que coinciden con la mejor TB
      • match_rate      – ratio  n_matches / n_tb_positions
    """
    board = game.board()
    n_tb_positions = n_matches = 0

    for move in game.mainline_moves():
        if is_tb_position(board):
            wdl = tb.probe_wdl(board)          # 1/0/‑1 relative to side‑to‑move
            best_moves = [
                m for m in board.legal_moves
                if tb.probe_wdl(board, m) == wdl
            ]
            n_tb_positions += 1
            if move in best_moves:
                n_matches += 1
        board.push(move)

    match_rate = n_matches / n_tb_positions if n_tb_positions else np.nan
    return n_tb_positions, n_matches, match_rate

###############################################################################
# 2.  DTM / DTZ DEVIATION #####################################################
###############################################################################

def dtz_deviation(game: chess.pgn.Game,
                  tb: Tablebase,
                  max_pieces: int = 7
                 ) -> List[int]:
    """
    Para cada posición TB‑legal devuelve delta_DTZ:
      DTZ(jugada) ‑ DTZ(óptima)
    Valores > 0 indican jugadas sub‑óptimas (tardan más en mate/tablas).
    """
    board = game.board()
    deviations = []

    for move in game.mainline_moves():
        if is_tb_position(board, max_pieces):
            best_dtz = tb.probe_dtz(board)
            move_dtz = tb.probe_dtz(board, move)
            deviations.append(move_dtz - best_dtz)
        board.push(move)

    return deviations


###############################################################################
# 3.  CONVERSION EFFICIENCY (de ventaja a mate) ###############################
###############################################################################

def find_first_significant_advantage(eval_series: pd.Series,
                                     threshold_cp: int = 500
                                    ) -> int | None:
    """
    Devuelve el índice (ply) en el que la evaluación del motor supera +threshold_cp
    por primera vez (para el bando que finalmente gana).
    """
    for idx, cp in enumerate(eval_series):
        if cp >= threshold_cp:
            return idx
    return None


def conversion_efficiency(game_df: pd.DataFrame,
                          eval_col: str = "eval_cp_after",
                          threshold_cp: int = 500
                         ) -> int | None:
    """
    game_df: una fila por jugada (tras la jugada del eventual ganador).
    Devuelve nº de jugadas entre:
      • la 1ª vez que eval ≥ +threshold_cp   y
      • el mate real. 0 → mate instantáneo.
    None si el lado ganador nunca tuvo ventaja ≥ threshold_cp.
    """
    idx_start = find_first_significant_advantage(game_df[eval_col], threshold_cp)
    if idx_start is None:
        return None                      # No hubo ventaja "ganadora"
    # buscar el mate (última fila)
    idx_end = game_df.index[-1]
    return idx_end - idx_start


###############################################################################
# 4.  AGREGADOR DE FEATURES ENDGAME ###########################################
###############################################################################


def aggregate_endgame_features(
        game: chess.pgn.Game,
        moves_df: pd.DataFrame,
        tb_path: str | Path | None,
        eval_col: str = "eval_cp_after"
) -> Dict[str, float | int]:
    """
    Calcula métricas de final; si `tb_path` es None o no existe, todas las
    métricas Syzygy devuelven NaN para no romper el pipeline.
    """
    logger.info("DEBUG ENDGAME: Starting endgame features calculation")
    logger.info(f"DEBUG ENDGAME: Tablebase path: {tb_path}")
    logger.info(f"DEBUG ENDGAME: Moves DataFrame shape: {moves_df.shape}")
    logger.info(f"DEBUG ENDGAME: Eval column: {eval_col}")
    
    tb_positions = match_pct = dtz_mean = np.nan

    if tb_path and Path(tb_path).exists():
        logger.info("DEBUG ENDGAME: Tablebase path exists, analyzing with Syzygy")
        try:
            from chess.syzygy import Tablebase
            with Tablebase(str(tb_path)) as tb:
                n_tb_pos, n_match, mrate = tablebase_match_rate(game, tb)
                tb_positions = n_tb_pos
                match_pct    = mrate * 100 if mrate is not np.nan else np.nan
                dtz_dev_list = dtz_deviation(game, tb)
                dtz_mean     = np.mean(dtz_dev_list) if dtz_dev_list else np.nan
                logger.info(f"DEBUG ENDGAME: Tablebase analysis - positions: {tb_positions}, match %: {match_pct}, DTZ mean: {dtz_mean}")
        except Exception as e:
            logger.warning(f"DEBUG ENDGAME: Tablebase analysis failed: {e}")
            logger.info("DEBUG ENDGAME: tb_match_rate and dtz_deviation will be null due to tablebase error")
    else:
        if tb_path is None:
            logger.info("DEBUG ENDGAME: No tablebase path provided (TB_PATH is None)")
        else:
            logger.info(f"DEBUG ENDGAME: Tablebase path does not exist: {tb_path}")
        logger.info("DEBUG ENDGAME: tb_match_rate and dtz_deviation will be null - tablebases not available")
        logger.info("DEBUG ENDGAME: To enable tablebase analysis, install Syzygy tablebases and set SYZYGY_PATH environment variable")

    conv_eff = conversion_efficiency(moves_df, eval_col)
    logger.info(f"DEBUG ENDGAME: Conversion efficiency: {conv_eff}")
    
    perfect_tb = match_pct is not np.nan and match_pct >= 95
    fast_conversion = conv_eff is not None and conv_eff <= 10
    logger.info(f"DEBUG ENDGAME: Perfect TB flag: {perfect_tb}, Fast conversion flag: {fast_conversion}")

    result = {
        "tb_positions"        : tb_positions,
        "tb_match_rate"       : match_pct,
        "dtz_deviation"       : dtz_mean,
        "conversion_efficiency": conv_eff,
        "perfect_tb_flag"     : perfect_tb,
        "fast_conversion_flag": fast_conversion
    }
    
    logger.info(f"DEBUG ENDGAME: Final endgame features: {result}")
    return result


def aggregate_endgame_efficiency(games_df: pd.DataFrame) -> dict:
    required = {"conversion_efficiency", "tb_match_rate", "dtz_deviation"}
    if required.isdisjoint(games_df.columns):
        return {}  # ➊ nada que agregar → no rompe flujo

    conv = (games_df["conversion_efficiency"].mean(skipna=True)
            if "conversion_efficiency" in games_df else np.nan)
    tb = (games_df["tb_match_rate"].mean(skipna=True)
          if "tb_match_rate" in games_df else np.nan)
    dtz = (games_df["dtz_deviation"].mean(skipna=True)
           if "dtz_deviation" in games_df else np.nan)

    return {
        "conversion_efficiency": int(round(conv)) if not np.isnan(conv) else None,
        "tb_match_rate": float(tb) if not np.isnan(tb) else None,
        "dtz_deviation": float(dtz) if not np.isnan(dtz) else None,
    }

###############################################################################
# 5.  DEMO (requiere TB locales) ##############################################
###############################################################################

if __name__ == "__main__":
    import io

    # Carga un PGN corto de ejemplo
    pgn_text = """
    [Event "Demo"]
    [Site "?"]
    [Result "1-0"]

    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6 dxc6 5. Nxe5 Qd4 6. Nf3 Qxe4+ 7. Qe2
    Qxe2+ 8. Kxe2 f6 9. Re1 Kf7 10. d3 Bd6 11. Nc3 Bg4 12. h3 Re8+ 13. Be3 Bxf3+
    14. Kxf3 Ne7 15. Ne4 Nf5 16. g3 Be5 17. c3 Rd8 18. d4 Bd6 19. Bf4 Bxf4
    20. Kxf4 Ne7 21. Nc5 Nd5+ 22. Kf3 b6 23. Nxa6 Ra8 24. Nb4 Nxb4 25. cxb4
    Rhd8 26. Re4 Rd5 27. a3 Rad8 28. Rc1 R8d6 29. Ke3 g5 30. f3 f5 31. Re5
    Rxd4 32. Rxf5+ Kg6 33. Re5 Rd3+ 34. Kf2 Rd2+ 35. Re2 R2d3 36. Rce1 Rf6
    37. Re6 Rxf3+ 38. Kf3+ 1-0
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))

    # Simula un moves_df ficticio (solo para el ejemplo)
    n_moves = len(list(game.mainline_moves()))
    dummy_moves_df = pd.DataFrame({
        'eval_cp_after': np.linspace(0, 900, n_moves)   # simulación creciente
    })

    # Ajusta la ruta a tus TB reales
    TB_PATH = Path("/tb/syzygy")

    if TB_PATH.exists():
        feats_end = aggregate_endgame_features(game, dummy_moves_df, TB_PATH)
        logger.info("Endgame‑feature snapshot:\n%s", feats_end)
    else:
        logger.info(">> Ruta de tablebases no encontrada; ejecuta la demo cuando las tengas.")





# how to implement:
# pip install python-chess[syzygy]
# mkdir -p /tb/syzygy

# from endgame_metrics import aggregate_endgame_features

# feats_e = aggregate_endgame_features(
#     game_pgn, moves_df, TB_PATH
# )
# player_features = {**feats_q, **feats_t, **feats_o, **feats_e}
