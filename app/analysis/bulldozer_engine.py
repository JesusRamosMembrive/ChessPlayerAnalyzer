# app/analysis/bulldozer_engine.py
"""
BULLDOZER TOTAL: Motor de análisis ultra-simplificado.

FILOSOFÍA:
- Una función que recibe PGN → retorna JSON con análisis
- Sin clases complejas, sin fases separadas, sin over-engineering
- Funciones puras, fácil de testear, fácil de entender
- Error handling simple: si algo falla, retorna None en esa métrica
"""
import logging
import math
import chess
import chess.engine
import chess.pgn
import io
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

from app.validation import validate_analysis_metrics, InvalidAnalysisDataError

# Import existing analysis modules (BULLDOZER: reuse calculations, simplify architecture)
try:
    import pandas as pd
    import numpy as np
    from . import quality, timing, openings, endgame, longitudinal
    ANALYSIS_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Analysis modules not available: {e}")
    ANALYSIS_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)


def analyze_game_complete(pgn_text: str, username: str, color: str,
                         stockfish_path: str = None, depth: int = 12) -> Dict:
    """
    BULLDOZER TOTAL: Análisis completo con TODAS las métricas.

    Args:
        pgn_text: PGN completo de la partida
        username: Usuario a analizar
        color: 'white' o 'black'
        stockfish_path: Path a stockfish (opcional)
        depth: Profundidad de análisis (default 12)

    Returns:
        Dict con estructura COMPLETA:
        {
            "quality": {
                "acpl": float, "wdl_loss": float, "robust_loss": float,
                "ipr": float, "ipr_z_score": float, "match_rate": float,
                "weighted_match_rate": float, "precision_burst_count": int,
                "phase_quality": {"opening_acpl": float, "middlegame_acpl": float, "endgame_acpl": float},
                "blunder_analysis": {"blunders": int, "mistakes": int, "inaccuracies": int},
                "second_choice_rate": float
            },
            "timing": {
                "mean_move_time": float, "time_variance": float, "uniformity_score": float,
                "lag_spike_count": int, "clutch_accuracy_diff": float,
                "time_complexity_corr": float, "moves": [float, ...]
            },
            "opening": {
                "opening_entropy": float, "novelty_depth": int,
                "repertoire_breadth": int, "second_choice_rate": float
            },
            "endgame": {
                "conversion_efficiency": int, "tb_match_rate": float,
                "dtz_deviation": float
            },
            "moves": [
                {
                    "move_number": int, "played": str, "best": str,
                    "cp_loss": int, "eval_before": int, "eval_after": int,
                    "time_spent": float, "phase": str, "legal_moves_count": int
                }, ...
            ]
        }
    """
    try:
        logger.info(f"Starting BULLDOZER COMPLETE analysis for {username} ({color})")

        # 1. Parse PGN and create chess objects
        game = _parse_pgn(pgn_text)
        if not game:
            return {"error": "Failed to parse PGN"}

        # 2. Run Stockfish analysis and create DataFrame
        moves_df = _create_moves_dataframe(game, username, color, stockfish_path, depth)
        if moves_df is None or moves_df.empty:
            return {"error": "Failed to create moves DataFrame"}

        # 3. Calculate ALL metrics using existing modules
        analysis = _calculate_complete_metrics(moves_df, game, username, color)

        # 4. Validate (fail-fast, no sanitization)
        try:
            validated_analysis = validate_analysis_metrics(analysis, f"game_analysis_{username}")
            logger.info(f"COMPLETE analysis finished for {username} ({color})")
            return validated_analysis
        except InvalidAnalysisDataError as e:
            logger.error(f"Analysis validation failed: {e}")
            return {"error": f"Invalid analysis data: {str(e)}"}

    except Exception as e:
        logger.error(f"BULLDOZER COMPLETE analysis failed for {username}: {e}")
        return {"error": f"Analysis failed: {str(e)}"}


def _parse_pgn(pgn_text: str) -> Optional[chess.pgn.Game]:
    """Parse PGN text into chess.pgn.Game object."""
    try:
        return chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception as e:
        logger.error(f"PGN parsing failed: {e}")
        return None


def _extract_player_moves(game: chess.pgn.Game, color: str) -> Optional[List[Tuple[int, str, float]]]:
    """
    Extract moves for specific player.
    Returns: [(move_number, move_san, time_spent), ...]
    """
    try:
        moves = []
        board = game.board()
        move_number = 1

        # Get move times from comments if available
        move_times = _extract_move_times(game)

        for i, move in enumerate(game.mainline_moves()):
            is_white_move = (i % 2 == 0)

            # Only collect moves for the requested color
            if (color == 'white' and is_white_move) or (color == 'black' and not is_white_move):
                move_san = board.san(move)
                time_spent = move_times.get(i, None) if move_times else None

                moves.append((move_number if is_white_move else move_number, move_san, time_spent))

            board.push(move)
            if not is_white_move:
                move_number += 1

        return moves if moves else None

    except Exception as e:
        logger.error(f"Move extraction failed: {e}")
        return None


def _extract_move_times(game: chess.pgn.Game) -> Dict[int, float]:
    """Extract move times from PGN comments."""
    move_times = {}
    try:
        node = game
        move_index = 0

        while node.variations:
            node = node.variations[0]
            comment = node.comment

            # Look for time patterns like [%clk 0:05:23]
            if "[%clk" in comment:
                import re
                time_match = re.search(r'\[%clk (\d+):(\d+):(\d+\.?\d*)\]', comment)
                if time_match:
                    hours = int(time_match.group(1))
                    minutes = int(time_match.group(2))
                    seconds = float(time_match.group(3))
                    total_seconds = hours * 3600 + minutes * 60 + seconds
                    move_times[move_index] = total_seconds

            move_index += 1

    except Exception as e:
        logger.debug(f"Move time extraction failed: {e}")

    return move_times


def _analyze_with_stockfish(game: chess.pgn.Game, stockfish_path: str = None, depth: int = 12) -> Optional[List[Dict]]:
    """
    Run Stockfish analysis on all positions.
    Returns: [{"move_number": int, "eval_before": int, "eval_after": int, "best_move": str}, ...]
    """
    stockfish_path = stockfish_path or os.getenv("STOCKFISH_PATH", "stockfish")

    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            results = []
            board = game.board()
            move_number = 1

            for i, move in enumerate(game.mainline_moves()):
                # Evaluate position before move
                info_before = engine.analyse(board, chess.engine.Limit(depth=depth))
                eval_before = _score_to_centipawns(info_before.get('score'))
                best_move = str(info_before.get('pv', [None])[0]) if info_before.get('pv') else None

                # Make the move
                board.push(move)

                # Evaluate position after move
                info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
                eval_after = _score_to_centipawns(info_after.get('score'))

                results.append({
                    "move_number": move_number if (i % 2 == 0) else move_number,
                    "move_index": i,
                    "eval_before": eval_before,
                    "eval_after": eval_after,
                    "best_move": best_move,
                    "played_move": str(move)
                })

                if i % 2 == 1:  # After black's move
                    move_number += 1

            return results

    except Exception as e:
        logger.error(f"Stockfish analysis failed: {e}")
        return None


def _score_to_centipawns(score) -> Optional[int]:
    """Convert chess.engine score to centipawns."""
    if score is None:
        return None

    try:
        if score.is_mate():
            # Convert mate in N to high centipawn value
            mate_in = score.mate()
            return 10000 if mate_in > 0 else -10000
        else:
            return score.relative.score()
    except Exception:
        return None


def _create_moves_dataframe(game: chess.pgn.Game, username: str, color: str,
                          stockfish_path: str = None, depth: int = 12) -> Optional[pd.DataFrame]:
    """
    BULLDOZER: Create DataFrame compatible with existing analysis modules.

    This bridges the BULLDOZER architecture with the existing metric calculations
    by creating a DataFrame that the quality/timing/opening modules expect.
    """
    if not ANALYSIS_MODULES_AVAILABLE:
        logger.error("Cannot create DataFrame - pandas/analysis modules not available")
        return None

    try:
        stockfish_path = stockfish_path or os.getenv("STOCKFISH_PATH", "stockfish")

        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            moves_data = []
            board = game.board()
            move_number = 1
            player_move_count = 0

            # Extract move times from PGN
            move_times = _extract_move_times_dict(game)

            for i, move in enumerate(game.mainline_moves()):
                is_white_move = (i % 2 == 0)
                is_player_move = (color == 'white' and is_white_move) or (color == 'black' and not is_white_move)

                if is_player_move:
                    # Analyze position before player's move
                    info_before = engine.analyse(board, chess.engine.Limit(depth=depth))
                    eval_before = _score_to_centipawns(info_before.get('score'))
                    best_move = str(info_before.get('pv', [None])[0]) if info_before.get('pv') else None

                    # Get the SAN notation BEFORE making the move
                    played_san = board.san(move)

                    # Make the move
                    board.push(move)

                    # Analyze position after player's move
                    info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
                    eval_after = _score_to_centipawns(info_after.get('score'))

                    # Calculate centipawn loss
                    cp_loss = None
                    delta_eval = None
                    if eval_before is not None and eval_after is not None:
                        try:
                            # For white: eval_after should be from white's perspective
                            # For black: we need to flip the evaluation
                            if color == 'black':
                                eval_before_adj = -eval_before
                                eval_after_adj = -eval_after
                            else:
                                eval_before_adj = eval_before
                                eval_after_adj = eval_after

                            # Ensure values are finite
                            if (math.isfinite(eval_before_adj) and math.isfinite(eval_after_adj)):
                                delta_eval = max(0, eval_before_adj - eval_after_adj)  # Loss in centipawns
                                cp_loss = delta_eval
                            else:
                                # Handle infinite values (mate scores, etc.)
                                cp_loss = 0.0
                                delta_eval = 0.0
                        except (TypeError, ValueError):
                            cp_loss = 0.0
                            delta_eval = 0.0

                    # Determine game phase (simplified)
                    piece_count = len([p for p in board.piece_map().values()])
                    if move_number <= 15:
                        phase = "opening"
                    elif piece_count <= 12:
                        phase = "endgame"
                    else:
                        phase = "middlegame"

                    # Get move time
                    time_spent = move_times.get(i, None)

                    # Create move record compatible with existing analysis modules
                    move_data = {
                        'move_number': move_number if is_white_move else move_number,
                        'played': played_san,  # The move that was just made
                        'best': best_move,
                        'best_rank': 1 if str(move) == best_move else 2,  # Simplified ranking
                        'cp_loss': cp_loss,
                        'delta_eval': delta_eval,
                        'eval_cp_before': eval_before,
                        'eval_cp_after': eval_after,
                        'eval_before': eval_before,
                        'eval_after': eval_after,
                        'legal_moves': len(list(board.legal_moves)),  # Count for quality module compatibility
                        'legal_moves_list': [str(m) for m in board.legal_moves],  # List for debugging if needed
                        'phase': phase,
                        'is_engine_best': str(move) == best_move,
                        'move_time': time_spent,
                        'time_spent': time_spent,
                        'player_clock_before': None,  # Not available from PGN
                        'depth': depth
                    }

                    moves_data.append(move_data)
                    player_move_count += 1
                else:
                    # Still need to make the move to maintain board state
                    board.push(move)

                if not is_white_move:
                    move_number += 1

            if not moves_data:
                logger.error("No player moves found")
                return None

            # Create DataFrame compatible with existing modules
            df = pd.DataFrame(moves_data)
            logger.info(f"Created moves DataFrame with {len(df)} moves for {username} ({color})")
            return df

    except Exception as e:
        logger.error(f"Failed to create moves DataFrame: {e}")
        return None


def _extract_move_times_dict(game: chess.pgn.Game) -> Dict[int, float]:
    """Extract move times from PGN comments into a dict indexed by move number."""
    move_times = {}
    try:
        node = game
        move_index = 0

        while node.variations:
            node = node.variations[0]
            comment = node.comment

            # Look for time patterns like [%clk 0:05:23]
            if "[%clk" in comment:
                import re
                time_match = re.search(r'\[%clk (\d+):(\d+):(\d+\.?\d*)\]', comment)
                if time_match:
                    hours = int(time_match.group(1))
                    minutes = int(time_match.group(2))
                    seconds = float(time_match.group(3))
                    total_seconds = hours * 3600 + minutes * 60 + seconds
                    move_times[move_index] = total_seconds

            move_index += 1

    except Exception as e:
        logger.debug(f"Move time extraction failed: {e}")

    return move_times


def _calculate_complete_metrics(moves_df: pd.DataFrame, game: chess.pgn.Game,
                              username: str, color: str) -> Dict:
    """
    BULLDOZER: Calculate ALL metrics using existing analysis modules.

    This function bridges the BULLDOZER architecture with the comprehensive
    analysis capabilities, maintaining full backward compatibility.
    """
    if not ANALYSIS_MODULES_AVAILABLE:
        logger.error("Cannot calculate complete metrics - analysis modules not available")
        return {"error": "Analysis modules not available"}

    try:
        analysis = {}

        # ═══════════════════════════════════════════════════════════════════════════════
        # QUALITY METRICS (from app/analysis/quality.py)
        # ═══════════════════════════════════════════════════════════════════════════════
        try:
            quality_metrics = {}

            # Core quality metrics
            quality_metrics['acpl'] = quality.acpl(moves_df, color)
            quality_metrics['wdl_loss'] = quality.wdl_loss(moves_df, color)
            quality_metrics['robust_loss'] = quality.robust_loss(moves_df, cap_cp=1000)

            # Match rate metrics
            quality_metrics['match_rate'] = (moves_df['is_engine_best'].sum() / len(moves_df)) if len(moves_df) > 0 else 0.0
            quality_metrics['weighted_match_rate'] = quality.complexity_weighted_match(moves_df)

            # IPR calculations
            if quality_metrics['match_rate'] is not None and quality_metrics['acpl'] is not None:
                quality_metrics['ipr'] = quality.intrinsic_performance_rating(
                    quality_metrics['match_rate'], quality_metrics['acpl']
                )
                # Approximate ELO from game headers for Z-score
                elo_estimate = 1500  # Default
                try:
                    if color == 'white':
                        elo_estimate = int(game.headers.get('WhiteElo', 1500))
                    else:
                        elo_estimate = int(game.headers.get('BlackElo', 1500))
                except:
                    pass
                quality_metrics['ipr_z_score'] = quality.ipr_z_score(quality_metrics['ipr'], elo_estimate)

            # Precision analysis
            quality_metrics['precision_burst_count'] = quality.precision_bursts(moves_df)
            quality_metrics['second_choice_rate'] = quality.compute_second_choice_behavior(moves_df)

            # Phase analysis
            phase_quality = quality.phase_acpl_single(moves_df)
            quality_metrics['phase_quality'] = phase_quality
            quality_metrics['opening_acpl'] = phase_quality.get('opening_acpl')
            quality_metrics['middlegame_acpl'] = phase_quality.get('middlegame_acpl')
            quality_metrics['endgame_acpl'] = phase_quality.get('endgame_acpl')

            # Blunder analysis
            blunder_analysis = quality.phase_blunder_rate_single(moves_df)
            quality_metrics['blunder_analysis'] = blunder_analysis
            quality_metrics['blunders'] = blunder_analysis.get('total_blunders', 0)
            quality_metrics['mistakes'] = blunder_analysis.get('total_mistakes', 0)
            quality_metrics['inaccuracies'] = blunder_analysis.get('total_inaccuracies', 0)
            quality_metrics['blunder_rate'] = blunder_analysis.get('blunder_rate', 0.0)

            analysis['quality'] = quality_metrics

        except Exception as e:
            import traceback
            logger.error(f"Quality metrics calculation failed: {e}")
            logger.error(f"Quality error traceback: {traceback.format_exc()}")
            analysis['quality'] = {"error": f"Quality calculation failed: {str(e)}"}

        # ═══════════════════════════════════════════════════════════════════════════════
        # TIMING METRICS (from app/analysis/timing.py)
        # ═══════════════════════════════════════════════════════════════════════════════
        try:
            timing_metrics = {}

            # Basic timing stats
            mean_time, std_time, _ = timing.time_stats(moves_df)
            timing_metrics['mean_move_time'] = mean_time
            timing_metrics['time_variance'] = std_time ** 2 if std_time is not None else None
            timing_metrics['std_move_time'] = std_time

            # Advanced timing analysis
            timing_metrics['uniformity_score'] = timing.uniformity_score(moves_df)
            timing_metrics['lag_spike_count'] = timing.detect_lag_spikes(moves_df)
            timing_metrics['time_complexity_corr'] = timing.time_complexity_correlation(moves_df)
            timing_metrics['clutch_accuracy_diff'] = timing.clutch_accuracy(moves_df)
            timing_metrics['low_variance_flag'] = timing.low_variance_flag(moves_df)

            # Move times array
            timing_metrics['moves'] = moves_df['time_spent'].tolist()

            analysis['timing'] = timing_metrics

        except Exception as e:
            logger.error(f"Timing metrics calculation failed: {e}")
            analysis['timing'] = {"error": f"Timing calculation failed: {str(e)}"}

        # ═══════════════════════════════════════════════════════════════════════════════
        # OPENING METRICS (from app/analysis/openings.py)
        # ═══════════════════════════════════════════════════════════════════════════════
        try:
            opening_metrics = {}

            # Extract opening information from game
            opening_key = game.headers.get('ECO', 'Unknown')

            # This would typically require multiple games for proper entropy calculation
            # For single game, we provide basic metrics
            opening_metrics['opening_key'] = opening_key
            opening_metrics['novelty_depth'] = openings.novelty_ply(game, None) if hasattr(openings, 'novelty_ply') else None
            opening_metrics['second_choice_rate'] = openings.second_choice_rate(moves_df) if hasattr(openings, 'second_choice_rate') else None

            # Note: Full opening entropy requires multiple games
            opening_metrics['opening_entropy'] = None  # Requires game collection
            opening_metrics['repertoire_breadth'] = 1  # Single game = 1 opening

            analysis['opening'] = opening_metrics

        except Exception as e:
            logger.error(f"Opening metrics calculation failed: {e}")
            analysis['opening'] = {"error": f"Opening calculation failed: {str(e)}"}

        # ═══════════════════════════════════════════════════════════════════════════════
        # ENDGAME METRICS (from app/analysis/endgame.py)
        # ═══════════════════════════════════════════════════════════════════════════════
        try:
            endgame_metrics = {}

            # Endgame analysis requires tablebase access
            endgame_metrics['conversion_efficiency'] = endgame.conversion_efficiency(moves_df)
            endgame_metrics['tb_match_rate'] = None  # Requires Syzygy tablebases
            endgame_metrics['dtz_deviation'] = None  # Requires Syzygy tablebases

            analysis['endgame'] = endgame_metrics

        except Exception as e:
            logger.error(f"Endgame metrics calculation failed: {e}")
            analysis['endgame'] = {"error": f"Endgame calculation failed: {str(e)}"}

        # ═══════════════════════════════════════════════════════════════════════════════
        # MOVES DETAIL
        # ═══════════════════════════════════════════════════════════════════════════════
        try:
            moves_detail = []
            for _, row in moves_df.iterrows():
                move_detail = {
                    'move_number': row.get('move_number'),
                    'played': row.get('played'),
                    'best': row.get('best'),
                    'cp_loss': row.get('cp_loss'),
                    'eval_before': row.get('eval_before'),
                    'eval_after': row.get('eval_after'),
                    'time_spent': row.get('time_spent'),
                    'phase': row.get('phase'),
                    'legal_moves_count': row.get('legal_moves_count'),
                    'is_engine_best': row.get('is_engine_best')
                }
                moves_detail.append(move_detail)

            analysis['moves'] = moves_detail

        except Exception as e:
            logger.error(f"Moves detail creation failed: {e}")
            analysis['moves'] = []

        logger.info(f"Complete metrics calculated successfully for {username} ({color})")
        return analysis

    except Exception as e:
        logger.error(f"Complete metrics calculation failed: {e}")
        return {"error": f"Complete analysis failed: {str(e)}"}


def _calculate_all_metrics(moves_data: List[Tuple], stockfish_results: List[Dict], game: chess.pgn.Game) -> Dict:
    """
    Calculate all analysis metrics.
    BULLDOZER PRINCIPLE: Simple calculations, None if no data, error if mathematical problem.
    """
    analysis = {
        "quality": _calculate_quality_metrics(moves_data, stockfish_results),
        "timing": _calculate_timing_metrics(moves_data),
        "moves": _format_move_analysis(moves_data, stockfish_results)
    }

    return analysis


def _calculate_quality_metrics(moves_data: List[Tuple], stockfish_results: List[Dict]) -> Dict:
    """Calculate quality metrics: ACPL, match rate, blunders, etc."""
    quality = {
        "acpl": None,
        "match_rate": None,
        "blunders": 0,
        "mistakes": 0,
        "inaccuracies": 0,
        "opening_acpl": None,
        "middlegame_acpl": None,
        "endgame_acpl": None
    }

    try:
        if not stockfish_results:
            return quality

        # Calculate ACPL (Average Centipawn Loss)
        cp_losses = []
        blunders = mistakes = inaccuracies = 0

        for result in stockfish_results:
            eval_before = result.get("eval_before")
            eval_after = result.get("eval_after")

            if eval_before is not None and eval_after is not None:
                # Calculate centipawn loss (simplified)
                cp_loss = abs(eval_after - eval_before)
                cp_losses.append(cp_loss)

                # Categorize move quality
                if cp_loss >= 300:
                    blunders += 1
                elif cp_loss >= 100:
                    mistakes += 1
                elif cp_loss >= 50:
                    inaccuracies += 1

        if cp_losses:
            quality["acpl"] = sum(cp_losses) / len(cp_losses)

        quality["blunders"] = blunders
        quality["mistakes"] = mistakes
        quality["inaccuracies"] = inaccuracies

        # Simple match rate calculation
        if stockfish_results:
            matches = sum(1 for r in stockfish_results
                         if r.get("played_move") == r.get("best_move"))
            quality["match_rate"] = matches / len(stockfish_results)

    except Exception as e:
        logger.error(f"Quality metrics calculation failed: {e}")

    return quality


def _calculate_timing_metrics(moves_data: List[Tuple]) -> Dict:
    """Calculate timing metrics: mean time, variance, etc."""
    timing = {
        "mean_move_time": None,
        "time_variance": None,
        "moves": []
    }

    try:
        move_times = [time for _, _, time in moves_data if time is not None]

        if move_times:
            timing["mean_move_time"] = sum(move_times) / len(move_times)

            if len(move_times) > 1:
                mean_time = timing["mean_move_time"]
                variance = sum((t - mean_time) ** 2 for t in move_times) / len(move_times)
                timing["time_variance"] = variance

        # Include all move times (with None for missing data)
        timing["moves"] = [time for _, _, time in moves_data]

    except Exception as e:
        logger.error(f"Timing metrics calculation failed: {e}")

    return timing


def _format_move_analysis(moves_data: List[Tuple], stockfish_results: List[Dict]) -> List[Dict]:
    """Format detailed move-by-move analysis."""
    formatted_moves = []

    try:
        for i, (move_number, move_san, time_spent) in enumerate(moves_data):
            # Find corresponding stockfish result
            stockfish_data = None
            for result in stockfish_results:
                if result.get("move_index") == i * 2:  # Adjust for player color
                    stockfish_data = result
                    break

            move_data = {
                "move_number": move_number,
                "played": move_san,
                "time_spent": time_spent
            }

            if stockfish_data:
                move_data.update({
                    "best": stockfish_data.get("best_move"),
                    "cp_loss": abs(stockfish_data.get("eval_after", 0) - stockfish_data.get("eval_before", 0))
                })

            formatted_moves.append(move_data)

    except Exception as e:
        logger.error(f"Move formatting failed: {e}")

    return formatted_moves


# ══════════════════════════════════════════════════════════════════════════════
# BULLDOZER DATABASE OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _clean_nan_for_json(obj):
    """
    Recursively convert NaN and Infinity values to None for JSON serialization.
    PostgreSQL JSON type doesn't support NaN/Infinity.
    """
    import math

    if isinstance(obj, dict):
        return {k: _clean_nan_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_nan_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj


def save_analysis_to_db(pgn_text: str, username: str, color: str,
                       analysis: Dict, session) -> Optional[int]:
    """
    Save analysis to BULLDOZER single table.
    Returns: analysis_id if successful, None if failed
    """
    try:
        from app.models import GameAnalysis

        # Extract game metadata from PGN
        game = _parse_pgn(pgn_text)
        if not game:
            logger.error("Cannot save analysis: PGN parsing failed")
            return None

        headers = game.headers

        # Clean NaN values for JSON storage
        clean_analysis = _clean_nan_for_json(analysis)

        # Create single record with everything
        game_analysis = GameAnalysis(
            pgn=pgn_text,
            white_username=headers.get("White"),
            black_username=headers.get("Black"),
            white_elo=int(headers.get("WhiteElo", 0)) or None,
            black_elo=int(headers.get("BlackElo", 0)) or None,
            time_control=headers.get("TimeControl"),
            game_date=headers.get("Date"),
            termination=headers.get("Termination"),
            analyzed_username=username,
            analyzed_color=color,
            analyzed_at=datetime.now(timezone.utc),
            moves_analyzed=len(analysis.get("moves", [])),
            analysis=clean_analysis
        )

        session.add(game_analysis)
        session.flush()  # Get the ID

        logger.info(f"Analysis saved to DB with ID {game_analysis.id}")
        return game_analysis.id

    except Exception as e:
        logger.error(f"Failed to save analysis to DB: {e}")
        return None


def get_player_analysis_summary(username: str, session) -> Dict:
    """
    Get aggregated analysis summary for a player.
    BULLDOZER: Complete aggregation including longitudinal metrics.
    """
    try:
        from app.models import GameAnalysis
        from sqlmodel import select

        # Get all analyses for the player
        stmt = select(GameAnalysis).where(GameAnalysis.analyzed_username == username)
        analyses = session.exec(stmt).all()

        if not analyses:
            return {"error": f"No analyses found for {username}"}

        # Convert to pandas DataFrame for longitudinal analysis
        if ANALYSIS_MODULES_AVAILABLE:
            return _calculate_complete_player_summary(analyses, username)
        else:
            return _calculate_simple_player_summary(analyses, username)

    except Exception as e:
        logger.error(f"Failed to get player summary: {e}")
        return {"error": f"Summary calculation failed: {str(e)}"}


def _calculate_simple_player_summary(analyses, username: str) -> Dict:
    """Simple aggregation when analysis modules are not available."""
    total_games = len(analyses)
    acpl_values = []
    total_blunders = 0
    total_mistakes = 0

    for analysis in analyses:
        quality = analysis.analysis.get("quality", {})
        if quality.get("acpl") is not None:
            acpl_values.append(quality["acpl"])
        total_blunders += quality.get("blunders", 0)
        total_mistakes += quality.get("mistakes", 0)

    summary = {
        "username": username,
        "games_analyzed": total_games,
        "avg_acpl": sum(acpl_values) / len(acpl_values) if acpl_values else None,
        "total_blunders": total_blunders,
        "total_mistakes": total_mistakes,
        "blunder_rate": total_blunders / total_games if total_games > 0 else None,
        "first_analysis": min(a.analyzed_at for a in analyses),
        "last_analysis": max(a.analyzed_at for a in analyses)
    }

    return summary


def _calculate_complete_player_summary(analyses, username: str) -> Dict:
    """
    BULLDOZER: Complete player summary with ALL longitudinal metrics.

    This implements all the metrics from docs/metricas_actuales matching
    the exact structure expected by the frontend and API.
    """
    try:
        # Create DataFrame from all game analyses
        games_data = []
        successful_analyses = 0
        failed_analyses = 0

        for analysis in analyses:
            quality = analysis.analysis.get("quality", {})
            timing = analysis.analysis.get("timing", {})
            opening = analysis.analysis.get("opening", {})

            # Skip analyses that have critical errors (like the old robust_loss error)
            if 'error' in quality and 'robust_loss() got an unexpected keyword argument' in quality.get('error', ''):
                failed_analyses += 1
                logger.debug(f"Skipping analysis with critical error: {quality['error']}")
                continue

            # Debug what we actually have in the analysis
            if successful_analyses < 3:  # Only log first few for debugging
                logger.info(f"DEBUG Analysis structure: quality keys = {list(quality.keys())}")
                logger.info(f"DEBUG Quality values: acpl={quality.get('acpl')}, wdl_loss={quality.get('wdl_loss')}, match_rate={quality.get('match_rate')}")

            # Skip analyses where ALL quality metrics are missing
            # But allow analyses with some metrics even if others failed
            has_acpl = quality.get('acpl') is not None
            has_wdl = quality.get('wdl_loss') is not None
            has_match_rate = quality.get('match_rate') is not None

            if not has_acpl and not has_wdl and not has_match_rate:
                failed_analyses += 1
                logger.debug(f"Skipping analysis with no quality metrics")
                continue

            successful_analyses += 1

            game_record = {
                # Basic info
                "analyzed_at": analysis.analyzed_at,
                "game_date": analysis.game_date,

                # Quality metrics
                "acpl": quality.get("acpl"),
                "match_rate": quality.get("match_rate"),
                "ipr": quality.get("ipr"),
                "wdl_loss": quality.get("wdl_loss"),
                "robust_loss": quality.get("robust_loss"),
                "blunders": quality.get("blunders", 0),
                "mistakes": quality.get("mistakes", 0),
                "precision_burst_count": len(quality.get("precision_burst_count", [])) if isinstance(quality.get("precision_burst_count"), list) else quality.get("precision_burst_count", 0),

                # Phase quality
                "opening_acpl": quality.get("opening_acpl"),
                "middlegame_acpl": quality.get("middlegame_acpl"),
                "endgame_acpl": quality.get("endgame_acpl"),

                # Timing
                "mean_move_time": timing.get("mean_move_time"),
                "time_variance": timing.get("time_variance"),
                "uniformity_score": timing.get("uniformity_score"),
                "clutch_accuracy_diff": timing.get("clutch_accuracy_diff"),
                "lag_spike_count": len(timing.get("lag_spike_count", [])) if isinstance(timing.get("lag_spike_count"), list) else timing.get("lag_spike_count", 0),

                # Opening
                "opening_key": opening.get("opening_key"),
                "second_choice_rate": quality.get("second_choice_rate")
            }
            games_data.append(game_record)

        games_df = pd.DataFrame(games_data)

        logger.info(f"Player {username}: {successful_analyses} successful analyses, {failed_analyses} failed analyses")

        if games_df.empty:
            return {"error": f"No valid game data found. {failed_analyses} analyses had errors."}

        # ═══════════════════════════════════════════════════════════════════════════════
        # BASIC AGGREGATIONS
        # ═══════════════════════════════════════════════════════════════════════════════

        summary = {
            "username": username,
            "games_analyzed": len(games_df),
            "first_game_date": min(a.analyzed_at for a in analyses),
            "last_game_date": max(a.analyzed_at for a in analyses),
            "analyzed_at": datetime.now(timezone.utc)
        }

        # ═══════════════════════════════════════════════════════════════════════════════
        # QUALITY METRICS AGGREGATION
        # ═══════════════════════════════════════════════════════════════════════════════

        # Basic quality
        acpl_values = games_df['acpl'].dropna()
        match_rate_values = games_df['match_rate'].dropna()
        ipr_values = games_df['ipr'].dropna()

        summary.update({
            "avg_acpl": float(acpl_values.mean()) if not acpl_values.empty else 0.0,
            "std_acpl": float(acpl_values.std()) if not acpl_values.empty else 0.0,
            "avg_match_rate": float(match_rate_values.mean()) if not match_rate_values.empty else 0.0,
            "std_match_rate": float(match_rate_values.std()) if not match_rate_values.empty else 0.0,
            "avg_ipr": float(ipr_values.mean()) if not ipr_values.empty else 0.0,
        })

        # Robust loss calculation
        robust_loss_values = games_df['robust_loss'].dropna()
        if not robust_loss_values.empty:
            summary["robust_loss"] = float(robust_loss_values.mean())
        else:
            summary["robust_loss"] = 0.0

        # WDL loss
        wdl_loss_values = games_df['wdl_loss'].dropna()
        if not wdl_loss_values.empty:
            summary["avg_wdl_loss"] = float(wdl_loss_values.mean())
        else:
            summary["avg_wdl_loss"] = 0.0

        # ═══════════════════════════════════════════════════════════════════════════════
        # LONGITUDINAL ANALYSIS (ROI, TRENDS, PATTERNS)
        # ═══════════════════════════════════════════════════════════════════════════════

        # Set defaults first
        summary.update({
            "step_function_detected": False,
            "step_function_magnitude": 0.0,
            "longest_streak": 0,
            "selectivity_score": 0.0,
            "trend_acpl": 0.0,
            "trend_match_rate": 0.0
        })

        try:
            # ROI calculations using existing longitudinal module
            roi_metrics = longitudinal.aggregate_roi(games_df)
            summary.update({
                "roi_mean": float(roi_metrics.get("roi_mean", 0.0)),
                "roi_max": float(roi_metrics.get("roi_max", 0.0)),
                "roi_std": float(roi_metrics.get("roi_std", 0.0)) if roi_metrics.get("roi_std") else None,
                "roi_curve": roi_metrics.get("roi_curve", [])
            })

            # Step function detection (use keyword arguments only)
            step_function_result = longitudinal.detect_step_function(games_df)
            summary.update({
                "step_function_detected": bool(step_function_result.get("detected", False)),
                "step_function_magnitude": float(step_function_result.get("magnitude", 0.0))
            })

            # Longest streak
            if 'ipr' in games_df.columns:
                roi_series = games_df['ipr'].dropna()
                summary["longest_streak"] = int(longitudinal.longest_streak(roi_series))

            # Trends
            trend_metrics = longitudinal.compute_trends(games_df)
            summary.update({
                "trend_acpl": float(trend_metrics.get("trend_acpl", 0.0)) if trend_metrics.get("trend_acpl") else None,
                "trend_match_rate": float(trend_metrics.get("trend_match_rate", 0.0)) if trend_metrics.get("trend_match_rate") else None
            })

            # Selectivity score (use keyword arguments only)
            selectivity_result = longitudinal.selectivity_score(games_df)
            summary["selectivity_score"] = float(selectivity_result.get("selectivity_score", 0.0))

        except Exception as e:
            logger.warning(f"Longitudinal analysis failed: {e}")

        # ═══════════════════════════════════════════════════════════════════════════════
        # PEER COMPARISON
        # ═══════════════════════════════════════════════════════════════════════════════

        # Set defaults first
        summary.update({
            "peer_delta_acpl": 0.0,
            "peer_delta_match": 0.0
        })

        try:
            # Peer comparison requires reference data - disable for single player analysis
            # peer_deltas = longitudinal.peer_group_delta(games_df, reference_df, elo_col='ipr')
            summary.update({
                "peer_delta_acpl": 0.0,  # Placeholder - requires population reference data
                "peer_delta_match": 0.0  # Placeholder - requires population reference data
            })
        except Exception as e:
            logger.warning(f"Peer comparison failed: {e}")

        # ═══════════════════════════════════════════════════════════════════════════════
        # TIME PATTERNS
        # ═══════════════════════════════════════════════════════════════════════════════

        time_metrics = {}
        timing_cols = ['mean_move_time', 'time_variance', 'uniformity_score', 'clutch_accuracy_diff']
        for col in timing_cols:
            if col in games_df.columns:
                values = games_df[col].dropna()
                if not values.empty:
                    time_metrics[col] = float(values.mean())

        # Add lag_spike_count as sum (not average) since it's a count
        if 'lag_spike_count' in games_df.columns:
            total_lag_spikes = 0
            for lag_spike in games_df['lag_spike_count'].dropna():
                if isinstance(lag_spike, list):
                    total_lag_spikes += len(lag_spike)
                elif isinstance(lag_spike, (int, float)):
                    total_lag_spikes += int(lag_spike)
            time_metrics['lag_spike_count'] = total_lag_spikes

        summary["time_patterns"] = time_metrics
        # Frontend compatibility alias
        summary["time_management"] = time_metrics

        # ═══════════════════════════════════════════════════════════════════════════════
        # OPENING PATTERNS
        # ═══════════════════════════════════════════════════════════════════════════════

        opening_patterns = {
            "mean_entropy": 0.0,
            "novelty_depth": 8.0,  # Default average novelty depth
            "opening_breadth": 0,
            "second_choice_rate": None
        }

        # Calculate opening entropy (requires multiple games)
        if 'opening_key' in games_df.columns:
            opening_counts = games_df['opening_key'].value_counts()
            if len(opening_counts) > 1:
                # Shannon entropy calculation
                probs = opening_counts / opening_counts.sum()
                opening_patterns["mean_entropy"] = float(-(probs * np.log2(probs)).sum())
            else:
                opening_patterns["mean_entropy"] = 0.0

            opening_patterns["opening_breadth"] = int(len(opening_counts))

            # Favorite openings - convert to list format expected by API
            top_openings = opening_counts.head(5)
            summary["favorite_openings"] = [
                {"eco_code": "???", "name": str(opening), "count": int(count)}
                for opening, count in top_openings.items()
            ]
        else:
            summary["favorite_openings"] = []

        # Second choice rate - extract numeric value from dict
        if 'second_choice_rate' in games_df.columns:
            scr_values = []
            for scr in games_df['second_choice_rate'].dropna():
                if isinstance(scr, dict):
                    # Extract the main second_choice_rate value from the dict
                    main_scr = scr.get('second_choice_rate')
                    if main_scr is not None:
                        scr_values.append(main_scr)
                elif isinstance(scr, (int, float)):
                    scr_values.append(scr)

            if scr_values:
                opening_patterns["second_choice_rate"] = float(sum(scr_values) / len(scr_values))

        summary["opening_patterns"] = opening_patterns

        # ═══════════════════════════════════════════════════════════════════════════════
        # PHASE QUALITY
        # ═══════════════════════════════════════════════════════════════════════════════

        phase_quality = {}
        phase_cols = ['opening_acpl', 'middlegame_acpl', 'endgame_acpl']
        for col in phase_cols:
            if col in games_df.columns:
                values = games_df[col].dropna()
                if not values.empty:
                    phase_quality[col] = values.mean()

        # Blunder rates
        total_blunders = int(games_df['blunders'].sum())
        total_mistakes = int(games_df['mistakes'].sum())
        phase_quality.update({
            "blunder_rate": float(total_blunders) / len(games_df) if len(games_df) > 0 else 0.0,
            "total_blunders": total_blunders,
            "total_mistakes": total_mistakes
        })

        summary["phase_quality"] = phase_quality

        # ═══════════════════════════════════════════════════════════════════════════════
        # TACTICAL ANALYSIS
        # ═══════════════════════════════════════════════════════════════════════════════

        tactical = {}
        if 'precision_burst_count' in games_df.columns:
            # precision_burst_count can be a list or integer, convert to count
            total_precision_bursts = 0
            for pbc in games_df['precision_burst_count'].dropna():
                if isinstance(pbc, list):
                    total_precision_bursts += len(pbc)
                elif isinstance(pbc, (int, float)):
                    total_precision_bursts += int(pbc)
            tactical["precision_burst_count"] = total_precision_bursts

        summary["tactical"] = tactical

        # ═══════════════════════════════════════════════════════════════════════════════
        # CLUTCH ACCURACY (Frontend compatibility)
        # ═══════════════════════════════════════════════════════════════════════════════

        clutch_accuracy = {}
        if 'clutch_accuracy_diff' in games_df.columns:
            clutch_values = games_df['clutch_accuracy_diff'].dropna()
            if not clutch_values.empty:
                clutch_accuracy["avg_clutch_diff"] = float(clutch_values.mean())
            else:
                clutch_accuracy["avg_clutch_diff"] = 0.0
        else:
            clutch_accuracy["avg_clutch_diff"] = 0.0

        summary["clutch_accuracy"] = clutch_accuracy

        # ═══════════════════════════════════════════════════════════════════════════════
        # ENDGAME EFFICIENCY (Frontend compatibility)
        # ═══════════════════════════════════════════════════════════════════════════════

        # Aggregate endgame metrics from individual game analyses
        endgame_efficiency = {}

        # Check if we have endgame data in any of the analyses
        conversion_values = []
        for analysis in analyses:
            endgame_data = analysis.analysis.get("endgame", {})
            if endgame_data and "conversion_efficiency" in endgame_data:
                conv_eff = endgame_data["conversion_efficiency"]
                if conv_eff is not None and not (isinstance(conv_eff, float) and np.isnan(conv_eff)):
                    conversion_values.append(conv_eff)

        if conversion_values:
            endgame_efficiency["conversion_efficiency"] = int(np.mean(conversion_values))
        else:
            endgame_efficiency["conversion_efficiency"] = None

        endgame_efficiency["tb_match_rate"] = None  # Requires Syzygy tablebases
        endgame_efficiency["dtz_deviation"] = None  # Requires Syzygy tablebases

        summary["endgame"] = endgame_efficiency

        # ═══════════════════════════════════════════════════════════════════════════════
        # RISK ASSESSMENT (Simplified)
        # ═══════════════════════════════════════════════════════════════════════════════

        risk_factors = {}
        risk_score = 0

        # Low ACPL risk
        if summary.get("avg_acpl") and summary["avg_acpl"] < 25:
            risk_factors["low_acpl"] = True
            risk_score += 20

        # High match rate risk
        if summary.get("avg_match_rate") and summary["avg_match_rate"] > 0.8:
            risk_factors["high_match_rate"] = True
            risk_score += 15

        # Step function risk
        if summary.get("step_function_detected"):
            risk_factors["step_function"] = True
            risk_score += 25

        # Uniformity risk
        uniformity_scores = games_df['uniformity_score'].dropna()
        if not uniformity_scores.empty and uniformity_scores.mean() > 0.8:
            risk_factors["high_uniformity"] = True
            risk_score += 20

        summary["risk"] = {
            "risk_score": min(risk_score, 100),
            "risk_factors": risk_factors,
            "confidence_level": len(games_df),  # More games = more confidence
            "suspicious_games_count": len(games_df[games_df.get('acpl', float('inf')) < 15]) if 'acpl' in games_df.columns else 0
        }

        # ═══════════════════════════════════════════════════════════════════════════════
        # PERFORMANCE OBJECT (for UI compatibility)
        # ═══════════════════════════════════════════════════════════════════════════════
        summary["performance"] = {
            "trend_acpl": summary.get("trend_acpl"),
            "trend_match_rate": summary.get("trend_match_rate"),
            "roi_curve": summary.get("roi_curve", [])
        }

        # ═══════════════════════════════════════════════════════════════════════════════
        # BENCHMARK (Placeholder - requires population reference data)
        # ═══════════════════════════════════════════════════════════════════════════════

        # Calculate rough percentiles based on ACPL (placeholder until we have population data)
        avg_acpl = summary.get("avg_acpl", 0)
        if avg_acpl <= 30:
            percentile_acpl = 95  # Excellent
        elif avg_acpl <= 50:
            percentile_acpl = 85  # Very good
        elif avg_acpl <= 100:
            percentile_acpl = 70  # Good
        elif avg_acpl <= 200:
            percentile_acpl = 50  # Average
        elif avg_acpl <= 400:
            percentile_acpl = 30  # Below average
        else:
            percentile_acpl = 10  # Poor

        # Get existing benchmark or create new one
        existing_benchmark = summary.get("benchmark", {})

        summary["benchmark"] = {
            "percentile_acpl": existing_benchmark.get("percentile_acpl", percentile_acpl),
            "percentile_match_rate": 50,  # Placeholder
            "percentile_entropy": existing_benchmark.get("percentile_entropy"),  # Preserve if exists
            "elo_estimate": summary.get("avg_ipr", 1500),  # Use IPR as ELO estimate
            "strength_category": "Intermediate" if avg_acpl <= 200 else "Beginner"
        }

        # Final frontend compatibility fixes (must be last to avoid overwrite)
        if "benchmark" in summary:
            summary["benchmark"]["percentile_match_rate"] = 50  # Frontend compatibility
        else:
            summary["benchmark"] = {"percentile_acpl": 50, "percentile_match_rate": 50}

        logger.info(f"Complete player summary calculated for {username}: {len(games_df)} games analyzed")
        return summary

    except Exception as e:
        logger.error(f"Complete player summary calculation failed: {e}")
        return {"error": f"Complete summary calculation failed: {str(e)}"}