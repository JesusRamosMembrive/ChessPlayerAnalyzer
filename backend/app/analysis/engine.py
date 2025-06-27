# app/analysis/engine.py
"""
Motor principal de análisis que orquesta todos los módulos.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Dict, Optional
import pandas as pd
import chess.pgn
import io

from pathlib import Path

from app.models import Game, GameAnalysisDetailed, PlayerAnalysisDetailed
from app.database import engine as db_engine
from sqlmodel import Session, select

# Importar módulos de análisis
from . import quality
from . import timing
from . import openings
from . import endgame
from . import longitudinal
from app import models
from app.database import engine
from app.analysis.openings import aggregate_player_opening_patterns
from app.utils import clean_json_numbers
from app.analysis.eco_table import ECO_NAMES
from app.analysis.longitudinal import compute_trends
from app.analysis.quality import compute_phase_quality
import numpy as np
from app.analysis.benchmark import compute_benchmark
from app.analysis.timing import aggregate_time_management
from app.analysis.quality import aggregate_clutch_accuracy
from app.analysis.quality import aggregate_tactical_trends
from app.analysis.endgame import aggregate_endgame_efficiency

from app.analysis.quality import (
    aggregate_tactical_trends,
    aggregate_blunders_by_phase,      # nuevo
)
from app.analysis.timing import (
    aggregate_time_management,
    aggregate_time_complexity_corr,   # nuevo
)
from app.utils_sanitize import clean_json_numbers


logger = logging.getLogger(__name__)

def prepare_moves_dataframe(game: models.Game, username: Optional[str] = None) -> pd.DataFrame:
    rows = []
    player_color = None
    
    if username:
        if game.white_username == username:
            player_color = 'white'
        elif game.black_username == username:
            player_color = 'black'
    
    # ── Calcular tiempo restante en el reloj ──────────────────────────────
    # Asumimos tiempo inicial de 600 segundos (10 minutos) para partidas rápidas
    initial_time = 600.0  # 10 minutos en segundos
    
    for i, m in enumerate(game.moves):
        if player_color and (i % 2 == 0) != (player_color == 'white'):
            continue
            
        # Calcular el tiempo restante antes de este movimiento
        current_clock = initial_time
        
        # Si tenemos move_times, calcular el tiempo restante real
        if game.move_times and len(game.move_times) > 0:
            # Calcular el tiempo acumulado hasta este punto para el jugador
            accumulated_time = 0.0
            for j, time_change in enumerate(game.move_times):
                # Solo contar movimientos del jugador actual
                if (player_color == 'white' and j % 2 == 0) or (player_color == 'black' and j % 2 == 1):
                    # Solo contar hasta el movimiento actual (i)
                    if j < i:
                        accumulated_time += abs(time_change)
            
            current_clock = max(0.0, initial_time - accumulated_time)
            
        rows.append({
            "move_number": m.move_number,
            "played"     : m.played,
            "best_rank"  : m.best_rank,
            "cp_loss"    : m.cp_loss,
            "eval_cp_before": m.eval_before,
            "eval_cp_after" : m.eval_after,
            "move_time"     : m.time_spent or 0,
            "legal_moves"   : m.legal_moves_count or 0,   # ← SIEMPRE entero
            "player_clock_before": current_clock,  # Tiempo real en el reloj antes del movimiento
            "is_engine_best": m.best_rank == 0,
            "player_color": player_color,
        })
    df = pd.DataFrame(rows)

    # ── NUEVO · etiquetar la fase de cada jugada ──────────────────
    total_plies = len(df)                       # nº medias-jugadas
    opening_cut = int(total_plies * 0.25)       # 0-25 %  → opening
    endgame_cut = int(total_plies * 0.80)       # 80-100 %→ endgame

    df["phase"] = np.select(
        [
            df.index <= opening_cut,
            df.index >= endgame_cut,
        ],
        ["opening", "endgame"],
        default="middlegame",
    )

    # ── NUEVO · crear delta_eval si falta ────────────────────────────
    if "delta_eval" not in df.columns:
        if "cp_loss" in df.columns:
            df["delta_eval"] = df["cp_loss"]
        elif {"eval_cp_before", "eval_cp_after"}.issubset(df.columns):
            df["delta_eval"] = (df.eval_cp_before - df.eval_cp_after).abs()
        else:
            df["delta_eval"] = np.nan  # dejar NaN si no hay datos
    # ─────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────
    # Garantizar la presencia de las columnas clave
    # ─────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────
    # Normalizar esquema: asegurar columnas requeridas
    # ──────────────────────────────────────────────────────────────
    REQUIRED_COLS = {
        # nombre      → valor por defecto
        "played": "",
        "best": "",
        "best_rank": np.nan,
        "cp_loss": np.nan,
        "eval_cp_before": np.nan,
        "eval_cp_after": np.nan,
        "is_engine_best": False,
        "legal_moves": np.nan,
        "move_time": np.nan,
    }

    # alias/compat: eval_before/eval_after vienen de versiones viejas
    if "eval_before" in df.columns and "eval_cp_before" not in df.columns:
        df["eval_cp_before"] = df["eval_before"]
    if "eval_after" in df.columns and "eval_cp_after" not in df.columns:
        df["eval_cp_after"] = df["eval_after"]

    # derivar eval_cp_after con cp_loss si aún falta
    if "eval_cp_after" not in df.columns and {
        "eval_cp_before", "cp_loss"
    }.issubset(df.columns):
        df["eval_cp_after"] = df.eval_cp_before - df.cp_loss

    # derivar is_engine_best si falta
    if "is_engine_best" not in df.columns and "best_rank" in df.columns:
        df["is_engine_best"] = df.best_rank.eq(0)

    # finalmente, crea los que sigan faltando con su valor por defecto
    for col, default in REQUIRED_COLS.items():
        if col not in df.columns:
            df[col] = default

    # ── BLINDAJE final: si no existe la columna, créala a cero ──────────
    if "legal_moves" not in df:
        df = df.assign(legal_moves=0)

    return df



class ChessAnalysisEngine:
    """Motor principal que coordina todos los análisis."""

    def __init__(self,
                 reference_book_path: Optional[Path] = None,
                 tablebase_path: Optional[Path] = None,
                 reference_stats_df: Optional[pd.DataFrame] = None):
        """
        Args:
            reference_book_path: Ruta al libro de aperturas Polyglot
            tablebase_path: Ruta a las tablebases Syzygy
            reference_stats_df: DataFrame con estadísticas de referencia por ELO
        """
        self.reference_book = reference_book_path
        self.tablebase_path = tablebase_path
        self.reference_stats = reference_stats_df
        self.acpl_model = None

        # Entrenar modelo ACPL si hay datos de referencia
        if reference_stats_df is not None and 'elo' in reference_stats_df.columns:
            self.acpl_model = quality.ACPLModel()
            self.acpl_model.fit(reference_stats_df)

    def _get_player_color(self, game: Game, username: str) -> str | None:
        """Determine if username played as 'white' or 'black' in this game."""
        if game.white_username == username:
            return 'white'
        elif game.black_username == username:
            return 'black'
        return None

    def analyze_game(self, game_id: int, username: str) -> GameAnalysisDetailed:
        """
        Analiza una partida completa con todos los módulos.

        Args:
            game_id: ID de la partida en la BD

        Returns:
            GameAnalysisDetailed con todas las métricas
        """
        logger.info(f"DEBUG ENGINE: Starting analyze_game for game_id: {game_id}")

        with Session(db_engine) as session:
            # Cargar partida y movimientos
            game = session.get(Game, game_id)
            if not game:
                raise ValueError(f"Game {game_id} not found")
            
            logger.info(f"DEBUG ENGINE: Loaded game - white: {game.white_username}, black: {game.black_username}")
            logger.info(f"DEBUG ENGINE: Game metadata - eco_code: {game.eco_code}, opening: {game.opening_key}")

            player_color = self._get_player_color(game, username)
            if not player_color:
                raise ValueError(f"Player {username} not found in game {game_id}")

            # Preparar DataFrames para análisis
            moves_df = self.prepare_moves_dataframe(game, username)
            logger.info(f"DEBUG ENGINE: Prepared moves DataFrame - shape: {moves_df.shape}")
            logger.info(f"DEBUG ENGINE: Moves DataFrame columns: {list(moves_df.columns)}")
            logger.info(f"DEBUG ENGINE: Sample moves data:\n{moves_df.head(3).to_string()}")

            # Parsear PGN para análisis que lo requieren
            pgn_game = chess.pgn.read_game(io.StringIO(game.pgn))
            logger.info("DEBUG ENGINE: Parsed PGN game successfully")

            # 1. MÉTRICAS DE CALIDAD
            quality_features = self._analyze_quality(moves_df, game, player_color)
            logger.info("DEBUG ENGINE: Starting quality analysis")
            logger.info(f"DEBUG ENGINE: Quality features: {quality_features}")

            # 2. MÉTRICAS DE TIEMPO
            logger.info("DEBUG ENGINE: Starting timing analysis")
            timing_features = timing.aggregate_time_features(moves_df)
            logger.info(f"DEBUG ENGINE: Timing features: {timing_features}")

            # 3. MÉTRICAS DE APERTURA (si es aplicable)
            opening_features = {}
            if self.reference_book:
                logger.info("DEBUG ENGINE: Starting opening analysis with reference book")
                # Necesitamos múltiples partidas del jugador para entropía
                games_df = self._get_player_games_df(username, session)

                logger.info(f"DEBUG ENGINE: Player games DataFrame shape: {games_df.shape}")
                opening_features = openings.aggregate_opening_features(
                    game.opening_key or "",
                    game.eco_code,
                    moves_df,
                    games_df
                )
                logger.info(f"DEBUG ENGINE: Opening features: {opening_features}")
            else:
                logger.info("DEBUG ENGINE: Skipping opening analysis - no reference book")

            # 4. MÉTRICAS DE FINAL (si hay tablebases)
            endgame_features = {}
            if self.tablebase_path and self._has_endgame(moves_df):
                logger.info("DEBUG ENGINE: Starting endgame analysis with tablebases")
                endgame_features = endgame.aggregate_endgame_features(
                    pgn_game, moves_df, self.tablebase_path
                )
                logger.info(f"DEBUG ENGINE: Endgame features: {endgame_features}")
            else:
                logger.info("DEBUG ENGINE: Skipping endgame analysis - no tablebases or not endgame")

            # 5. COMBINAR TODAS LAS FEATURES
            all_features = {
                **quality_features,
                **timing_features,
                **opening_features,
                **endgame_features
            }
            logger.info(f"DEBUG ENGINE: Combined features count: {len(all_features)}")
            logger.info(f"DEBUG ENGINE: All features: {all_features}")

            # 6. CALCULAR FLAGS DE SOSPECHA
            logger.info("DEBUG ENGINE: Calculating suspicious flags")
            suspicious_flags = self._calculate_suspicious_flags(all_features)
            logger.info(f"DEBUG ENGINE: Suspicious flags: {suspicious_flags}")

            # 7. CREAR REGISTRO DE ANÁLISIS
            analysis = GameAnalysisDetailed(
                game_id=game_id,
                # Quality
                acpl=all_features.get('acpl', 0),
                match_rate=all_features.get('match_rate', 0),
                weighted_match_rate=all_features.get('match_weighted', 0),
                ipr=all_features.get('ipr', 0),
                ipr_z_score=all_features.get('ipr_z', 0),
                # Timing
                mean_move_time=all_features.get('mean_time', 0),
                time_variance=all_features.get('std_time', 0),
                time_complexity_corr=all_features.get('time_complexity_corr', 0),
                lag_spike_count=all_features.get('lag_spike_count', 0),
                uniformity_score=all_features.get('uniformity_score', 0),
                # Opening
                opening_entropy=all_features.get('H_opening', 0),
                novelty_depth=all_features.get('mean_tn_depth', 0),
                second_choice_rate=all_features.get('second_choice_pct', 0),
                # Endgame
                tb_match_rate=all_features.get('tb_match_pct'),
                conversion_efficiency=all_features.get('conversion_moves'),
                # Flags
                **suspicious_flags
            )

            session.add(analysis)
            session.commit()

            logger.info(f"Análisis detallado completado para partida {game_id}")
            logger.info("GameAnalysisDetailed result: %s", analysis)
            return analysis

    def analyze_player(self, username: str) -> PlayerAnalysisDetailed:
        """
        Analiza todas las partidas de un jugador y genera métricas agregadas.

        Args:
            username: Nombre del jugador

        Returns:
            PlayerAnalysisDetailed con métricas longitudinales
        """
        logger.info(f"DEBUG ENGINE: Starting analyze_player for {username}")

        with Session(db_engine) as session:
            # ── 1. Recuperar todas las partidas + sus análisis ───────────────
            games_df = self._get_player_games_with_analysis(username, session)

            if not games_df.empty:
                from app.analysis.longitudinal import roi_per_game
                logger.info(f"DEBUG ENGINE: Games DataFrame columns before ROI: {list(games_df.columns)}")
                games_df['roi'] = roi_per_game(games_df)
                logger.info(f"DEBUG ENGINE: Added ROI column, shape: {games_df.shape}")
                logger.info(f"DEBUG ENGINE: Before column handling - columns: {list(games_df.columns)}")
                logger.info(f"DEBUG ENGINE: 'date' in columns: {'date' in games_df.columns}")
                if 'date' in games_df.columns:
                    logger.info("DEBUG ENGINE: Dropping existing 'date' column and renaming 'created_at' to 'date'")
                    games_df_for_trends = games_df.drop(columns=['date']).rename(columns={'created_at': 'date'})
                else:
                    logger.info("DEBUG ENGINE: No existing 'date' column, just renaming 'created_at' to 'date'")
                    games_df_for_trends = games_df.rename(columns={'created_at': 'date'})
                logger.info(f"DEBUG ENGINE: After column handling - columns: {list(games_df_for_trends.columns)}")
                logger.info(f"DEBUG ENGINE: Checking for duplicate 'date' columns: {games_df_for_trends.columns.duplicated().any()}")
                if games_df_for_trends.columns.duplicated().any():
                    logger.error(f"DEBUG ENGINE: Found duplicate columns: {games_df_for_trends.columns[games_df_for_trends.columns.duplicated()].tolist()}")
                    games_df_for_trends = games_df_for_trends.loc[:, ~games_df_for_trends.columns.duplicated()]
                    logger.info(f"DEBUG ENGINE: After removing duplicates - columns: {list(games_df_for_trends.columns)}")
                try:
                    logger.info(f"DEBUG ENGINE: About to call compute_trends with DataFrame shape: {games_df_for_trends.shape}")
                    logger.info(f"DEBUG ENGINE: DataFrame columns before compute_trends: {list(games_df_for_trends.columns)}")
                    logger.info(f"DEBUG ENGINE: Sample data - acpl: {games_df_for_trends['acpl'].head().tolist()}")
                    logger.info(f"DEBUG ENGINE: Sample data - match_rate: {games_df_for_trends['match_rate'].head().tolist()}")
                    logger.info(f"DEBUG ENGINE: Sample data - roi: {games_df_for_trends['roi'].head().tolist()}")
                    trend_feats = compute_trends(games_df_for_trends)
                    logger.info(f"DEBUG ENGINE: compute_trends result: {trend_feats}")
                    if not trend_feats:
                        logger.warning(f"DEBUG ENGINE: compute_trends returned empty dict. DataFrame shape: {games_df_for_trends.shape}, columns: {list(games_df_for_trends.columns)}")
                except Exception as e:
                    logger.error(f"DEBUG ENGINE: compute_trends failed with error: {e}")
                    import traceback
                    logger.error(f"DEBUG ENGINE: Full traceback: {traceback.format_exc()}")
                    trend_feats = {}
            else:
                trend_feats = {}
                logger.info("DEBUG ENGINE: Empty games_df, setting empty trend_feats")

            if games_df.empty:
                raise ValueError(f"No analyzed games found for {username}")

            # ### NEW · 1-bis  ── DataFrames de movimientos de *cada* partida
            moves_dfs = []
            for gid in games_df["game_id"]:
                game_obj = session.get(Game, gid)
                mv_df = self.prepare_moves_dataframe(game_obj, username)

                # ── NUEVO: marcar fase para cada jugada ──────────────────────────
                total_plies = len(mv_df)  # nº medias-jugadas
                opening_cut = int(total_plies * 0.25)  # 0–25 %  → opening
                endgame_cut = int(total_plies * 0.80)  # 80–100 % → endgame

                mv_df["phase"] = np.select(
                    [
                        mv_df.index <= opening_cut,
                        mv_df.index >= endgame_cut,
                    ],
                    ["opening", "endgame"],
                    default="middlegame",
                )
                # ─────────────────────────────────────────────────────────────────

                moves_dfs.append(mv_df)

            logger.info(f"DEBUG ENGINE: Generated {len(moves_dfs)} moves-DFs with phase column")


            # ── 2. Longitudinal genérico (ROI, step-functions, …) ────────────
            long_features = longitudinal.aggregate_longitudinal_features(
                games_df, self.reference_stats
            )

            long_features.update({"performance": trend_feats})
            logger.info(f"DEBUG ENGINE: Updated long_features with performance: {trend_feats}")

            # ### NEW · 2  ── Métricas globales de APERTURA
            opening_feats = aggregate_player_opening_patterns(games_df, moves_dfs)
            opening_feats = clean_json_numbers(opening_feats)
            long_features.update({"opening_patterns": opening_feats})  # ← inyectar
            logger.info(f"DEBUG ENGINE: Opening patterns: {opening_feats}")

            # ── 3. Evaluación de riesgo ──────────────────────────────────────
            risk_score, risk_factors = self._calculate_risk_score(
                games_df, long_features
            )

            top = (
                games_df["eco_code"]
                .value_counts()
                .head(5)
                .reset_index(name="count")  # ✅ crea columna 'count' directamente
                .rename(columns={"index": "eco_code"})
            )

            favorite_openings = [
                {
                    "eco_code": row["eco_code"],
                    "name": ECO_NAMES.get(row["eco_code"], "Unknown"),
                    "count": int(row["count"]),
                }
                for _, row in top.iterrows()
            ]
            long_features["favorite_openings"] = favorite_openings  # se inyectará abajo

            phase_feats = compute_phase_quality(moves_dfs)
            long_features.update({"phase_quality": phase_feats})

            user_games = games_df[
                (games_df["white_username"] == username) | (games_df["black_username"] == username)
                ]

            # 2.  columna rating del jugador en cada partida

            # ---------- START rating del jugador (robusto) -----------------------
            WHITE_USR = "white_username"

            if {"white_elo", "black_elo"}.issubset(games_df.columns):
                W_RTG, B_RTG = "white_elo", "black_elo"
            elif {"white_rating", "black_rating"}.issubset(games_df.columns):
                W_RTG, B_RTG = "white_rating", "black_rating"
            else:
                W_RTG = B_RTG = None

            if W_RTG:
                # Añade columna player_rating con np.where
                games_df = games_df.assign(
                    player_rating=lambda df: np.where(
                        df[WHITE_USR] == username, df[W_RTG], df[B_RTG]
                    )
                )

                # Usa el último rating **válido** (descarta NaN/None)
                valid_ratings = games_df["player_rating"].dropna()
                player_elo = int(valid_ratings.iloc[-1]) if not valid_ratings.empty else None
            else:
                games_df = games_df.assign(player_rating=np.nan)
                player_elo = None
            # ---------- END rating del jugador -----------------------------------

            # 4.  benchmark
            avg_acpl = user_games["acpl"].mean()
            mean_entropy = opening_feats.get("mean_entropy")
            
            logger.info(f"DEBUG BENCHMARK PARAMS: avg_acpl={avg_acpl}, mean_entropy={mean_entropy}, player_elo={player_elo}")
            logger.info(f"DEBUG BENCHMARK PARAMS: opening_feats keys={list(opening_feats.keys())}")
            logger.info(f"DEBUG BENCHMARK PARAMS: user_games shape={user_games.shape}")
            
            benchmark = compute_benchmark(avg_acpl, mean_entropy, player_elo)
            logger.info(f"DEBUG BENCHMARK RESULT: {benchmark}")
            long_features["benchmark"] = benchmark

            time_feats = aggregate_time_management(moves_dfs)
            long_features["time_management"] = time_feats

            # 2. games_df lo tienes al principio:
            clutch_feats = aggregate_clutch_accuracy(games_df)
            long_features["clutch_accuracy"] = clutch_feats

            tactical_feats = aggregate_tactical_trends(games_df)
            endgame_feats = aggregate_endgame_efficiency(games_df)

            long_features["tactical"] = tactical_feats
            long_features["endgame"] = endgame_feats

            # después de generar moves_dfs y games_df ...
            complex_corr = aggregate_time_complexity_corr(games_df)

            # fusiona con lo que ya tenías
            long_features["phase_quality"] = phase_feats
            long_features["time_complexity"] = complex_corr

            time_patterns = clean_json_numbers(long_features.get("time_patterns"))
            opening_patterns = clean_json_numbers(long_features.get("opening_patterns"))
            performance = clean_json_numbers(long_features.get("performance"))
            phase_quality = clean_json_numbers(long_features.get("phase_quality"))
            benchmark = clean_json_numbers(long_features.get("benchmark"))
            time_mgmt = clean_json_numbers(long_features.get("time_management"))
            clutch = clean_json_numbers(long_features.get("clutch_accuracy"))
            tactical = clean_json_numbers(long_features.get("tactical"))
            endgame_feats = clean_json_numbers(long_features.get("endgame"))
            time_complexity = clean_json_numbers(long_features.get("time_complexity"))
            performance = clean_json_numbers(long_features.get("performance"))
            risk_factors = clean_json_numbers(risk_factors)
            time_patterns = clean_json_numbers(time_patterns)
            # ── 4. Estadísticos globales y objeto PlayerAnalysisDetailed ─────
            analysis = models.PlayerAnalysisDetailed(
                username=username,
                games_analyzed=len(games_df),
                # ─ Calidad ─
                avg_acpl=_safe_mean(games_df, "acpl"),
                std_acpl=games_df["acpl"].std(ddof=1) or 0.0,
                avg_match_rate=_safe_mean(games_df, "match_rate"),
                std_match_rate=games_df["match_rate"].std(ddof=1) or 0.0,
                avg_ipr=_safe_mean(games_df, "ipr"),
                # ─ Longitudinal ─
                roi_mean=long_features.get("roi_mean"),
                roi_max=long_features.get("roi_max"),
                roi_std=long_features.get("roi_sd"),  # Fix field name mismatch
                step_function_detected=long_features.get("step_acpl_flag"),
                step_function_magnitude=long_features.get("step_acpl_delta"),
                peer_delta_acpl=long_features.get("peer_delta_acpl"),
                peer_delta_match=long_features.get("peer_delta_match"),
                longest_streak=long_features.get("longest_streak"),
                selectivity_score=long_features.get("selectivity_pct"),
                # ─ Nuevos patrones ─
                time_patterns=time_patterns,
                opening_patterns=opening_patterns,  # ← ya poblado
                favorite_openings=long_features.get("favorite_openings"),
                performance=performance,
                phase_quality=phase_quality,  # añade columna JSON al modelo
                benchmark=benchmark,
                time_management=time_mgmt,
                clutch_accuracy=clutch,
                tactical=tactical,
                time_complexity=time_complexity,
                endgame=endgame_feats,
                # ─ Riesgo ─
                risk_score=risk_score,
                risk_factors=risk_factors,
                confidence_level=0,
                first_game_date=long_features.get("first_game_date"),
                last_game_date=long_features.get("last_game_date"),
                analyzed_at=datetime.now(timezone.utc),
            )
            session.add(analysis)
            session.commit()
            return analysis

    def prepare_moves_dataframe(self, game: Game, username: Optional[str] = None) -> pd.DataFrame:
        """Convierte los movimientos de la BD a DataFrame para análisis."""
        return prepare_moves_dataframe(game, username)  # Llamada libre para reutilizar


    def _analyze_quality(self, moves_df: pd.DataFrame, game: Game, player_color: str) -> Dict:
        """Ejecuta análisis de calidad."""
        features = {}

        # ACPL básico with player color awareness
        features['acpl'] = quality.acpl(moves_df, player_color)

        # Match rate
        features['match_rate'] = moves_df['is_engine_best'].mean()

        # Weighted match rate
        features['match_weighted'] = quality.complexity_weighted_match(moves_df)

        # IPR
        features['ipr'] = quality.intrinsic_performance_rating(
            features['match_rate'], features['acpl']
        )

        # Z-scores si tenemos modelo
        if self.acpl_model:
            username = game.white_username if player_color == 'white' else game.black_username
            if username:
                elo = self._estimate_player_elo(username, game)
            else:
                elo = 1800
            features['acpl_z'] = self.acpl_model.z_score(elo, features['acpl'])
            features['ipr_z'] = quality.ipr_z_score(features['ipr'], elo)

        # Detección de rachas
        bursts = quality.precision_bursts(moves_df)
        features['precision_burst_count'] = len(bursts)

        return features

    def _calculate_suspicious_flags(self, features: Dict) -> Dict:
        """Calcula flags de comportamiento sospechoso."""
        return {
            'suspicious_quality': (
                    features.get('acpl', 100) < 20 and
                    features.get('match_rate', 0) > 0.70
            ),
            'suspicious_timing': (
                    features.get('time_complexity_corr', 1) < 0.1 or
                    features.get('lag_spike_count', 0) > 2
            ),
            'suspicious_opening': (
                    features.get('H_opening', 10) < 1.0 and
                    features.get('second_choice_pct', 0) > 0.80
            )
        }

    def _calculate_risk_score(self, games_df: pd.DataFrame,
                              long_features: Dict) -> tuple[float, Dict]:
        """
        Calcula un score de riesgo 0-100 basado en múltiples factores.
        """
        logger.info("DEBUG ENGINE: Starting risk score calculation")

        # ── Normalizar columnas faltantes ────────────────────────────
        REQUIRED = {
            "time_complexity_corr": np.nan,
            "uniformity_score": np.nan,
            "weighted_match_rate": np.nan,  # alias posible
            "acpl": np.nan,
        }
        for col, default in REQUIRED.items():
            if col not in games_df.columns:
                games_df[col] = default

        risk_factors = {}
        risk_score = 0

        # Factor 1: ACPL demasiado bajo
        avg_acpl = games_df['acpl'].mean()
        logger.info(f"DEBUG ENGINE: Average ACPL: {avg_acpl}")
        if avg_acpl < 25:
            risk_factors['low_acpl'] = True
            risk_score += 20
            logger.info("DEBUG ENGINE: Risk factor added - low ACPL")

        # Factor 2: ROI consistentemente alto
        roi_mean = long_features.get('roi_mean', 0)
        logger.info(f"DEBUG ENGINE: ROI mean: {roi_mean}")
        if roi_mean > 2.0:
            risk_factors['high_roi'] = True
            risk_score += 25
            logger.info("DEBUG ENGINE: Risk factor added - high ROI")

        # Factor 3: Step function detectado
        step_detected = long_features.get('step_acpl_flag', False)
        logger.info(f"DEBUG ENGINE: Step function detected: {step_detected}")
        if step_detected:
            risk_factors['step_function'] = True
            risk_score += 20
            logger.info("DEBUG ENGINE: Risk factor added - step function")

        # Factor 4: Streaks largos
        longest_streak = long_features.get('longest_roi_streak', 0)
        logger.info(f"DEBUG ENGINE: Longest streak: {longest_streak}")
        if longest_streak >= 8:
            risk_factors['long_streak'] = True
            risk_score += 15
            logger.info("DEBUG ENGINE: Risk factor added - long streak")

        # Factor 5: Timing anormal
        timing_corr = games_df['time_complexity_corr'].mean()
        logger.info(f"DEBUG ENGINE: Time complexity correlation: {timing_corr}")
        if timing_corr < 0.1:
            risk_factors['abnormal_timing'] = True
            risk_score += 20
            logger.info("DEBUG ENGINE: Risk factor added - abnormal timing")

        final_score = min(risk_score, 100)
        logger.info(f"DEBUG ENGINE: Risk factors identified: {risk_factors}")
        logger.info(f"DEBUG ENGINE: Total risk score: {final_score}")
        return final_score, risk_factors

    def _get_player_games_df(self, username: str, session: Session) -> pd.DataFrame:
        """Obtiene DataFrame con todas las partidas del jugador."""
        stmt = select(Game).where(
            (Game.white_username == username) |
            (Game.black_username == username)
        )
        games = session.exec(stmt).all()

        return pd.DataFrame([{
            'game_id': g.id,
            'eco_code': g.eco_code or 'A00',
            'opening_key': g.opening_key
        } for g in games])

    # --------------------------------------------------------------------------- #
    # 1.  Partidas + análisis detallado                                           #
    # --------------------------------------------------------------------------- #
    def _get_player_games_with_analysis(self, username: str,
                                        session: Session) -> pd.DataFrame:
        """
        Devuelve un DataFrame que une Game ←→ GameAnalysisDetailed
        para todas las partidas en las que `username` jugó con blancas o negras.

        Columnas devueltas:
            game_id, created_at, white_username, black_username,
            result, acpl, match_rate, overall_suspicion_score, analyzed_at, player_color,
            clutch_accuracy_diff, tb_match_rate, dtz_deviation, conversion_efficiency
        """
        stmt = (
            select(models.Game, models.GameAnalysisDetailed)
            .join(models.GameAnalysisDetailed,
                  models.Game.id == models.GameAnalysisDetailed.game_id)
            .where(
                (models.Game.white_username == username) |
                (models.Game.black_username == username)
            )
        )
        rows = session.exec(stmt).all()

        # -- a DataFrame --------------------------------------------------------
        data = []
        for game, detail in rows:
            player_color = 'white' if game.white_username == username else 'black'
            pgn_headers = chess.pgn.read_game(io.StringIO(game.pgn)).headers
            result_tag = pgn_headers.get("Result", "*")
            
            if player_color == 'black':
                if result_tag == "1-0":
                    result_tag = "0-1"
                elif result_tag == "0-1":
                    result_tag = "1-0"
            
            data.append({
                "game_id": game.id,
                "created_at": game.created_at,
                "white": game.white_username,
                "black": game.black_username,
                "white_username": game.white_username,
                "black_username": game.black_username,
                "white_elo": game.white_elo,  # 🆕  ELOs
                "black_elo": game.black_elo,
                "result": result_tag,  # adjusted for player perspective
                "player_color": player_color,  # new field
                # métricas de GameAnalysisDetailed
                "eco_code": game.eco_code,  # 🆕
                "opening_key": game.opening_key,  # (opcional, útil para otras métricas)
                "acpl": detail.acpl,
                "match_rate": detail.match_rate,
                "ipr": detail.ipr,
                "suspicion": detail.overall_suspicion_score,
                "analyzed_at": detail.analyzed_at,
                # 🆕 Incluir clutch_accuracy_diff para el análisis agregado
                "clutch_accuracy_diff": detail.clutch_accuracy_diff if hasattr(detail, 'clutch_accuracy_diff') else np.nan,
                # 🆕 Incluir métricas de endgame para el análisis agregado
                "tb_match_rate": detail.tb_match_rate if hasattr(detail, 'tb_match_rate') else np.nan,
                "dtz_deviation": detail.dtz_deviation if hasattr(detail, 'dtz_deviation') else np.nan,
                "conversion_efficiency": detail.conversion_efficiency if hasattr(detail, 'conversion_efficiency') else np.nan,
            })
        df = pd.DataFrame(data)
        
        return df

    # --------------------------------------------------------------------------- #
    # 2.  Estimación de ELO por media robusta                                     #
    # --------------------------------------------------------------------------- #
    def _estimate_player_elo(self, username: str, game: Optional[Game] = None) -> int:
        """
        Estima el ELO del jugador, usando el contexto del juego específico si está disponible.

        - Usa la BD directamente → no hace peticiones externas.
        - Si no hay datos, devuelve 1800 por defecto.
        """
        with Session(engine) as s:
            if game:
                if game.white_username == username and game.white_elo:
                    return game.white_elo
                elif game.black_username == username and game.black_elo:
                    return game.black_elo
            
            stmt = select(models.Game.white_elo, models.Game.black_elo, 
                         models.Game.white_username, models.Game.black_username).where(
                (models.Game.white_username == username) |
                (models.Game.black_username == username)
            )
            elos = []
            for white_elo, black_elo, white_user, black_user in s.exec(stmt):
                if white_user == username and white_elo and white_elo > 0:
                    elos.append(white_elo)
                elif black_user == username and black_elo and black_elo > 0:
                    elos.append(black_elo)

        if not elos:
            return 1800  # sin datos ⇒ default

        # mediana robusta contra outliers
        return int(np.median(elos))

    # --------------------------------------------------------------------------- #
    # 3.  Detección de final real                                                #
    # --------------------------------------------------------------------------- #
    def _has_endgame(self, moves_df: pd.DataFrame) -> bool:
        """
        Considera que hay final si:
            • Se alcanzó una posición con ≤ 7 piezas   *o*
            • Hay tabla Syzygy disponible (ECO = 'E*') *o*
            • La partida superó 60 jugadas y no quedan damas

        Necesita las SAN de cada jugada en `moves_df.played`.
        """
        board = chess.Board()
        for san in moves_df.played:
            move = board.parse_san(san)
            board.push(move)

            # condición 1: pocas piezas (reloj Syzygy)
            if len(board.piece_map()) <= 7:
                return True

        # condición 2: jugadas largas sin damas
        if board.fullmove_number > 60 and not board.pieces(chess.QUEEN, chess.WHITE | chess.BLACK):
            return True

        return False
