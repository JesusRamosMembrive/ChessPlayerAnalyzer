"""
Analysis Tasks Module

This module contains all Celery tasks related to game and player analysis,
including Stockfish analysis, detailed metrics computation, and player longitudinal analysis.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import chess.pgn
import chess.engine

from celery import current_task
from celery.exceptions import SoftTimeLimitExceeded
from sqlmodel import Session, select
from sqlalchemy import func
from sqlalchemy.orm import selectinload

from app.models import GameAnalysisDetailed
from app import models
from app.analysis.engine import prepare_moves_dataframe
from app.analysis import (
    aggregate_quality_features as q_feats,
    aggregate_time_features    as t_feats,
    aggregate_opening_features as o_feats,
    aggregate_endgame_features as e_feats,
)
from app.analysis.bayesian import BayesianSuspicionModel
from app.utils import TB_PATH

from .shared_config import (
    logger, engine, analysis_engine, engine_helper, safe, export_analysis_to_json,
    is_task_aborted, redis_client, notify_ws, update_progress, task_progress,
    cache_get, cache_set, ENGINE_PATH, MAX_DEPTH,
    TASK_SOFT_TIME_LIMIT, TASK_TIME_LIMIT, TASK_MAX_RETRIES
)

def register_analysis_tasks(app):
    """Register analysis tasks with the Celery app instance."""

    # Register tasks with decorators
    analyze_player_detailed_task = app.task(
        name="analyze_player_detailed",
        autoretry_for=(Exception, SoftTimeLimitExceeded),
        retry_backoff=True,
        retry_backoff_max=600,
        retry_jitter=True,
        retry_kwargs={"max_retries": TASK_MAX_RETRIES},
        soft_time_limit=TASK_SOFT_TIME_LIMIT,
        time_limit=TASK_TIME_LIMIT,
    )(analyze_player_detailed)

    analyze_game_task_decorated = app.task(
        name="analyze_game_task",
        bind=True,
        autoretry_for=(Exception, SoftTimeLimitExceeded),
        retry_backoff=True,
        retry_backoff_max=600,
        retry_jitter=True,
        retry_kwargs={"max_retries": TASK_MAX_RETRIES},
        soft_time_limit=TASK_SOFT_TIME_LIMIT,
        time_limit=TASK_TIME_LIMIT,
    )(analyze_game_task)

    analyze_game_detailed_task = app.task(
        name="analyze_game_detailed",
        autoretry_for=(Exception, SoftTimeLimitExceeded),
        retry_backoff=True,
        retry_backoff_max=600,
        retry_jitter=True,
        retry_kwargs={"max_retries": TASK_MAX_RETRIES},
        soft_time_limit=TASK_SOFT_TIME_LIMIT,
        time_limit=TASK_TIME_LIMIT,
    )(analyze_game_detailed)

    extract_game_id_task = app.task(name="extract_game_id")(extract_game_id)

    return {
        'analyze_player_detailed': analyze_player_detailed_task,
        'analyze_game_task': analyze_game_task_decorated,
        'analyze_game_detailed': analyze_game_detailed_task,
        'extract_game_id': extract_game_id_task,
    }

def analyze_player_detailed(_, username: str):
    """
    Análisis longitudinal detallado de un jugador.
    Se ejecuta después de que todas sus partidas han sido analizadas.
    """
    logger.info(f"DEBUG PLAYER: Starting player detailed analysis for {username}")
    cached = cache_get("analyze_player_detailed", [username], {})
    if cached:
        logger.info("DEBUG PLAYER: Returning cached result for analyze_player_detailed")
        return cached

    try:
        # Verificar que hay suficientes partidas analizadas
        with Session(engine) as s:
            analyzed_count = s.exec(
                select(func.count(GameAnalysisDetailed.game_id))
                .join(models.Game)
                .where(
                    (models.Game.white_username == username) |
                    (models.Game.black_username == username)
                )
            ).one()

            logger.info(f"DEBUG PLAYER: Pre-analysis check - Found {analyzed_count} games with existing analysis for {username} in GameAnalysisDetailed table")

            if analyzed_count < 1:
                logger.warning(f"Insuficientes partidas analizadas para {username}: {analyzed_count}")
                return {
                    "username": username,
                    "status": "insufficient_data",
                    "games_analyzed": analyzed_count
                }

        # Ejecutar análisis del jugador
        logger.info(f"DEBUG PLAYER: Starting analysis engine for {username}")
        player_analysis = analysis_engine.analyze_player(username)
        logger.info("DEBUG PLAYER: PlayerAnalysisDetailed result: %s", player_analysis)

        with Session(engine) as s:
           pa = s.get(models.PlayerAnalysisDetailed, username)
           logger.info(f"DEBUG PLAYER: Retrieved player analysis from DB for {username}: risk_score={pa.risk_score}, games_analyzed={pa.games_analyzed} (this count reflects games with completed analysis in GameAnalysisDetailed table)")

           export_analysis_to_json(pa, username, "player")

        # Notificar resultado
        notify_ws(username, {
            "type": "player_analysis_complete",
            "risk_score": pa.risk_score,
            "risk_factors": pa.risk_factors
        })

        result = {
            "username": username,
            "risk_score": pa.risk_score,
            "games_analyzed": pa.games_analyzed,
            "analyzed_at": pa.analyzed_at.isoformat()
        }

        with Session(engine) as s:
            player = s.get(models.Player, username)
            if player:
                player.status = "ready"
                player.progress = 100
                player.finished_at = datetime.now(timezone.utc)
                s.add(player)
                s.commit()
        from app.main import clear_analysis_in_progress
        clear_analysis_in_progress()
        logger.info(f"Analysis completed for user {username} - system ready for new requests")

        notify_ws(username, {"status": "ready", "progress": 100})

        logger.info(f"DEBUG PLAYER: Final analysis result for {username}: risk_score={result['risk_score']}, games_analyzed={result['games_analyzed']} (total games processed in this analysis session)")
        # Store result in cache
        cache_set("analyze_player_detailed", [username], {}, result)
        return result

    except Exception as e:
        logger.error(f"DEBUG PLAYER: Error en análisis detallado de jugador {username}: {e}")
        # APM: Capturar excepción con contexto adicional
        raise

def analyze_game_task(
    self,
    pgn_text: str,
    game_id: int | None = None,
    *,
    move_times: list[int] | None = None,
    player: str | None = None,
    depth: int = MAX_DEPTH,
    multipv: int = 3,
):
    """
    Analiza una partida con Stockfish.

    • Si `game_id` es None se crea primero el registro Game.
    • Para cada jugada se guarda:
        – best_rank   (0 == mejor jugada)
        – cp_loss     (centipawns perdidos respecto PV1)
    """

    # Verificar si la tarea ha sido revocada
    if self.request.called_directly:
        # Si se ejecuta directamente, no verificar revocación
        pass
    else:
        # Verificar si la tarea ha sido revocada
        if is_task_aborted(self, player):
            logger.info(f"Task {self.request.id} aborted before start – exiting early")
            return {"status": "revoked", "game_id": game_id}

    logger.info(f"DEBUG STOCKFISH: Starting analyze_game_task for game_id: {game_id}, player: {player}")
    logger.info(f"DEBUG STOCKFISH: PGN length: {len(pgn_text)} chars, move_times: {len(move_times) if move_times else 0} entries")
    logger.info(f"DEBUG STOCKFISH: Engine settings - depth: {depth}, multipv: {multipv}")

    # ---- Result cache check ----
    cached = cache_get("analyze_game_task", [pgn_text, depth, multipv, move_times], {"player": player})
    if cached:
        logger.info("DEBUG STOCKFISH: Returning cached result for analyze_game_task")
        return cached

    # ---------- 1.  Asegurar objeto Game en BD --------------------
    if game_id is None:
        game_pgn = chess.pgn.read_game(io.StringIO(pgn_text))
        if game_pgn is None:
            raise ValueError("PGN inválido")

        game_headers = game_pgn.headers
        white = game_headers.get("White")
        black = game_headers.get("Black")
        logger.info(f"DEBUG STOCKFISH: Creating new game record - White: {white}, Black: {black}")

        with Session(engine) as s:
            game_db = models.Game(
                pgn=pgn_text,
                move_times=move_times or [],
                white_username=white,
                black_username=black,
            )
            s.add(game_db)
            s.commit()
            s.refresh(game_db)
            game_id = game_db.id
            logger.info(f"DEBUG STOCKFISH: Created game with ID: {game_id}")
    else:
        #  buscamos el registro ya existente
        with Session(engine) as s:
            game_db = s.get(models.Game, game_id)
            if game_db is None:
                raise ValueError(f"Game id {game_id} no existe")
            logger.info(f"DEBUG STOCKFISH: Using existing game ID: {game_id}")

    # ---------- 2.  Preparar tablero y motor ----------------------
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    board = game.board()
    logger.info(f"DEBUG STOCKFISH: Starting position: {board.fen()}")

    engine_sf = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    logger.info(f"DEBUG STOCKFISH: Stockfish engine initialized from: {ENGINE_PATH}")

    times_iter = iter(move_times or [])

    analyses: list[models.MoveAnalysis] = []
    total_moves = len(list(game.mainline_moves()))
    progress_every = max(1, total_moves // 20)  # ~5% granularity
    logger.info(f"DEBUG STOCKFISH: Total moves to analyze: {total_moves}")

    for idx, move in enumerate(game.mainline_moves(), start=1):
        # Verificar revocación cada movimiento (más frecuente para respuesta rápida)
        if not self.request.called_directly and is_task_aborted(self, player):
            logger.info(f"Task {self.request.id} has been revoked during move analysis, stopping execution")
            engine_sf.quit()
            return {"status": "revoked", "game_id": game_id, "moves_analyzed": idx - 1}

        legal_cnt = board.legal_moves.count()

        # 1. Eval antes de mover (posición actual)
        info_before = engine_sf.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        eval_before = info_before[0]["score"].white().score(mate_score=100000) or 0
        best_move = info_before[0]["pv"][0]

        # 2. Eval después de la jugada real
        board.push(move)
        info_after = engine_sf.analyse(board, chess.engine.Limit(depth=depth))
        eval_after = info_after["score"].white().score(mate_score=100000) or 0
        board.pop()

        rank = next((i for i, pv in enumerate(info_before) if pv["pv"][0] == move), multipv)
        cp_loss = abs(eval_before - eval_after)
        time_spent = next(times_iter, None)  # simplemente None si no hay clocks

        if idx <= 5 or idx % 10 == 0:  # Log first 5 moves and every 10th move
            logger.info(f"DEBUG STOCKFISH: Move {idx}/{total_moves} - {board.san(move)}, eval_before: {eval_before}, eval_after: {eval_after}, cp_loss: {cp_loss}, rank: {rank}, legal_moves: {legal_cnt}")

        analyses.append(models.MoveAnalysis(
            game_id=game_id,
            move_number=idx,
            played=board.san(move),
            best=board.san(best_move),
            best_rank=rank,
            cp_loss=cp_loss,
            eval_before=eval_before,
            eval_after=eval_after,
            legal_moves_count=legal_cnt,
            time_spent=time_spent,
        ))
        board.push(move)

        # ── Progress update ─────────────────────────────────────
        if idx % progress_every == 0 or idx == total_moves:
            try:
                task_progress(self, idx, total_moves, player)
            except Exception:
                pass

    engine_sf.quit()
    logger.info(f"DEBUG STOCKFISH: Stockfish analysis complete, analyzed {len(analyses)} moves")

    # ---------- 3.  Guardar análisis y metadatos ------------------
    opening_key = " ".join(
        node.san() for i, node in enumerate(game.mainline()) if i < 8
    )
    eco_code = game.headers.get("eco_code", "Unknown")
    logger.info(f"DEBUG STOCKFISH: Opening key: {opening_key}, eco_code: {eco_code}")

    with Session(engine) as s:
        if game_id is None:
            g = models.Game(
                pgn=pgn_text,
                move_times=move_times or [],
                white_username=game.headers.get("White"),
                black_username=game.headers.get("Black"),
            )
            s.add(g)
            s.commit()
            s.refresh(g)
            game_id = g.id

        if not self.request.called_directly and is_task_aborted(self, player):
            logger.info(f"Task {self.request.id} aborted before database commit, skipping save")
            return {"status": "revoked", "game_id": game_id, "moves_analyzed": len(analyses)}

        game_db = s.get(models.Game, game_id)
        if not game_db:
            logger.warning(f"Game {game_id} no longer exists, skipping analysis save")
            return {"status": "cancelled", "game_id": game_id, "reason": "game_deleted"}

        s.add_all(analyses)

        # actualizar campos en Game
        game_db = s.get(models.Game, game_id)
        game_db.opening_key = opening_key
        game_db.eco_code = eco_code
        s.add(game_db)
        s.commit()
        logger.info(f"DEBUG STOCKFISH: Saved {len(analyses)} move analyses to database")

    if player:
        update_progress(player, increment=1)

    result = {
        "game_id": game_id,
        "move_count": len(analyses),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("DEBUG STOCKFISH: analyze_game_task result: %s", result)
    # Store result in cache
    cache_set("analyze_game_task", [pgn_text, depth, multipv, move_times], {"player": player}, result)
    return result

def extract_game_id(result: dict) -> int:
    """Extract game_id from analyze_game_task result."""
    gid = result.get("game_id")
    if gid is None:
        raise ValueError("Missing game_id in analyze_game_task result")
    return int(gid)

def analyze_game_detailed(game_id: int, username: str) -> dict[str, int | str | bool]:
    """
    Calcula las métricas detalladas de una partida **sin** volver a usar Stockfish.

    Reglas:
    • Lee las evaluaciones ya almacenadas en MoveAnalysis (eval_before / eval_after).
    • Persiste el resultado en GameAnalysisDetailed.
    • Notifica progreso (+1 unidad) y devuelve un resumen ligero.
    """
    logger.info(f"DEBUG DETAILED: Starting detailed analysis for game_id: {game_id}, username: {username}")

    # Check for cancellation at the start
    cancellation_key = f"cancel:{username}"
    if redis_client.get(cancellation_key):
        logger.info(f"analyze_game_detailed detected Redis cancellation flag for {username}")
        return {"status": "cancelled", "game_id": game_id, "username": username}

    # ---- Result cache check ----
    cached = cache_get("analyze_game_detailed", [game_id], {"username": username})
    if cached:
        logger.info("DEBUG DETAILED: Returning cached result for analyze_game_detailed")
        return cached

    # ── 1. Cargar partida + movimientos ────────────────────────────────

    # ── 1. Cargar Game + movimientos ────────────────────────────────────────
    with Session(engine) as s:
        # ── 1. Cargar Game + movimientos ────────────────────────────────────────
        game = s.exec(
            select(models.Game)
            .options(selectinload(models.Game.moves))  # eager-load moves
            .where(models.Game.id == game_id)
        ).one()

        # DataFrame de la partida actual (necesita game.moves *antes* de cerrar)
        game_df = prepare_moves_dataframe(game, username)

        player_color = 'white' if game.white_username == username else 'black'
        logger.info(f"DEBUG DETAILED: Game DataFrame shape: {game_df.shape}")
        logger.info(f"DEBUG DETAILED: Game DataFrame columns: {list(game_df.columns)}")
        logger.info(f"DEBUG DETAILED: Sample game data:\n{game_df.head(3).to_string()}")

        # Copiamos los primitivos que usaremos luego
        opening_key = game.opening_key
        eco_code = game.eco_code
        logger.info(f"DEBUG DETAILED: Opening key: {opening_key}, ECO: {eco_code}")

        # ── 2. DataFrame de todas las partidas del jugador ──────────────────────
        # Creamos el objeto python-chess Game *antes* de procesar
        game_pgn_obj = chess.pgn.read_game(io.StringIO(game.pgn))

        try:
            games_df = engine_helper._get_player_games_with_analysis(username, s)
        except Exception as e:
            logger.warning(f"Could not get player games for analysis: {e}")
            import pandas as pd
            games_df = pd.DataFrame()

    # -- sesión cerrada -----------------------------------------------

    # Llamadas a agregadores
    if redis_client.get(cancellation_key):
        logger.info(f"analyze_game_detailed detected Redis cancellation flag for {username} after quality stage")
        return {"status": "cancelled", "game_id": game_id, "username": username}

    logger.info("DEBUG DETAILED: Starting quality features calculation")
    q = q_feats(game_df, elo=game.white_elo if player_color == 'white' else game.black_elo, player_color=player_color)
    logger.info(f"DEBUG DETAILED: Quality features: {q}")

    # Progress 1/5 (we include final save as last step)
    try:
        task_progress(current_task, 1, 5, username)
    except Exception:
        pass

    logger.info("DEBUG DETAILED: Starting timing features calculation")
    if redis_client.get(cancellation_key):
        logger.info(f"analyze_game_detailed detected Redis cancellation flag for {username} after timing stage")
        return {"status": "cancelled", "game_id": game_id, "username": username}

    t = t_feats(game_df)
    logger.info(f"DEBUG DETAILED: Timing features: {t}")

    try:
        task_progress(current_task, 2, 5, username)
    except Exception:
        pass

    logger.info("DEBUG DETAILED: Starting opening features calculation")
    if redis_client.get(cancellation_key):
        logger.info(f"analyze_game_detailed detected Redis cancellation flag for {username} after opening stage")
        return {"status": "cancelled", "game_id": game_id, "username": username}

    o = o_feats(opening_key, eco_code, game_df, games_df)
    logger.info(f"DEBUG DETAILED: Opening features: {o}")

    try:
        task_progress(current_task, 3, 5, username)
    except Exception:
        pass

    logger.info("DEBUG DETAILED: Starting endgame features calculation")
    if redis_client.get(cancellation_key):
        logger.info(f"analyze_game_detailed detected Redis cancellation flag for {username} after endgame stage")
        return {"status": "cancelled", "game_id": game_id, "username": username}

    e = e_feats(game_pgn_obj, game_df, TB_PATH if TB_PATH and Path(TB_PATH).exists() else None)
    logger.info(f"DEBUG DETAILED: Endgame features: {e}")

    try:
        task_progress(current_task, 4, 5, username)
    except Exception:
        pass

    if redis_client.get(cancellation_key):
        logger.info(f"analyze_game_detailed detected Redis cancellation flag for {username} before scoring stage")
        return {"status": "cancelled", "game_id": game_id, "username": username}
    model = BayesianSuspicionModel()
    rating = game.white_elo if player_color == 'white' else game.black_elo
    experience = len(games_df)
    evidence = {
        "acpl": q.get("acpl", 0),
        "match_rate": q.get("match_rate", 0),
        "time_complexity_corr": t.get("time_complexity_corr", 0),
        "lag_spike_count": t.get("lag_spike_count", 0),
        "opening_entropy": o.get("opening_entropy", 0),
        "second_choice_rate": o.get("second_choice_rate", 0),
    }
    suspicion_score = model.update(rating, experience, evidence)
    logger.info(f"DEBUG DETAILED: Bayesian suspicion score: {suspicion_score}")

    # ── 4. Persistir en BD ─────────────────────────────────────────────
    with Session(engine) as s:
        detailed = models.GameAnalysisDetailed(
            game_id=game_id,
            analyzed_at=datetime.now(timezone.utc),
            # ─ Calidad ─
            acpl=safe(q.get("acpl", 0)),
            wdl_loss=safe(q.get("wdl_loss", 0.0)),
            match_rate=safe(q.get("match_rate", 0)),
            weighted_match_rate=safe(q.get("weighted_match_rate")),
            ipr=safe(q.get("ipr", 0)),
            ipr_z_score=safe(q.get("ipr_z_score", 0)),
            # ─ Tiempo ─
            mean_move_time=safe(t.get("mean_move_time", 0)),
            time_variance=safe(t.get("time_variance", 0)),
            time_complexity_corr=safe(t.get("time_complexity_corr")),
            lag_spike_count=int(t.get("lag_spike_count") or 0),
            clutch_accuracy_diff=(None if (t.get("clutch_accuracy_diff") is None or (t.get("clutch_accuracy_diff") != t.get("clutch_accuracy_diff"))) else float(t.get("clutch_accuracy_diff"))),
            uniformity_score=safe(t.get("uniformity_score")),
            # ─ Apertura ─
            opening_entropy=safe(o.get("opening_entropy")),
            novelty_depth=(None if (o.get("novelty_depth") is None or (o.get("novelty_depth") != o.get("novelty_depth"))) else int(o.get("novelty_depth"))),
            second_choice_rate=safe(o.get("second_choice_rate")),
            opening_breadth=int(o.get("opening_breadth") or 0),
            # ─ Final ─
            tb_match_rate=(None if (e.get("tb_match_rate") is None or (e.get("tb_match_rate") != e.get("tb_match_rate"))) else float(e.get("tb_match_rate"))),
            dtz_deviation=(None if (e.get("dtz_deviation") is None or (e.get("dtz_deviation") != e.get("dtz_deviation"))) else float(e.get("dtz_deviation"))),
            conversion_efficiency=(None if (e.get("conversion_efficiency") is None or (e.get("conversion_efficiency") != e.get("conversion_efficiency"))) else int(e.get("conversion_efficiency"))),
            # ─ Flags & score ─
            suspicious_quality=False,
            suspicious_timing=False,
            suspicious_opening=False,
            overall_suspicion_score=safe(suspicion_score),
        )
        cancellation_key = f"cancel:{username}"
        if redis_client.get(cancellation_key):
            logger.info(f"analyze_game_detailed detected Redis cancellation flag for {username} before database commit")
            return {"status": "cancelled", "game_id": game_id, "username": username}

        game_check = s.get(models.Game, game_id)
        if not game_check:
            logger.warning(f"Game {game_id} no longer exists, skipping detailed analysis save")
            return {"status": "cancelled", "game_id": game_id, "reason": "game_deleted"}

        s.merge(detailed)     # create-or-update
        s.commit()
        logger.info("DEBUG DETAILED: GameAnalysisDetailed saved: %s", detailed)

        export_analysis_to_json(detailed, username, "game")

        # ⚠️  capturamos los valores **antes** de cerrar la sesión
        suspicion_flag = suspicion_score > 0.5
        analyzed_at    = detailed.analyzed_at.isoformat()

    # ── 4. Actualizar progreso del jugador ─────────────────────────────
    update_progress(username, increment=1)

    # ── 5. Notificar por WebSocket (opcional) ──────────────────────────
    notify_ws(
        username,
        {
            "game_id": game_id,
            "suspicious": suspicion_flag,
            "analyzed_at": analyzed_at,
        },
    )

    # ── 6. Respuesta liviana para el `chord` / caller ──────────────────
    result = {
        "game_id": game_id,
        "suspicious": suspicion_flag,
        "score": round(suspicion_score, 3),
        "analyzed_at": analyzed_at,
    }

    logger.info("DEBUG DETAILED: analyze_game_detailed result: %s", result)
    # Store result in cache
    cache_set("analyze_game_detailed", [game_id], {"username": username}, result)

    # Final progress 5/5
    try:
        task_progress(current_task, 5, 5, username)
    except Exception:
        pass

    return result