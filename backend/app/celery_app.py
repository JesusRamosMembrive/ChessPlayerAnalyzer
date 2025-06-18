from __future__ import annotations

import io
import logging
import os
from datetime import datetime, UTC
from pathlib import Path

import chess.engine
import chess.pgn

from app.analysis.engine import ChessAnalysisEngine
from app.database import engine


from app.utils import fetch_games, notify_ws, update_progress
from celery import Celery
from celery import chain, group, chord
from celery.signals import task_failure
from celery_once import QueueOnce
from sqlmodel import Session, select
from sqlalchemy import func

from app.models import GameAnalysisDetailed
from app import models

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("chess_tasks", broker=REDIS_URL, backend=REDIS_URL)

ENGINE_PATH = os.getenv("STOCKFISH_PATH", "stockfish")
MAX_DEPTH = int(os.getenv("STOCKFISH_DEPTH", "12"))



@celery_app.task(name="analyze_player_detailed")
def analyze_player_detailed(username: str):
    """
    Análisis longitudinal detallado de un jugador.
    Se ejecuta después de que todas sus partidas han sido analizadas.
    """
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

            if analyzed_count < 10:
                logging.warning(f"Insuficientes partidas analizadas para {username}: {analyzed_count}")
                return {
                    "username": username,
                    "status": "insufficient_data",
                    "games_analyzed": analyzed_count
                }

        # Ejecutar análisis del jugador
        player_analysis = analysis_engine.analyze_player(username)

        # Notificar resultado
        notify_ws(username, {
            "type": "player_analysis_complete",
            "risk_score": player_analysis.risk_score,
            "risk_factors": player_analysis.risk_factors
        })

        return {
            "username": username,
            "risk_score": player_analysis.risk_score,
            "games_analyzed": player_analysis.games_analyzed,
            "analyzed_at": player_analysis.analyzed_at.isoformat()
        }

    except Exception as e:
        logging.error(f"Error en análisis detallado de jugador {username}: {e}")
        raise


@celery_app.task(name="analyze_game_task", bind=True)
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


    # ---------- 1.  Asegurar objeto Game en BD --------------------
    if game_id is None:
        game_pgn = chess.pgn.read_game(io.StringIO(pgn_text))
        if game_pgn is None:
            raise ValueError("PGN inválido")

        game_headers = game_pgn.headers
        white = game_headers.get("White")
        black = game_headers.get("Black")

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
    else:
        #  buscamos el registro ya existente
        with Session(engine) as s:
            game_db = s.get(models.Game, game_id)
            if game_db is None:
                raise ValueError(f"Game id {game_id} no existe")

    # ---------- 2.  Preparar tablero y motor ----------------------
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    board = game.board()
    engine_sf = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

    analyses: list[models.MoveAnalysis] = []

    for idx, move in enumerate(game.mainline_moves(), start=1):
        infos = engine_sf.analyse(
            board, chess.engine.Limit(depth=depth), multipv=multipv
        )

        # lista de los movimientos PV1..PVn
        top_moves = [info["pv"][0] for info in infos]
        best_eval = infos[0]
        best_move = best_eval["pv"][0]
        best_score = best_eval["score"].white().score(mate_score=100000)

        # rank: 0-based; multipv si está fuera del top-N
        try:
            rank = top_moves.index(move)
        except ValueError:
            rank = multipv

        # evaluamos la posición *después* del movimiento real
        board.push(move)
        after = engine_sf.analyse(board, chess.engine.Limit(depth=depth))
        after_score = after["score"].white().score(mate_score=100000)
        board.pop()

        cp_loss = abs((best_score or 0) - (after_score or 0))

        analyses.append(
            models.MoveAnalysis(
                game_id=game_id,
                move_number=idx,
                played=board.san(move),
                best=board.san(best_move),
                best_rank=rank,
                cp_loss=int(cp_loss),
            )
        )

        board.push(move)

    engine_sf.quit()

    # ---------- 3.  Guardar análisis y metadatos ------------------
    opening_key = " ".join(
        node.san() for i, node in enumerate(game.mainline()) if i < 8
    )
    eco_code = game.headers.get("ECO")

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

        s.add_all(analyses)

        # actualizar campos en Game
        game_db = s.get(models.Game, game_id)
        game_db.opening_key = opening_key
        game_db.eco_code = eco_code
        s.add(game_db)
        s.commit()

    # if player:
    #     pl = s.get(models.Player, player)
    #     if pl:
    #         # ¿terminado?
    #         if pl.done_games == pl.total_games:
    #             pl.status = "ready"
    #             pl.finished_at = datetime.now(UTC)
    #             notify_ws(player, {"status": "ready"})
    #         else:
    #             notify_ws(
    #                 player,
    #                 {"progress": pl.progress, "status": "pending"},
    #             )
    #         s.add(pl)
    #         s.commit()

    update_progress(player, increment=1)

    return {
        "game_id": game_id,
        "move_count": len(analyses),
        "analyzed_at": datetime.now(UTC).isoformat(),
    }



# Inicializar el motor de análisis (configurar rutas según tu sistema)
analysis_engine = ChessAnalysisEngine(
    reference_book_path=Path("/data/reference_book.bin") if Path("/data/reference_book.bin").exists() else None,
    tablebase_path=Path("/data/syzygy") if Path("/data/syzygy").exists() else None,
)


@celery_app.task(name="analyze_game_detailed")
def analyze_game_detailed(game_id: int, username: str):
    """
    Análisis detallado de una partida usando los nuevos módulos.
    Esta tarea complementa a analyze_game_task existente.
    """
    try:
        # Ejecutar análisis detallado
        detailed_analysis = analysis_engine.analyze_game(game_id)

        # Notificar progreso
        with Session(engine) as s:
            game = s.get(models.Game, game_id)
            if game and game.white_username:
                notify_ws(game.white_username, {
                    "type": "game_analysis_complete",
                    "game_id": game_id,
                    "suspicious": detailed_analysis.overall_suspicion_score > 50
                })

        update_progress(username, increment=1)

        return {
            "game_id": game_id,
            "acpl": detailed_analysis.acpl,
            "suspicious_score": detailed_analysis.overall_suspicion_score,
            "analyzed_at": detailed_analysis.analyzed_at.isoformat()
        }

    except Exception as e:
        logging.error(f"Error en análisis detallado de partida {game_id}: {e}")
        raise

@celery_app.task(name="process_player_enhanced")
def process_player_enhanced(username: str, months: int = 6):
    # 1. DESCARGAR partidas y crear registros Game ──────────────────────────
    games = fetch_games(username, months)
    game_ids = []

    with Session(engine) as s:
        player = models.Player(
            username=username,
            status="pending",
            requested_at=datetime.now(UTC),
            progress=0,
            total_games=len(games),
            done_games=0,
        )
        s.merge(player);  s.commit()

    chains = []          # ← aquí iremos acumulando chain por partida
    for g in games:
        with Session(engine) as s:
            game_db = models.Game(
                pgn=g["pgn"],
                move_times=g.get("move_times", []),
                white_username=g.get("white"),
                black_username=g.get("black"),
                white_elo=g.get("white_elo"),
                black_elo=g.get("black_elo"),
            )
            s.add(game_db);  s.commit();  s.refresh(game_db)
            gid = game_db.id
            game_ids.append(gid)

        # 2.  chain:  básico → detallado ───────────────────────────────────
        basic = analyze_game_task.s(g["pgn"], gid, move_times=g.get("move_times"), player=username)
        detailed = analyze_game_detailed.si(gid, username)
        chains.append(chain(basic, detailed))

    # 3. group & chord: cuando todas las partidas acaben … ──────────────────
    #    se lanza analyze_player_detailed(username)
    full_workflow = chord(group(chains), analyze_player_detailed.s(username))
    full_workflow.delay()

    return {
        "username": username,
        "games_queued": len(games),
        "enhanced_analysis": True,
        "task_id": full_workflow.id,
    }


@task_failure.connect
def on_task_failure(sender=None, task_id=None, args=None, kwargs=None, **k):
    if sender.name == "process_player_enhanced":
        username = args[0]
        with Session(engine) as s:
            pl = s.get(models.Player, username)
            if pl:
                pl.status = "error"
                pl.error = str(k.get("exception", "unknown"))
                s.add(pl); s.commit()
        notify_ws(username, {"status": "error"})


@celery_app.task(name="process_player")
def _deprecated(*a, **kw):
    raise RuntimeError("Deprecated. Use process_player_enhanced")