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

from app.analysis.engine import prepare_moves_dataframe
from sqlalchemy.orm import selectinload
from app.utils import TB_PATH

from app.analysis import (
    aggregate_quality_features as q_feats,
    aggregate_time_features    as t_feats,
    aggregate_opening_features as o_feats,
    aggregate_endgame_features as e_feats,
)

from app.analysis.engine import ChessAnalysisEngine

engine_helper = ChessAnalysisEngine()


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
    times_iter = iter(move_times or [])

    analyses: list[models.MoveAnalysis] = []

    for idx, move in enumerate(game.mainline_moves(), start=1):
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

def safe(v):
    # v puede ser None o np.nan; devuélvelo como 0.0 si no es numérico
    try:
        return float(v) if v == v else 0.0      # np.nan != np.nan
    except (TypeError, ValueError):
        return 0.0


@celery_app.task(name="analyze_game_detailed")
def analyze_game_detailed(game_id: int, username: str) -> dict[str, int | str | bool]:
    """
    Calcula las métricas detalladas de una partida **sin** volver a usar Stockfish.

    Reglas:
    • Lee las evaluaciones ya almacenadas en MoveAnalysis (eval_before / eval_after).
    • Persiste el resultado en GameAnalysisDetailed.
    • Notifica progreso (+1 unidad) y devuelve un resumen ligero.
    """
    # ── 1. Cargar partida + movimientos ────────────────────────────────

    # ── 1. Cargar Game + movimientos ────────────────────────────────────────
    with Session(engine) as s:
        game = s.exec(
            select(models.Game)
            .options(selectinload(models.Game.moves))  # eager-load moves
            .where(models.Game.id == game_id)
        ).one()

        # DataFrame de la partida actual (necesita game.moves *antes* de cerrar)
        game_df = prepare_moves_dataframe(game)

        # Copiamos los primitivos que usaremos luego
        opening_key = game.opening_key
        eco_code = game.eco_code

    # ── 2. DataFrame de todas las partidas del jugador ──────────────────────

    with Session(engine) as s:
        orm_game = s.exec(
            select(models.Game)
            .options(selectinload(models.Game.moves))
            .where(models.Game.id == game_id)
        ).one()

        # Creamos el objeto python-chess Game *antes* de cerrar la sesión
        game_pgn_obj = chess.pgn.read_game(io.StringIO(orm_game.pgn))
        game_df = prepare_moves_dataframe(orm_game)  # ya lo tenías
        opening_key = orm_game.opening_key
        eco_code = orm_game.eco_code
    # -- sesión cerrada -----------------------------------------------

    with Session(engine) as s:
        games_df = engine_helper._get_player_games_with_analysis(username, s)

    # Llamadas a agregadores
    q = q_feats(game_df)
    t = t_feats(game_df)
    o = o_feats(opening_key, eco_code, game_df, games_df)
    e = e_feats(game_pgn_obj, game_df, TB_PATH if TB_PATH else None)

    quality_score = q.get("quality_score", 0) or 0  # None → 0

    overall_score = (
            safe(quality_score) * 0.4 +
            safe(t.get("timing_score")) * 0.25 +
            safe(o.get("opening_score")) * 0.2 +
            safe(e.get("endgame_score")) * 0.15
    )


    # ── 4. Persistir en BD ─────────────────────────────────────────────
    with Session(engine) as s:
        detailed = models.GameAnalysisDetailed(
            game_id=game_id,
            analyzed_at=datetime.now(UTC),
            # ─ Calidad ─
            acpl=q.get("acpl", 0),
            match_rate=q.get("match_rate", 0),
            weighted_match_rate=q.get("weighted_match_rate"),
            ipr=q.get("ipr", 0),
            ipr_z_score=q.get("ipr_z_score", 0),
            # ─ Tiempo ─
            mean_move_time=t.get("mean_move_time", 0),
            time_variance=t.get("time_variance", 0),
            time_complexity_corr=t.get("time_complexity_corr"),
            lag_spike_count=t.get("lag_spike_count"),
            uniformity_score=t.get("uniformity_score"),
            # ─ Apertura ─
            opening_entropy=o.get("opening_entropy"),
            novelty_depth=o.get("novelty_depth"),
            second_choice_rate=o.get("second_choice_rate"),
            opening_breadth=o.get("opening_breadth"),
            # ─ Final ─
            tb_match_rate=e.get("tb_match_rate"),
            dtz_deviation=e.get("dtz_deviation"),
            conversion_efficiency=e.get("conversion_efficiency"),
            # ─ Flags & score ─
            suspicious_quality=quality_score > 50,
            suspicious_timing=t.get("timing_score") > 50,
            suspicious_opening=o.get("opening_score") > 50,
            overall_suspicion_score=overall_score,
        )
        s.merge(detailed)     # create-or-update
        s.commit()

        # ⚠️  capturamos los valores **antes** de cerrar la sesión
        suspicion_flag = overall_score > 50
        analyzed_at    = detailed.analyzed_at.isoformat()

    # ── 5. Actualizar progreso del jugador ─────────────────────────────
    update_progress(username, increment=1)

    # ── 6. Notificar por WebSocket (opcional) ──────────────────────────
    notify_ws(
        username,
        {
            "game_id": game_id,
            "suspicious": suspicion_flag,
            "analyzed_at": analyzed_at,
        },
    )

    # ── 7. Respuesta liviana para el `chord` / caller ──────────────────
    return {
        "game_id": game_id,
        "suspicious": suspicion_flag,
        "score": round(overall_score, 1),
        "analyzed_at": analyzed_at,
    }

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