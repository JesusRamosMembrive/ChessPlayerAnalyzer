"""Tareas Celery relacionadas con el análisis **individual** de partidas.

Paso 1 del refactor: reexportar funciones que ya viven en ``app.celery_app``
para poder importar desde otros módulos sin depender de un fichero gigante.
Más adelante se moverá la implementación aquí y se limpiará ``celery_app.py``.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from app.celery_app import (
    celery_app,
    analyze_game_task_legacy,
    analyze_game_detailed_legacy,
    redis_client,
    ENGINE_PATH,
    MAX_DEPTH,
    TB_PATH,
)
from app.tasks.utils import safe, export_analysis_to_json  # noqa: F401 (se usarán en siguientes etapas)
from app.utils import cache_get, cache_set  # noqa: F401

from app.database import engine
from sqlmodel import Session
from sqlalchemy import select
from app import models
import io
import chess.pgn

# ---------------------------------------------------------------------------
# Helper: crear u obtener el registro Game (1ª parte de analyze_game_task)
# ---------------------------------------------------------------------------

def _prepare_game_record(pgn_text: str, game_id: int | None, move_times: list[int] | None) -> tuple[int, chess.pgn.Game]:
    """Normaliza el registro Game.

    Si *game_id* es ``None`` crea un nuevo registro; si existe lo valida.
    Devuelve el ``game_id`` final y el objeto *python-chess* ``Game`` ya parseado.
    """
    # Parsear PGN (se necesitará siempre)
    game_pgn = chess.pgn.read_game(io.StringIO(pgn_text))
    if game_pgn is None:
        raise ValueError("PGN inválido")

    if game_id is None:
        headers = game_pgn.headers
        white = headers.get("White")
        black = headers.get("Black")

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
        logger.info("DEBUG MIGRATE: Created new Game id=%s", game_id)
    else:
        with Session(engine) as s:
            if not s.get(models.Game, game_id):
                raise ValueError(f"Game id {game_id} no existe")
            logger.info("DEBUG MIGRATE: Using existing Game id=%s", game_id)

    return game_id, game_pgn

# ---------------------------------------------------------------------------
# Helper trasladado desde celery_app.analyze_game_task_legacy
# ---------------------------------------------------------------------------

def _is_aborted(task, player_username: str | None = None) -> bool:  # noqa: D401 (helper interno)
    """Detecta si la tarea Celery ha sido revocada.

    1. Comprueba primero una clave Redis específica del usuario (cancel:<username>)
       para reaccionar rápidamente a las cancelaciones a nivel de aplicación.
    2. Si el worker soporta :py:meth:`Task.is_aborted`, la utiliza.
    3. De forma retro-compatible inspecciona :pyattr:`task.request.is_aborted` o
       la marca *stopped*.
    """
    try:
        # Cancelación vía Redis (más rápida y simple)
        if player_username:
            cancellation_key = f"cancel:{player_username}"
            if redis_client.get(cancellation_key):
                logger.info(
                    "Task %s detected Redis cancellation flag for %s",
                    getattr(task.request, "id", "<no-id>"),
                    player_username,
                )
                return True

        # Celery >=5.3
        if hasattr(task, "is_aborted") and callable(task.is_aborted):
            return bool(task.is_aborted())

        # Celery <5.3 – intentar a través de request
        if hasattr(task.request, "is_aborted") and callable(task.request.is_aborted):
            return bool(task.request.is_aborted())
        # A veces Celery marca .stopped
        if getattr(task.request, "stopped", False):
            return True

    except Exception:
        # Nunca lanzar excepción desde chequeo de cancelación
        pass

    return False

# ---------------------------------------------------------------------------
# Helper: cache lookup
# ---------------------------------------------------------------------------

def _cached_result(pgn_text: str, depth: int, multipv: int, move_times, *, player: str | None):
    """Devuelve resultado cacheado o ``None``."""
    return cache_get("analyze_game_task", [pgn_text, depth, multipv, move_times], {"player": player})

# ---------------------------------------------------------------------------
# Implementación incremental del análisis de partida (fase 1)
# ---------------------------------------------------------------------------

def _analyze_game_task_impl(task, pgn_text: str, game_id: int | None = None, *,
                            move_times: list[int] | None = None,
                            player: str | None = None,
                            depth: int = MAX_DEPTH,
                            multipv: int = 3):
    """Fase 1 – solo normaliza Game y delega al legacy.

    Poco a poco reemplazaremos llamadas al legacy hasta eliminarlo.
    """
    # Cancelación rápida antes de consumir CPU
    if not task.request.called_directly and _is_aborted(task, player):
        return {"status": "revoked", "game_id": game_id}

    # --- Cache -----------------------------------------------------------
    cached = _cached_result(pgn_text, depth, multipv, move_times, player=player)
    if cached:
        return cached

    # Crear / obtener registro Game
    gid, game_pgn_obj = _prepare_game_record(pgn_text, game_id, move_times)  # noqa: F841 (se usará en próximas fases)
    assert gid is not None  # para mypy / linter

    # ── Inicialización de tablero y contadores (migrado) ──────────────────
    board = game_pgn_obj.board()
    total_moves = len(list(game_pgn_obj.mainline_moves()))
    progress_every = max(1, total_moves // 20)
    logger.info(
        "DEBUG MIGRATE: Game %s ready – total moves %s, update every %s",
        gid,
        total_moves,
        progress_every,
    )

    # ── Motor Stockfish (migrado) ─────────────────────────────────────────
    import chess.engine  # import local para reducir tiempo de carga global
    engine_sf = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    logger.info("DEBUG MIGRATE: Stockfish engine inicializado en %s", ENGINE_PATH)

    # ── Iteradores y estructuras de datos para análisis (migrado) ─────────
    times_iter = iter(move_times or [])
    analyses: list[models.MoveAnalysis] = []
    # Aún no llenamos 'analyses'; se hará en la siguiente extracción.

    # ── Bucle preliminar: sólo revocación y progreso ---------------------
    for idx, move in enumerate(game_pgn_obj.mainline_moves(), start=1):
        # Verificar revocación con alta frecuencia
        if not task.request.called_directly and _is_aborted(task, player):
            logger.info(
                "Task %s revoked mid-loop at move %s",
                getattr(task.request, "id", "<no-id>"),
                idx,
            )
            try:
                engine_sf.quit()
            except Exception:
                pass
            return {"status": "revoked", "game_id": gid, "moves_analyzed": idx - 1}

        # --- Evaluación con Stockfish -----------------------------------
        legal_cnt = board.legal_moves.count()

        info_before = engine_sf.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        eval_before = info_before[0]["score"].white().score(mate_score=100000) or 0
        best_move = info_before[0]["pv"][0]

        board.push(move)
        info_after = engine_sf.analyse(board, chess.engine.Limit(depth=depth))
        eval_after = info_after["score"].white().score(mate_score=100000) or 0
        board.pop()

        rank = next((i for i, pv in enumerate(info_before) if pv["pv"][0] == move), multipv)
        cp_loss = abs(eval_before - eval_after)
        time_spent = next(times_iter, None)

        if idx <= 5 or idx % 10 == 0:
            logger.info(
                "DEBUG MIGRATE: Move %s/%s – %s, cp_loss=%s, rank=%s, eval_before=%s, eval_after=%s, legal=%s",
                idx,
                total_moves,
                board.san(move),
                cp_loss,
                rank,
                eval_before,
                eval_after,
                legal_cnt,
            )

        analyses.append(models.MoveAnalysis(
            game_id=gid,
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

        # Progreso al cliente cada ~5 % ó al final
        if idx % progress_every == 0 or idx == total_moves:
            try:
                from app.utils import task_progress  # import local para evitar ciclos
                task_progress(task, idx, total_moves, player)
            except Exception:
                pass

    # --- Guardar análisis y metadatos en BD -----------------------------
    opening_key = " ".join(node.san() for i, node in enumerate(game_pgn_obj.mainline()) if i < 8)
    eco_code = game_pgn_obj.headers.get("eco_code", "Unknown")

    # Verificación de revocación antes de commit final
    if not task.request.called_directly and _is_aborted(task, player):
        try:
            engine_sf.quit()
        except Exception:
            pass
        return {"status": "revoked", "game_id": gid, "moves_analyzed": len(analyses)}

    with Session(engine) as s:
        game_db = s.get(models.Game, gid)
        if not game_db:
            logger.warning("Game %s no existe al guardar análisis", gid)
            return {"status": "cancelled", "game_id": gid, "reason": "game_deleted"}

        s.add_all(analyses)
        game_db.opening_key = opening_key
        game_db.eco_code = eco_code
        s.add(game_db)
        s.commit()

    # Cerrar motor
    try:
        engine_sf.quit()
    except Exception:
        pass

    result = {
        "status": "ok",
        "game_id": gid,
        "moves_analyzed": len(analyses),
        "eco_code": eco_code,
    }

    # Guardar en caché
    try:
        cache_set("analyze_game_task", [pgn_text, depth, multipv, move_times], {"player": player}, result)
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Wrappers actualizados
# ---------------------------------------------------------------------------

@celery_app.task(name="analyze_game_task", bind=True)
def analyze_game_task(self, pgn_text: str, game_id: int | None = None, *,
                      move_times: list[int] | None = None,
                      player: str | None = None,
                      depth: int = MAX_DEPTH,
                      multipv: int = 3):  # noqa: D401
    """Nueva entrada que usa la fase 1 de la implementación nativa."""

    return _analyze_game_task_impl(self, pgn_text, game_id, move_times=move_times,
                                   player=player, depth=depth, multipv=multipv)


@celery_app.task(name="analyze_game_detailed")
def analyze_game_detailed(*args, **kwargs):  # noqa: D401
    """Nueva entrada para análisis detallado.

    Paso 0 de migración: delega en la implementación incremental que a su
    vez (de momento) llama a la versión legacy. Así evitamos bucles y nos
    preparamos para trasladar la lógica real en etapas pequeñas.
    """
    return _analyze_game_detailed_impl(*args, **kwargs)


# ---------------------------------------------------------------------------
# Implementación incremental de análisis detallado (fase 0)
# ---------------------------------------------------------------------------

def _analyze_game_detailed_impl(game_id: int, username: str):
    """Fase 0 – simple passthrough hacia la versión legacy.

    Se crea para poder ir migrando la lógica desde
    ``celery_app.analyze_game_detailed_legacy``.
    """
    # 1) Cancelación vía Redis
    cancellation_key = f"cancel:{username}"
    if redis_client.get(cancellation_key):
        logger.info("analyze_game_detailed detectó cancelación para %s", username)
        return {"status": "cancelled", "game_id": game_id, "username": username}

    # 2) Caché rápida
    cached = cache_get("analyze_game_detailed", [game_id], {"username": username})
    if cached:
        return cached

    # 3) Por ahora delegamos la lógica pesada al legacy
    return analyze_game_detailed_legacy(game_id, username)

__all__ = [
    "analyze_game_task",
    "analyze_game_detailed",
] 