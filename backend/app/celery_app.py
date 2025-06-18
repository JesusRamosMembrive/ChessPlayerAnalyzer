from __future__ import annotations

import os
from datetime import datetime, UTC, timezone
from statistics import stdev
import logging

from app import models
from app.database import engine
from celery import Celery
from sqlalchemy import or_
from sqlmodel import Session, select
from pathlib import Path
from app.utils import fetch_games, notify_ws
from celery_once import QueueOnce

from app.analysis.engine import ChessAnalysisEngine

from sqlmodel import Session
from celery import states
from celery.result import AsyncResult

from app.database import engine
from app import models
from app.utils import fetch_games, notify_ws            # :contentReference[oaicite:0]{index=0}


REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("chess_tasks", broker=REDIS_URL, backend=REDIS_URL)

ENGINE_PATH = os.getenv("STOCKFISH_PATH", "stockfish")
MAX_DEPTH = int(os.getenv("STOCKFISH_DEPTH", "12"))

@celery_app.task(name="compute_game_metrics")
def compute_game_metrics(game_id: int):
    """Calcula métricas básicas y las graba en GameMetrics."""
    from statistics import mean

    with Session(engine) as session:
        game = session.get(models.Game, game_id)
        if not game or not game.moves:
            raise ValueError("Game or moves not found")

        ranks = [m.best_rank for m in game.moves]
        cp_losses = [m.cp_loss for m in game.moves]
        n = len(ranks)
        pct_top1 = sum(1 for r in ranks if r == 0) / n * 100.0
        pct_top3 = sum(1 for r in ranks if r <= 2) / n * 100.0
        acl = mean(cp_losses) if cp_losses else 0.0

        suspicious = (pct_top3 > 85 and acl < 20)

        times = game.move_times or []

        # valores por defecto para que siempre existan
        sigma_total: float | None = None
        constant_time = False
        pause_spike = False

        if times:
            sigma_total = stdev(times) if len(times) > 1 else 0.0
            constant_time = sigma_total < 1.0

            # pausa + pico
            T_PAUSE = 10
            pause_index = next((i for i, t in enumerate(times) if t > T_PAUSE), None)
            if pause_index is not None and pause_index + 5 < len(game.moves):
                ranks_after = [m.best_rank for m in game.moves[pause_index + 1: pause_index + 6]]
                pct_top3_after = sum(r <= 2 for r in ranks_after) / 5 * 100
                pause_spike = pct_top3_after >= 80
        else:
            sigma_total = None

        gm = models.GameMetrics(
            game_id=game_id,
            pct_top1=pct_top1,
            pct_top3=pct_top3,
            acl=acl,
            sigma_total=sigma_total,
            constant_time=constant_time,
            pause_spike=pause_spike,
            suspicious=suspicious or constant_time or pause_spike,
        )
        session.add(gm)
        session.commit()

        compute_player_metrics.delay(game.white_username)
        compute_player_metrics.delay(game.black_username)

    return {"game_id": game_id, "pct_top3": pct_top3, "acl": acl}

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
    • Al final dispara compute_game_metrics y actualiza claves de apertura.
    """
    import io, chess.pgn, chess.engine
    from sqlmodel import Session

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

    # ---------- 4.  Encolar métricas ------------------------------
    compute_game_metrics.delay(game_id)

    if player:
        pl = s.get(models.Player, player)
        if pl:
            pl.done_games += 1
            pl.progress = int(pl.done_games / pl.total_games * 100)
            # ¿terminado?
            if pl.done_games == pl.total_games:
                pl.status = "ready"
                pl.finished_at = datetime.now(UTC)
                notify_ws(player, {"status": "ready"})
            else:
                notify_ws(
                    player,
                    {"progress": pl.progress, "status": "pending"},
                )
            s.add(pl)
            s.commit()

    return {
        "game_id": game_id,
        "move_count": len(analyses),
        "analyzed_at": datetime.now(UTC).isoformat(),
    }


@celery_app.task(name="compute_player_metrics")
def compute_player_metrics(username: str, months: int = 12):
    """
    Recalcula las métricas agregadas del jugador:
        • nº de partidas en BD
        • entropía de aperturas
        • apertura más frecuente
        • flag de baja entropía
    """
    from collections import Counter
    from math import log2
    from sqlmodel import Session, select

    with Session(engine) as s:
        # --- extraemos las aperturas almacenadas -------------------
        rows = (
            s.exec(
                select(models.Game.opening_key)
                .where(
                    (models.Game.white_username == username)
                    | (models.Game.black_username == username)
                )
                .where(models.Game.opening_key.is_not(None))
            )
            .all()
        )
        openings = [r[0] for r in rows]          # <─ cada fila es (opening_key,)

        total = len(openings)
        if total < 5:            # aún no hay muestra suficiente
            return

        # --- estadísticas ------------------------------------------
        counts = Counter(openings)
        probs  = [c / total for c in counts.values()]
        entropy = -sum(p * log2(p) for p in probs)
        most_played = counts.most_common(1)[0][0]
        low_entropy = entropy < 1.0

        # --- upsert en PlayerMetrics -------------------------------
        pm = (
            s.exec(select(models.PlayerMetrics)
                   .where(models.PlayerMetrics.username == username))
            .first()
        )
        if pm is None:
            pm = models.PlayerMetrics(username=username)

        pm.game_count = total
        pm.opening_entropy = entropy
        pm.most_played = most_played
        pm.low_entropy = low_entropy
        pm.updated_at = datetime.now(UTC)

        s.add(pm)
        s.commit()


@celery_app.task(
    name="process_player",
    bind=True,           # para acceder a self.request
    acks_late=True,      # la tarea sólo se ACKea cuando acaba
    max_retries=3,       # reintentos automáticos (p.ej. si Chess.com da 5xx)
    default_retry_delay=30
)
def process_player(self, username: str, months: int = 6) -> dict:
    """
    Descarga las partidas recientes de *username*, crea/actualiza el registro
    Player y encola una tarea `analyze_game_task` por cada partida.

    – Idempotente: si ya hay un Player.pending REAL no hace nada.
    – Trazabilidad: guarda el `last_task_id` para poder comprobar si la tarea
      sigue viva desde el endpoint.
    – Robusto: cualquier excepción deja al jugador en estado 'error' con mensaje.
    """
    from celery.exceptions import Reject

    try:
        # ────────────────────────── 1.  Pre-chequeos ─────────────────────────

        with Session(engine) as s:
            pl = s.get(models.Player, username)
            if pl and pl.status == "pending":
                # ¿de verdad hay una tarea viva?
                if pl.last_task_id:
                    result = AsyncResult(pl.last_task_id)
                    # PENDING puede significar que la tarea nunca existió
                    # Verificamos si realmente está en el broker
                    if result.state == states.STARTED:
                        # Definitivamente está ejecutándose
                        return {"username": username, "status": "already_processing"}
                    elif result.state == states.PENDING:
                        # Verificar si la tarea existe en el backend
                        try:
                            # Si no existe, info será None
                            task_info = result.info
                            if task_info is not None:
                                return {"username": username, "status": "already_processing"}
                        except Exception:
                            # La tarea no existe, continuamos
                            pass

                # Verificar si el análisis está colgado (más de 30 minutos sin actualización)
                if pl.requested_at:
                    # Asegurar que ambos datetimes tengan timezone
                    now = datetime.now(UTC)
                    requested_at = pl.requested_at
                    if requested_at.tzinfo is None:
                        # Si requested_at es naive, asumimos UTC
                        requested_at = requested_at.replace(tzinfo=UTC)

                    time_since_start = (now - requested_at).total_seconds()
                    if time_since_start > 1800:  # 30 minutos
                        # Análisis colgado, lo reiniciamos
                        logging.warning(f"Análisis colgado para {username}, reiniciando...")
                    else:
                        # Aún dentro del tiempo límite
                        return {"username": username, "status": "already_processing", "time_elapsed": time_since_start}
            # si estamos aquí relanzaremos análisis
        # ───────────────────────── 2.  Download PGNs ─────────────────────────
        games = fetch_games(username, months)      # puede lanzar excepción
        total = len(games)

        # ─────────────────────── 3.  Registrar Player ───────────────────────
        with Session(engine) as s:
            pl = s.get(models.Player, username) or models.Player(username=username)

            if total == 0:
                pl.status = "ready"
                pl.progress = 100
                pl.total_games = 0
                pl.done_games = 0
                pl.requested_at = datetime.now(UTC)
                pl.finished_at = datetime.now(UTC)
                pl.error = None
                pl.last_task_id = self.request.id
                s.add(pl)
                s.commit()
                notify_ws(username, {"status": "ready", "progress": 100})
                return {"username": username, "games_queued": 0}

            pl.status = "pending"
            pl.progress = 0
            pl.total_games = total
            pl.done_games = 0
            pl.requested_at = datetime.now(UTC)
            pl.finished_at = None
            pl.error = None
            pl.last_task_id = self.request.id
            s.add(pl)
            s.commit()

        notify_ws(username, {"status": "pending", "progress": 0})

        # ─────────────────────── 4.  Encolar partidas ───────────────────────
        for g in games:
            analyze_game_task.delay(
                g["pgn"],
                None,                     # game_id lo crea el task
                move_times=g["move_times"],
                player=username
            )

        return {
            "username": username,
            "games_queued": total,
            "task_id": self.request.id,
            "status": "pending",
        }

    # ─────────────────────────── 5.  Errores ────────────────────────────────
    except Exception as exc:
        with Session(engine) as s:
            pl = s.get(models.Player, username) or models.Player(username=username)
            pl.status = "error"
            pl.error = str(exc)
            pl.finished_at = datetime.now(UTC)
            pl.progress = 0
            pl.last_task_id = self.request.id
            s.add(pl)
            s.commit()

        notify_ws(username, {"status": "error", "error": str(exc)})
        # rechazamos la tarea sin re-queue si se trata de un 4xx de Chess.com
        raise Reject(exc, requeue=isinstance(exc, (ConnectionError, TimeoutError)))



# Inicializar el motor de análisis (configurar rutas según tu sistema)
analysis_engine = ChessAnalysisEngine(
    reference_book_path=Path("/data/reference_book.bin") if Path("/data/reference_book.bin").exists() else None,
    tablebase_path=Path("/data/syzygy") if Path("/data/syzygy").exists() else None,
)


@celery_app.task(name="analyze_game_detailed")
def analyze_game_detailed(game_id: int):
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

        return {
            "game_id": game_id,
            "acpl": detailed_analysis.acpl,
            "suspicious_score": detailed_analysis.overall_suspicion_score,
            "analyzed_at": detailed_analysis.analyzed_at.isoformat()
        }

    except Exception as e:
        logging.error(f"Error en análisis detallado de partida {game_id}: {e}")
        raise


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
                .join(Game)
                .where(
                    (Game.white_username == username) |
                    (Game.black_username == username)
                )
            ).one()

            if analyzed_count < 10:
                logger.warning(f"Insuficientes partidas analizadas para {username}: {analyzed_count}")
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
        logger.error(f"Error en análisis detallado de jugador {username}: {e}")
        raise


# Modificar process_player existente para incluir el nuevo análisis
@celery_app.task(name="process_player_enhanced")
def process_player_enhanced(username: str, months: int = 6):
    """
    Versión mejorada de process_player que incluye análisis detallado.
    """
    games = fetch_games(username, months)
    total = len(games)

    with Session(engine) as s:
        player = models.Player(
            username=username,
            status="pending",
            requested_at=datetime.now(UTC),
            progress=0,
            total_games=total,
            done_games=0,
        )
        s.merge(player)
        s.commit()

    # Fase 1: Análisis básico (existente)
    game_ids = []
    for i, g in enumerate(games):
        # Crear partida
        with Session(engine) as s:
            game_db = models.Game(
                pgn=g["pgn"],
                move_times=g.get("move_times"),
                white_username=g.get("white"),
                black_username=g.get("black"),
                white_elo=g.get("white_elo"),  # Nuevo
                black_elo=g.get("black_elo"),  # Nuevo
            )
            s.add(game_db)
            s.commit()
            s.refresh(game_db)
            game_ids.append(game_db.id)

        # Análisis básico
        analyze_game_task.delay(
            g["pgn"],
            game_db.id,
            move_times=g.get("move_times"),
            player=username
        )

        # Actualizar progreso (50% para análisis básico)
        progress = int((i + 1) / total * 50)
        update_progress(username, progress)

    # Fase 2: Análisis detallado (nuevo)
    # Esperar un poco para que termine el análisis básico
    analyze_game_detailed.apply_async(
        args=[game_ids],
        countdown=60  # Esperar 1 minuto
    )

    # Fase 3: Análisis del jugador
    analyze_player_detailed.apply_async(
        args=[username],
        countdown=300  # Esperar 5 minutos
    )

    return {
        "username": username,
        "games_queued": total,
        "enhanced_analysis": True
    }


from celery.signals import task_failure

@task_failure.connect
def on_task_failure(sender=None, task_id=None, args=None, kwargs=None, **k):
    if sender.name == "process_player":
        username = args[0]
        with Session(engine) as s:
            pl = s.get(models.Player, username)
            if pl:
                pl.status = "error"
                pl.error = str(k.get("exception", "unknown"))
                s.add(pl); s.commit()
        notify_ws(username, {"status": "error"})