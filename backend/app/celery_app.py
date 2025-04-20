import os, io, chess.pgn, chess.engine
from datetime import datetime, UTC
from typing import List
from celery import Celery
from sqlmodel import Session

from app.database import engine
from app import models

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("chess_tasks", broker=REDIS_URL, backend=REDIS_URL)

ENGINE_PATH = os.getenv("STOCKFISH_PATH", "stockfish")
MAX_DEPTH = int(os.getenv("STOCKFISH_DEPTH", "12"))

@celery_app.task(name="compute_game_metrics")
def compute_game_metrics(game_id: int):
    """Calcula métricas básicas y las graba en GameMetrics."""
    from statistics import mean
    import statistics as stats

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
        if times:
            sigma_total = stats.stdev(times) if len(times) > 1 else 0
            constant_time = sigma_total < 1.0

            # pausa + pico
            T_PAUSE = 10
            pause_index = next((i for i, t in enumerate(times) if t > T_PAUSE), None)
            pause_spike = False
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
    return {"game_id": game_id, "pct_top3": pct_top3, "acl": acl}

@celery_app.task(name="analyze_game_task")
def analyze_game_task(pgn_text: str, game_id: int, depth: int = MAX_DEPTH, multipv: int = 3):
    """Analiza la partida usando Stockfish con *multipv* y guarda rank y cp_loss.

    • *rank*  → posición (0‑based) de la jugada entre las `multipv` mejores.
      Si no aparece, se registra multipv (p.ej. 3 ⇒ «fuera del top‑3»).
    • *cp_loss*  → diferencia absolutizada entre la evaluación
      del mejor movimiento y la jugada elegida.
    """
    # --- Recupera el objeto Game --------------------------------------------------
    with Session(engine) as session:
        game_db = session.get(models.Game, game_id)
        if not game_db:
            raise ValueError("Game not found in DB")

    # --- Analiza PGN --------------------------------------------------------------
    import io, chess.pgn, chess.engine
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Invalid PGN")

    engine_sf = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    board = game.board()
    moves_for_db: list[models.MoveAnalysis] = []

    for idx, move in enumerate(game.mainline_moves(), 1):
        infos = engine_sf.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        # "infos" es lista ordenada PV1, PV2… cuando multipv>1
        best_eval = infos[0]
        best_move = best_eval["pv"][0]
        best_score = best_eval["score"].white().score(mate_score=100000)

        # Lista de los primeros movimientos de cada PV
        top_moves = [inf["pv"][0] for inf in infos]
        try:
            rank = top_moves.index(move)  # 0 = mejor
        except ValueError:
            rank = multipv  # fuera del top‑multipv

        # Evalúa tras la jugada del jugador para estimar cp_loss
        board.push(move)
        after_info = engine_sf.analyse(board, chess.engine.Limit(depth=depth))
        after_score = after_info["score"].white().score(mate_score=100000)
        board.pop()
        cp_loss = abs((best_score or 0) - (after_score or 0))

        moves_for_db.append(models.MoveAnalysis(
            game_id=game_id,
            move_number=idx,
            played=board.san(move),
            best=board.san(best_move),
            best_rank=rank,
            cp_loss=int(cp_loss),
        ))

        board.push(move)

    engine_sf.quit()

    with Session(engine) as session:
        session.add_all(moves_for_db)
        session.commit()

    # Encola el cálculo de métricas en una cadena
    compute_game_metrics.delay(game_id)

    return {
        "game_id": game_id,
        "move_count": len(moves_for_db),
        "analyzed_at": datetime.now(UTC).isoformat(),
    }