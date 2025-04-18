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


@celery_app.task(name="analyze_game_task")
def analyze_game_task(pgn_text: str, game_id: int, depth: int = MAX_DEPTH):
    game_record = None
    with Session(engine) as session:
        game_record = session.get(models.Game, game_id)
        if not game_record:
            raise ValueError("Game not found in DB")

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Invalid PGN")

    engine_sf = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    board = game.board()
    moves: List[models.MoveAnalysis] = []

    for idx, move in enumerate(game.mainline_moves(), 1):
        info = engine_sf.analyse(board, chess.engine.Limit(depth=depth))
        best = info["pv"][0]
        rank = 0 if move == best else 1
        cp_loss = abs(info["score"].relative.score(mate_score=100000))
        moves.append(
            models.MoveAnalysis(
                game_id=game_id,
                move_number=idx,
                played=board.san(move),
                best=board.san(best),
                best_rank=rank,
                cp_loss=cp_loss,
            )
        )
        board.push(move)

    engine_sf.quit()

    with Session(engine) as session:
        session.add_all(moves)
        session.commit()

    return {
        "game_id": game_id,
        "move_count": len(moves),
        "analyzed_at": datetime.now(UTC).isoformat(),
    }
