from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from app.analysis.quality import precision_bursts
from app.analysis.timing import clutch_accuracy

logger = logging.getLogger(__name__)


def _is_pause_series(move_time: pd.Series,
                     pause_sec: Tuple[float, float]) -> pd.Series:
    """
    Boolean mask for moves considered a 'pause' given time spent on that move.
    """
    mt = pd.to_numeric(move_time, errors="coerce")
    return (mt >= pause_sec[0]) & (mt <= pause_sec[1])


def detect_pause_then_perfect(
    game_df: pd.DataFrame,
    acpl_threshold_cp: int = 25,
    window_size: int = 3,
    pause_sec: Tuple[float, float] = (5.0, 12.0),
    clutch_threshold: float = 30.0,
) -> Dict[str, Any]:
    """
    Joint quality+timing investigative signal: "pause then perfect".

    Idea:
      - Find low-ACPL windows ("precision bursts") using evaluation deltas.
      - Flag those windows where the immediately preceding move is a 'pause'
        (move_time within [pause_sec[0], pause_sec[1]]).
      - Also report clutch_accuracy_diff to provide context about player's
        accuracy under time pressure.

    Expected columns:
      - For precision bursts (preferred): 'eval_cp_before', 'eval_cp_after'.
        If missing, no bursts will be found by precision_bursts() and count=0.
      - For timing: 'move_time' to detect pauses.
      - For clutch accuracy (optional): 'player_clock_before' and either
        'delta_eval' or (eval_cp_before/after) or 'is_engine_best'.

    Returns a dictionary with:
      - pause_then_perfect_count: number of detected sequences.
      - pause_then_perfect_windows: list of (start_idx, end_idx) tuples.
      - clutch_accuracy_diff: float (normal - clutch), or 0.0 if unavailable.
      - meta: parameters used for transparency.
    """
    if game_df is None or len(game_df) == 0:
        return {
            "pause_then_perfect_count": 0,
            "pause_then_perfect_windows": [],
            "clutch_accuracy_diff": 0.0,
            "meta": {
                "acpl_threshold_cp": int(acpl_threshold_cp),
                "window_size": int(window_size),
                "pause_sec": tuple(pause_sec),
                "clutch_threshold": float(clutch_threshold),
                "rows_total": 0,
                "rows_valid_time": 0,
            },
        }

    df = game_df.copy()

    # Compute precision bursts (quality windows)
    try:
        bursts: List[Tuple[int, int]] = precision_bursts(
            df, threshold_cp=acpl_threshold_cp, window_size=window_size
        )
    except Exception as e:
        logger.info("joint_signals: precision_bursts failed: %s", e)
        bursts = []

    # Pause mask (timing)
    has_move_time = "move_time" in df.columns
    pause_mask = _is_pause_series(df["move_time"], pause_sec) if has_move_time else pd.Series(False, index=df.index)
    valid_time_rows = int(pd.to_numeric(df["move_time"], errors="coerce").notna().sum()) if has_move_time else 0

    # Select windows preceded by a pause
    matched_windows: List[Tuple[int, int]] = []
    if bursts and has_move_time:
        for (start, end) in bursts:
            prev_idx = start - 1
            if prev_idx >= 0 and bool(pause_mask.iloc[prev_idx]):
                matched_windows.append((start, end))

    # Clutch accuracy context
    try:
        clutch_diff = float(clutch_accuracy(df, clutch_threshold=clutch_threshold))
    except Exception as e:
        logger.info("joint_signals: clutch_accuracy failed: %s", e)
        clutch_diff = 0.0

    result = {
        "pause_then_perfect_count": int(len(matched_windows)),
        "pause_then_perfect_windows": matched_windows,
        "clutch_accuracy_diff": clutch_diff,
        "meta": {
            "acpl_threshold_cp": int(acpl_threshold_cp),
            "window_size": int(window_size),
            "pause_sec": tuple(pause_sec),
            "clutch_threshold": float(clutch_threshold),
            "rows_total": int(df.shape[0]),
            "rows_valid_time": valid_time_rows,
            "precision_burst_count": int(len(bursts)),
        },
    }
    logger.info("joint_signals: result %s", result)
    return result
