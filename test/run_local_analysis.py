#!/usr/bin/env python3
import argparse
import json
import io
from pathlib import Path
from datetime import datetime
import sys
import importlib.util
import numpy as np
import pandas as pd
import chess.pgn

REPO_ROOT = Path(__file__).resolve().parents[1]

def _load_module(module_name: str, rel_path: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

timing = _load_module("timing", "app/analysis/timing.py")
quality = _load_module("quality", "app/analysis/quality.py")
openings = _load_module("openings", "app/analysis/openings.py")
longitudinal = _load_module("longitudinal", "app/analysis/longitudinal.py")
def _to_native(obj):
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (pd.Series,)):
        return obj.to_dict()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return obj.total_seconds()
    if isinstance(obj, (set,)):
        return list(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def load_json_any(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("JSON must be a dict or a list")


def parse_pgn(pgn_text: str):
    game = chess.pgn.read_game(io.StringIO(pgn_text or ""))
    return game or chess.pgn.Game()


def extract_meta_from_pgn(game_obj, fallback_white=None, fallback_black=None):
    headers = game_obj.headers
    eco = headers.get("ECO")
    result = headers.get("Result")
    time_control = headers.get("TimeControl")
    white = headers.get("White") or fallback_white
    black = headers.get("Black") or fallback_black
    return eco, result, time_control, white, black


def derive_opening_key_from_moves(moves_df: pd.DataFrame, max_moves: int = 12):
    if "played" not in moves_df:
        return None
    seq = moves_df["played"].tolist()[:max_moves]
    return " ".join(seq) if seq else None


def label_phases(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    if total == 0:
        df["phase"] = []
        return df
    opening_cut = int(total * 0.25)
    endgame_cut = int(total * 0.80)
    df = df.reset_index(drop=True)
    df["phase"] = np.select(
        [df.index <= opening_cut, df.index >= endgame_cut],
        ["opening", "endgame"],
        default="middlegame",
    )
    return df


def reconstruct_player_clock(times: list[float], time_control: str | None) -> list[float | None]:
    if not time_control:
        return [None] * len(times)
    parts = str(time_control).split("+")
    try:
        base = int(parts[0])
        inc = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        return [None] * len(times)
    clocks = []
    white_elapsed = 0.0
    black_elapsed = 0.0
    white_moves = 0
    black_moves = 0
    for i, t in enumerate(times):
        if i % 2 == 0:
            before = base + inc * white_moves - white_elapsed
            clocks.append(max(0.0, float(before)) if before == before else None)
            white_elapsed += float(t) if t == t else 0.0
            white_moves += 1
        else:
            before = base + inc * black_moves - black_elapsed
            clocks.append(max(0.0, float(before)) if before == before else None)
            black_elapsed += float(t) if t == t else 0.0
            black_moves += 1
    return clocks


def moves_df_from_game_object(game_obj: dict, reconstruct_clock: bool = False) -> tuple[pd.DataFrame, dict]:
    pgn_text = game_obj.get("pgn", "") or ""
    move_times = game_obj.get("move_times") or []
    times = [abs(float(x)) if x is not None else np.nan for x in move_times]
    pgn_game = parse_pgn(pgn_text)
    eco_code, result, time_control, white, black = extract_meta_from_pgn(
        pgn_game, fallback_white=game_obj.get("white"), fallback_black=game_obj.get("black")
    )
    board = pgn_game.board()
    rows = []
    san_list = []
    legal_counts = []
    for i, move in enumerate(pgn_game.mainline_moves()):
        legal_counts.append(board.legal_moves.count())
        san = board.san(move)
        san_list.append(san)
        t_spent = times[i] if i < len(times) else np.nan
        rows.append(
            {
                "move_number": i + 1,
                "played": san,
                "legal_moves": legal_counts[-1],
                "move_time": t_spent,
            }
        )
        board.push(move)
    df = pd.DataFrame(rows)
    if "legal_moves" not in df:
        df = df.assign(legal_moves=0)
    if "move_time" not in df:
        df = df.assign(move_time=np.nan)
    df = label_phases(df)
    if reconstruct_clock and len(times) == len(df):
        clocks = reconstruct_player_clock(times, time_control)
        df["player_clock_before"] = clocks
    meta = {
        "eco_code": eco_code,
        "result": result,
        "white": white,
        "black": black,
        "end_time": game_obj.get("end_time"),
        "opening_key": derive_opening_key_from_moves(df),
    }
    return df, meta


def per_game_features(mv_df: pd.DataFrame, meta: dict, username: str | None):
    feats_t = {}
    feats_q = {}
    feats_o = {}
    try:
        feats_t = timing.aggregate_time_features(mv_df)
    except Exception as e:
        feats_t = {"_timing_error": str(e)}
    try:
        feats_q = quality.aggregate_quality_features(mv_df)
    except Exception as e:
        feats_q = {"_quality_error": str(e)}
    try:
        one_games_df = pd.DataFrame(
            [
                {
                    "eco_code": meta.get("eco_code"),
                    "opening_key": meta.get("opening_key"),
                }
            ]
        )
        ok = meta.get("opening_key") or derive_opening_key_from_moves(mv_df)
        feats_o = openings.aggregate_opening_features(ok or "", meta.get("eco_code"), mv_df, one_games_df)
    except Exception as e:
        feats_o = {"_openings_error": str(e)}
    out = {**meta, **feats_t, **feats_q, **feats_o}
    return out


def analyze_input_file(path: Path, username: str | None, reconstruct_clock: bool):
    games = load_json_any(path)
    per = []
    per_for_long = []
    for g in games:
        mv_df, meta = moves_df_from_game_object(g, reconstruct_clock=reconstruct_clock)
        feats = per_game_features(mv_df, meta, username)
        per.append(feats)
        per_for_long.append(
            {
                "acpl": feats.get("acpl"),
                "match_rate": feats.get("match_rate"),
                "weighted_match_rate": feats.get("weighted_match_rate"),
                "time_complexity_corr": feats.get("time_complexity_corr"),
                "eco_code": feats.get("eco_code"),
                "opening_key": feats.get("opening_key"),
            }
        )
    games_df = pd.DataFrame(per_for_long) if per_for_long else pd.DataFrame()
    try:
        agg = longitudinal.aggregate_longitudinal_features(games_df) if not games_df.empty else {}
    except Exception as e:
        agg = {"_longitudinal_error": str(e)}
    out = {
        "input_file": str(path),
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "games_count": len(games),
        "per_game": per,
        "aggregates": agg,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Run local analysis without DB/Celery")
    ap.add_argument("--input", "-i")
    ap.add_argument("--input-dir", "-d")
    ap.add_argument("--pattern", "-p", default="*.json")
    ap.add_argument("--out-dir", "-o", default="test/out/results")
    ap.add_argument("--username", "-u")
    ap.add_argument("--reconstruct-clock", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets: list[Path] = []
    if args.input:
        targets = [Path(args.input)]
    elif args.input_dir:
        targets = sorted(Path(args.input_dir).glob(args.pattern))
    else:
        ap.error("Provide --input or --input-dir")

    for t in targets:
        try:
            res = analyze_input_file(t, args.username, args.reconstruct_clock)
        except Exception as e:
            print(f"[ERROR] {t.name}: {e}")
            continue
        out_path = out_dir / (t.stem + ".results.json")
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=_to_native), encoding="utf-8")
        per = res.get("per_game", [])
        n = len(per)
        def _clean(vals):
            out = []
            for v in vals:
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if fv == fv and np.isfinite(fv):
                    out.append(fv)
            return out
        mt_vals = _clean([x.get("mean_move_time") for x in per])
        acpl_vals = _clean([x.get("acpl") for x in per])
        match_vals = _clean([x.get("match_rate") for x in per])
        mean_move_t = float(np.nanmean(mt_vals)) if mt_vals else float("nan")
        mean_acpl = float(np.nanmean(acpl_vals)) if acpl_vals else float("nan")
        mean_match = float(np.nanmean(match_vals)) if match_vals else float("nan")
        mmts = "NaN" if not mt_vals else f"{mean_move_t:.2f}"
        macpl = "NaN" if not acpl_vals else f"{mean_acpl:.2f}"
        mmatch = "NaN" if not match_vals else f"{mean_match:.3f}"
        print(
            f"[OK] {t.name}: games={n} mean_move_time={mmts} acpl={macpl} match_rate={mmatch} -> {out_path}"
        )


if __name__ == "__main__":
    main()
