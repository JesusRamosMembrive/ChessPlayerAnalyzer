#!/usr/bin/env python3
import argparse
import json
import io
from pathlib import Path
from datetime import datetime, timezone
import sys
import importlib.util
import numpy as np
import pandas as pd
import chess.pgn

REPO_ROOT = Path(__file__).resolve().parents[1]

def _load_module(module_name: str, rel_path: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
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
    feats_q_user = {}
    feats_q_white = {}
    feats_q_black = {}
    feats_o = {}
    try:
        feats_t = timing.aggregate_time_features(mv_df)
    except Exception as e:
        feats_t = {"_timing_error": str(e)}
    user_color = None
    if username:
        if meta.get("white") and str(meta.get("white")).lower() == str(username).lower():
            user_color = "white"
        elif meta.get("black") and str(meta.get("black")).lower() == str(username).lower():
            user_color = "black"
    try:
        if user_color:
            feats_q_user = quality.aggregate_quality_features(mv_df, player_color=user_color)
        feats_q_white = quality.aggregate_quality_features(mv_df, player_color="white")
        feats_q_black = quality.aggregate_quality_features(mv_df, player_color="black")
    except Exception as e:
        feats_q_user = feats_q_user or {"_quality_error": str(e)}
    try:
        if "best_rank" not in mv_df.columns:
            mv_df = mv_df.assign(best_rank=np.nan)
        if "delta_eval" not in mv_df.columns:
            mv_df = mv_df.assign(delta_eval=np.nan)
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
    def _ns(prefix, d):
        return {f"{prefix}_{k}": v for k, v in (d or {}).items()}
    out = {
        **meta,
        **feats_t,
        **_ns("white", feats_q_white),
        **_ns("black", feats_q_black),
        **feats_o,
    }
    if user_color and feats_q_user:
        out["player_color_username"] = user_color
        out.update(_ns("user", feats_q_user))
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
                "acpl": feats.get("user_acpl") if "user_acpl" in feats else feats.get("white_acpl"),
                "match_rate": feats.get("user_match_rate") if "user_match_rate" in feats else feats.get("white_match_rate"),
                "weighted_match_rate": feats.get("user_weighted_match_rate") if "user_weighted_match_rate" in feats else feats.get("white_weighted_match_rate"),
                "ipr": feats.get("user_ipr") if "user_ipr" in feats else feats.get("white_ipr"),
                "quality_score": feats.get("user_quality_score") if "user_quality_score" in feats else feats.get("white_quality_score"),
                "time_complexity_corr": feats.get("time_complexity_corr"),
                "lag_spike_count": feats.get("lag_spike_count"),
                "eco_code": feats.get("eco_code"),
                "opening_key": feats.get("opening_key"),
                "precision_burst_count": feats.get("user_precision_burst_count") if "user_precision_burst_count" in feats else feats.get("white_precision_burst_count"),
                "second_choice_rate": feats.get("second_choice_rate"),
            }
        )
    games_df = pd.DataFrame(per_for_long) if per_for_long else pd.DataFrame()
    aggregates = {}
    try:
        if not games_df.empty:
            aggregates.update(longitudinal.aggregate_longitudinal_features(games_df) or {})
    except Exception as e:
        aggregates["_longitudinal_error"] = str(e)
    try:
        aggregates.update(quality.aggregate_tactical_trends(games_df) or {})
    except Exception as e:
        aggregates["_tactical_error"] = str(e)
    try:
        aggregates.update(quality.aggregate_clutch_accuracy(games_df) or {})
    except Exception as e:
        aggregates["_clutch_error"] = str(e)
    out = {
        "input_file": str(path),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "games_count": len(games),
        "per_game": per,
        "aggregates": aggregates,
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
    ap.add_argument("--no-color-summary", action="store_true")
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
        acpl_vals = _clean([x.get("user_acpl", x.get("white_acpl")) for x in per])
        wmatch_vals = _clean([x.get("user_weighted_match_rate", x.get("white_weighted_match_rate")) for x in per])
        ipr_vals = _clean([x.get("user_ipr", x.get("white_ipr")) for x in per])
        qscore_vals = _clean([x.get("user_quality_score", x.get("white_quality_score")) for x in per])
        tcc_vals = _clean([x.get("time_complexity_corr") for x in per])
        lags_vals = _clean([x.get("lag_spike_count") for x in per])

        w_acpl_vals = _clean([x.get("white_acpl") for x in per])
        b_acpl_vals = _clean([x.get("black_acpl") for x in per])
        w_match_vals = _clean([x.get("white_weighted_match_rate") for x in per])
        b_match_vals = _clean([x.get("black_weighted_match_rate") for x in per])
        w_mrate_vals = _clean([x.get("white_match_rate") for x in per])
        b_mrate_vals = _clean([x.get("black_match_rate") for x in per])

        mean_move_t = float(np.nanmean(mt_vals)) if mt_vals else float("nan")
        mean_acpl = float(np.nanmean(acpl_vals)) if acpl_vals else float("nan")
        mean_wmatch = float(np.nanmean(wmatch_vals)) if wmatch_vals else float("nan")
        mean_ipr = float(np.nanmean(ipr_vals)) if ipr_vals else float("nan")
        mean_qscore = float(np.nanmean(qscore_vals)) if qscore_vals else float("nan")
        mean_tcc = float(np.nanmean(tcc_vals)) if tcc_vals else float("nan")
        sum_lags = int(np.nansum(lags_vals)) if lags_vals else 0

        w_acpl = float(np.nanmean(w_acpl_vals)) if w_acpl_vals else float("nan")
        b_acpl = float(np.nanmean(b_acpl_vals)) if b_acpl_vals else float("nan")
        w_wmatch = float(np.nanmean(w_match_vals)) if w_match_vals else float("nan")
        b_wmatch = float(np.nanmean(b_match_vals)) if b_match_vals else float("nan")
        w_mrate = float(np.nanmean(w_mrate_vals)) if w_mrate_vals else float("nan")
        b_mrate = float(np.nanmean(b_mrate_vals)) if b_mrate_vals else float("nan")

        mmts = "NaN" if not mt_vals else f"{mean_move_t:.2f}"
        macpl = "NaN" if not acpl_vals else f"{mean_acpl:.1f}"
        mwmatch = "NaN" if not wmatch_vals else f"{mean_wmatch:.3f}"
        mipr = "NaN" if not ipr_vals else f"{mean_ipr:.0f}"
        mqscore = "NaN" if not qscore_vals else f"{mean_qscore:.1f}"
        mtcc = "NaN" if not tcc_vals else f"{mean_tcc:.3f}"

        msg = (
            f"[OK] {t.name}: games={n} mean_move_time={mmts}s acpl={macpl} "
            f"w_match={mwmatch} ipr={mipr} qscore={mqscore} "
            f"t_complexity_corr={mtcc} lag_spikes={sum_lags}"
        )

        if not args.no_color_summary:
            w_acpl_s = "NaN" if not w_acpl_vals else f"{w_acpl:.1f}"
            b_acpl_s = "NaN" if not b_acpl_vals else f"{b_acpl:.1f}"
            w_wmatch_s = "NaN" if not w_match_vals else f"{w_wmatch:.3f}"
            b_wmatch_s = "NaN" if not b_match_vals else f"{b_wmatch:.3f}"
            w_mrate_s = "NaN" if not w_mrate_vals else f"{w_mrate:.3f}"
            b_mrate_s = "NaN" if not b_mrate_vals else f"{b_mrate:.3f}"
            msg += (
                f" | white: acpl={w_acpl_s}, w_match={w_wmatch_s}, match_rate={w_mrate_s}"
                f" | black: acpl={b_acpl_s}, w_match={b_wmatch_s}, match_rate={b_mrate_s}"
            )

        msg += f" -> {out_path}"
        print(msg)


if __name__ == "__main__":
    main()
