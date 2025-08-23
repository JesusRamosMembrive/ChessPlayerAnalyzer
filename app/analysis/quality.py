# quality_metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from typing import Tuple, List
from numpy.typing import NDArray
import logging
logger = logging.getLogger(__name__)


import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure repository root is on the Python path so imports like ``app.*`` work
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from app.utils_debugging.tracer import trace


def eval_to_wdl_prob(evaluation_in_cp: float, k: float = 400.0) -> float:
    """
    Convierte una evaluación en centipawns a una probabilidad de WDL (Win/Draw/Loss).
    Utiliza una función sigmoide estándar. El resultado es un valor entre 0 y 1,
    que representa el resultado esperado de la partida (1=victoria, 0.5=tablas, 0=derrota).

    Args:
        evaluation_in_cp: La evaluación de la posición en centipawns.
        k: El factor de escala. Un valor de 400 es estándar y corresponde
           a la expectativa de que una ventaja de 400cp (4 peones) da una
           probabilidad de victoria muy alta.

    Returns:
        La probabilidad WDL como un flotante entre 0 y 1.
    """
    return 1.0 / (1.0 + 10 ** (-evaluation_in_cp / k))


###############################################################################
# 1.  Average Centipawn Loss (ACPL)   #########################################
###############################################################################
@trace
def acpl(game_df: pd.DataFrame, player_color: str = 'white', cap_cp: int | None = 1500, use_median: bool = True) -> float:
    """
    Calcula el Average Centipawn Loss (ACPL) usando la media o mediana.
    La mediana es más robusta a outliers (blunders), dando una visión más
    estable de la calidad de juego típica de un jugador.

    Args:
        game_df: DataFrame con datos de la partida. Debe contener `delta_eval`.
        player_color: Color del jugador ('white' o 'black'), usado en fallbacks.
        cap_cp: Límite superior para `delta_eval` en centipawns. Previene que las
                evaluaciones de mate (ej. 100000cp) distorsionen la métrica.
                Se recomienda un valor como 1500. `None` para desactivar.
        use_median: Si es True, usa la mediana en lugar de la media.

    Returns:
        ACPL como flotante.
    """
    if "delta_eval" in game_df.columns:
        # Usar la pérdida vs. la mejor jugada del motor directamente
        vals = pd.to_numeric(game_df["delta_eval"], errors="coerce").abs().dropna()

        # Aplicar cap para robustez frente a outliers (mates)
        if cap_cp is not None:
            vals = vals.clip(upper=cap_cp)

        if vals.empty:
            return 0.0

        agg_func = np.median if use_median else np.mean
        result = float(agg_func(vals))
        agg_name = "median" if use_median else "mean"
        logger.info(f"DEBUG QUALITY: ACPL calculated from 'delta_eval' (L1 loss, cap={cap_cp}, agg={agg_name}): {result:.2f} over {len(vals)} moves.")
        return result

    # --- Fallback si 'delta_eval' no está ---
    required_cols = {"eval_cp_before", "eval_cp_after"}
    if not required_cols.issubset(game_df.columns):
        logger.info("DEBUG QUALITY: ACPL fallback unavailable (missing eval columns); returning 0.0")
        return 0.0

    eval_before = pd.to_numeric(game_df["eval_cp_before"], errors="coerce")
    eval_after = pd.to_numeric(game_df["eval_cp_after"], errors="coerce")

    if player_color == 'black':
        eval_before = -eval_before
        eval_after = -eval_after

    diffs = (eval_after - eval_before).abs().dropna()

    if diffs.empty:
        return 0.0

    agg_func = np.median if use_median else np.mean
    result = float(agg_func(diffs))
    agg_name = "median" if use_median else "mean"
    logger.warning(
        f"ACPL calculated using fallback (eval swing) because 'delta_eval' was missing. "
        f"Count: {len(diffs)}, Agg: {agg_name}, Result: {result:.2f}"
    )
    return result


@trace
def wdl_loss(game_df: pd.DataFrame, player_color: str = 'white') -> float:
    """
    Calcula la pérdida de probabilidad de WDL (Win/Draw/Loss) para un jugador.

    Esta métrica es más robusta que el ACPL porque es insensible a blunders
    en posiciones ya perdidas o ganadas. Una pérdida de 50cp importa mucho
    más en una posición igualada que en una con +10 de ventaja.

    Args:
        game_df: DataFrame con datos de la partida. Debe contener
                 `eval_cp_before` y `eval_cp_after`.
        player_color: Color del jugador ('white' o 'black').

    Returns:
        Pérdida de WDL media como flotante.
    """
    required_cols = {"eval_cp_before", "eval_cp_after"}
    if not required_cols.issubset(game_df.columns):
        logger.warning("WDL loss calculation requires 'eval_cp_before' and 'eval_cp_after'.")
        return 0.0

    eval_before = pd.to_numeric(game_df["eval_cp_before"], errors="coerce")
    eval_after = pd.to_numeric(game_df["eval_cp_after"], errors="coerce")

    # Flip evaluations for black player so that positive is always good for the player
    if player_color == 'black':
        eval_before = -eval_before
        eval_after = -eval_after

    wdl_prob_before = eval_to_wdl_prob(eval_before)
    wdl_prob_after = eval_to_wdl_prob(eval_after)

    # Loss is the difference in win probability
    wdl_loss_per_move = (wdl_prob_before - wdl_prob_after).dropna()

    if wdl_loss_per_move.empty:
        return 0.0

    result = float(wdl_loss_per_move.mean())
    logger.info(f"DEBUG QUALITY: WDL loss calculated for {player_color}: {result:.4f} over {len(wdl_loss_per_move)} moves.")
    return result


@trace
def robust_loss(
    game_df: pd.DataFrame,
    cap_cp: int | None = 1000,
    trim_pct: float | None = 0.1,
    use_median: bool = False,
) -> float:
    """
    Calcula una métrica de pérdida robusta (mediana o media recortada)
    para delta_eval, limitando el impacto de outliers.

    Args:
        game_df: DataFrame con la columna 'delta_eval'.
        cap_cp: Límite superior para delta_eval antes de agregar.
        trim_pct: Porcentaje (0.0-1.0) de valores a recortar de cada
                  extremo si no se usa la mediana.
        use_median: Si es True, calcula la mediana; de lo contrario,
                    usa la media recortada.

    Returns:
        La métrica de pérdida robusta calculada.
    """
    if "delta_eval" not in game_df.columns:
        return 0.0

    vals = pd.to_numeric(game_df["delta_eval"], errors="coerce").abs().dropna()

    if cap_cp is not None:
        vals = vals.clip(upper=cap_cp)

    if vals.empty:
        return 0.0

    if use_median:
        return float(np.median(vals))

    # Usar media recortada si no es mediana
    if trim_pct is not None and 0 < trim_pct < 0.5:
        return trimmed_mean(vals, trim_pct)

    return float(np.mean(vals))


def trimmed_mean(series: pd.Series, trim_pct: float) -> float:
    """
    Calcula la media de una serie después de eliminar un porcentaje
    de los valores más pequeños y más grandes.
    """
    if not isinstance(series, pd.Series) or series.empty:
        return 0.0

    # Ordenar la serie para recortar los extremos
    sorted_series = series.sort_values()
    n = len(sorted_series)
    trim_count = int(n * trim_pct)

    # Recortar y calcular la media
    trimmed_series = sorted_series.iloc[trim_count : n - trim_count]

    if trimmed_series.empty:
        return 0.0

    return float(trimmed_series.mean())


###############################################################################
# 2.  ACPL ajustado al rating  #################################################
###############################################################################
@trace
class ACPLModel:
    """
    Ajusta una curva de referencia ACPL_expected(ELO) con un robust regressor
    sobre un conjunto grande de partidas 'limpias' y produce z‑scores.
    """

    def __init__(self):
        self.model = HuberRegressor()  # menos sensible a outliers
        self.sigma_: float | None = None  # desviación típica residual

    def fit(self, df_stats: pd.DataFrame) -> ACPLModel:
        """
        df_stats debe tener columnas: ['elo', 'acpl']
        """
        # Usar to_numpy con tipo explícito para robustez
        X: NDArray[np.float64] = df_stats[['elo']].to_numpy(dtype=float)
        y: NDArray[np.float64] = df_stats['acpl'].to_numpy(dtype=float)
        self.model.fit(X, y)

        # La predicción y los residuos deben usar los mismos tipos para evitar errores
        y_pred: NDArray[np.float64] = self.model.predict(X)
        residuals: NDArray[np.float64] = y - y_pred
        self.sigma_ = np.std(residuals, ddof=1)
        return self

    def z_score(self, elo: float, acpl_value: float) -> float:
        """
        z > +2,75 ≈ umbral FIDE de evidencia estadística.
        """
        if self.sigma_ is None:
            # El modelo debe ser fiteado antes de poder usarse
            raise ValueError("ACPLModel must be fitted before calling z_score.")

        mu: float = self.model.predict([[elo]])[0]
        return (mu - acpl_value) / self.sigma_

###############################################################################
# 3.  Intrinsic Performance Rating (IPR) ######################################
###############################################################################
@trace
def intrinsic_performance_rating(match_pct: float, acpl: float,
                                 coef_match: float = 800,
                                 coef_acpl: float = -0.5) -> float:
    """
    Aproximación lineal al modelo de Regan.
    – match_pct: % de jugadas que coinciden con la 1ª línea del motor (0‑1).
    – acpl: Average Centipawn Loss.

    Devuelve un rating ELO estimado que explicaría esa precisión.
    Los coeficientes se obtienen calibrando sobre tu base de datos de referencia.
    """
    return coef_match * match_pct + coef_acpl * acpl + 2000  # offset base

@trace
def ipr_z_score(ipr: float, elo: float, sigma: float = 60) -> float:
    """
    Desviación típica (~60 ELO) tomada de los papers de Regan.
    """
    return (ipr - elo) / sigma


###############################################################################
# 4.  Coincidencia ponderada por complejidad ##################################
###############################################################################
@trace
def complexity_weighted_match(game_df: pd.DataFrame,
                              max_moves_cap: int = 50) -> float:
    """
    % de coincidencia con Stockfish ponderado por complejidad.
    Si todos los pesos salen 0 (p.ej. legal_moves == max_moves_cap en
    todos los lances) se hace fallback a la media simple para evitar
    ZeroDivisionError.
    """
    if "is_engine_best" not in game_df.columns:
        return 0.0

    # Peso inverso a la complejidad: +difícil ⇒ +peso si acierta
    weights = np.log1p(max_moves_cap - game_df.legal_moves.clip(0, max_moves_cap)
                       if "legal_moves" in game_df.columns
                       else max_moves_cap)

    total_w = weights.sum()
    if total_w == 0 or np.isnan(total_w):
        # fallback seguro
        return game_df.is_engine_best.mean()

    return np.dot(game_df.is_engine_best, weights) / total_w
###############################################################################
# 5.  Detección de rachas de precisión ########################################
###############################################################################
@trace
def precision_bursts(game_df: pd.DataFrame,
                     threshold_cp: int = 25,
                     window_size: int = 5) -> List[Tuple[int, int]]:
    """
    Devuelve una lista de (move_index_start, move_index_end) donde
    el ACPL por jugada en la ventana < threshold_cp (p. ej. 10 cp).

    Ideal para encontrar momentos donde el jugador parece 'consultar' motor.
    """
    required_cols = {"eval_cp_before", "eval_cp_after"}
    if not required_cols.issubset(game_df.columns):
        return []  # Return empty list when engine data is missing

    # Usar to_numpy() para obtener un array de tipo concreto y predecible
    diffs: NDArray[np.float64] = np.abs(
        game_df["eval_cp_after"] - game_df["eval_cp_before"]
    ).to_numpy(dtype=float)
    bursts = []
    for i in range(len(diffs) - window_size + 1):
        window = diffs[i:i + window_size]
        if window.mean() < threshold_cp:
            bursts.append((i, i + window_size - 1))
    return bursts


BLUNDER_THRESHOLD = 300  # cp
@trace
def compute_phase_quality(moves_df_list: list[pd.DataFrame]) -> dict:
    """
    Agrega un bloque estandarizado de calidad por fase a nivel jugador.
    Retorna SIEMPRE las claves:
      - opening_acpl, middlegame_acpl, endgame_acpl (robustas: |delta_eval| con cap y mediana)
      - opening_blunder_rate, middlegame_blunder_rate, endgame_blunder_rate
    y mantiene 'blunder_rate' global para compatibilidad.

    Cada moves_df debe tener columnas:
      • 'phase'  ('opening' | 'middlegame' | 'endgame')
      • 'delta_eval'  (cp vs best)
    """

    # Helper para salida consistente
    def _empty_phase_block():
        return {
            "opening_acpl": None,
            "middlegame_acpl": None,
            "endgame_acpl": None,
            "opening_blunder_rate": None,
            "middlegame_blunder_rate": None,
            "endgame_blunder_rate": None,
            "blunder_rate": None,
        }

    if not moves_df_list:
        return _empty_phase_block()

    combined = pd.concat(moves_df_list, ignore_index=True)

    # Validación de columnas requeridas
    if "phase" not in combined.columns or "delta_eval" not in combined.columns:
        return _empty_phase_block()

    # Valores robustos: |delta_eval| con cap para evitar outliers del final
    vals = pd.to_numeric(combined["delta_eval"], errors="coerce").abs()
    vals = vals.clip(upper=1500)  # cap robusto consistente con acpl()

    tmp = pd.DataFrame({
        "phase": combined["phase"],
        "delta": vals,
    }).dropna()

    # ACPL robusto por fase (mediana)
    if tmp.empty:
        phase_acpl = {}
    else:
        phase_acpl = tmp.groupby("phase")["delta"].median().to_dict()

    # Tasa de blunders por fase y global
    is_blunder = tmp["delta"] > BLUNDER_THRESHOLD if not tmp.empty else pd.Series(dtype=bool)
    if not tmp.empty:
        tmp2 = tmp.assign(is_blunder=is_blunder)
        phase_blunders = tmp2.groupby("phase")["is_blunder"].mean().to_dict()
        overall_blunder_rate = float(is_blunder.mean())
    else:
        phase_blunders = {}
        overall_blunder_rate = None

    return {
        "opening_acpl": float(phase_acpl.get("opening")) if "opening" in phase_acpl else None,
        "middlegame_acpl": float(phase_acpl.get("middlegame")) if "middlegame" in phase_acpl else None,
        "endgame_acpl": float(phase_acpl.get("endgame")) if "endgame" in phase_acpl else None,
        "opening_blunder_rate": float(phase_blunders.get("opening")) if "opening" in phase_blunders else None,
        "middlegame_blunder_rate": float(phase_blunders.get("middlegame")) if "middlegame" in phase_blunders else None,
        "endgame_blunder_rate": float(phase_blunders.get("endgame")) if "endgame" in phase_blunders else None,
        "blunder_rate": overall_blunder_rate,
    }
@trace
def aggregate_clutch_accuracy(games_df):
    if "clutch_accuracy_diff" not in games_df:
        return {}

    diffs = games_df["clutch_accuracy_diff"].dropna().abs()
    if diffs.empty:
        return {}

    avg_diff = float(diffs.mean())
    pct_good = float((diffs < 100).mean())  # “bueno” si <100 cp

    return {
        "avg_clutch_diff": round(avg_diff, 1),
        "clutch_games_pct": round(pct_good, 3),
    }
@trace
def aggregate_tactical_trends(games_df: pd.DataFrame) -> dict:
    # Si no hay ninguna de las dos columnas, devolver dict vacío
    if all(col not in games_df for col in ["precision_burst_count", "second_choice_rate"]):
        return {}

    burst = (
        games_df["precision_burst_count"].sum(min_count=1)
        if "precision_burst_count" in games_df else np.nan
    )
    scr = (
        games_df["second_choice_rate"].mean(skipna=True)
        if "second_choice_rate" in games_df else np.nan
    )

    return {
        "precision_burst_count": int(burst) if not np.isnan(burst) else None,
        "second_choice_rate": float(scr) if not np.isnan(scr) else None,
    }


@trace
def phase_blunder_rate_single(game_df: pd.DataFrame) -> dict:
    """
    Calcula la tasa de blunders por fase para una sola partida.
    """
    if "phase" not in game_df.columns or "delta_eval" not in game_df.columns:
        return {}

    is_blunder = game_df["delta_eval"].abs() > BLUNDER
    temp_df = game_df.assign(is_blunder=is_blunder)

    phase_rates = temp_df.groupby("phase")["is_blunder"].mean().to_dict()
    overall_blunder_rate = float(is_blunder.mean())

    return {
        "opening_blunder_rate":    float(phase_rates.get("opening", np.nan)),
        "middlegame_blunder_rate": float(phase_rates.get("middlegame", np.nan)),
        "endgame_blunder_rate":    float(phase_rates.get("endgame", np.nan)),
        "blunder_rate":            overall_blunder_rate,
    }


BLUNDER = 300  # cp
@trace
def aggregate_blunders_by_phase(moves_dfs: list[pd.DataFrame]) -> dict:

    if not moves_dfs:
        return {}

    df = pd.concat(moves_dfs, ignore_index=True)
    if "phase" not in df or "delta_eval" not in df:
        return {}

    df["is_blunder"] = df["delta_eval"].abs() > BLUNDER

    phase_rates = (
        df.groupby("phase")["is_blunder"]
          .mean()
          .to_dict()
    )

    return {
        "opening_blunder_rate":  float(phase_rates.get("opening", np.nan)),
        "middlegame_blunder_rate": float(phase_rates.get("middlegame", np.nan)),
        "endgame_blunder_rate":   float(phase_rates.get("endgame", np.nan)),
        "blunder_rate":           float(df["is_blunder"].mean()),
    }

# ─────────────────────────────────────────────────────────────────────────
#  🔗  AGGREGATOR
# ------------------------------------------------------------------------
@trace
def phase_acpl_single(game_df: pd.DataFrame, cap_cp: int | None = 1500) -> dict:
    if "phase" not in game_df.columns or "delta_eval" not in game_df.columns:
        return {}
    vals = pd.to_numeric(game_df["delta_eval"], errors="coerce").abs()
    if cap_cp is not None:
        vals = vals.clip(upper=cap_cp)
    tmp = pd.DataFrame({"phase": game_df["phase"], "delta": vals}).dropna()
    if tmp.empty:
        return {}
    grp = tmp.groupby("phase")["delta"].mean()
    return {
        "opening_acpl": float(grp.get("opening", np.nan)),
        "middlegame_acpl": float(grp.get("middlegame", np.nan)),
        "endgame_acpl": float(grp.get("endgame", np.nan)),
    }

@trace
def aggregate_quality_features(game_df, elo: int | None = None, player_color: str = 'white') -> dict:
    logger.info("DEBUG QUALITY: Starting quality features calculation")
    logger.info(f"DEBUG QUALITY: Input DataFrame shape: {game_df.shape}")
    logger.info(f"DEBUG QUALITY: Input DataFrame columns: {list(game_df.columns)}")
    logger.info(f"DEBUG QUALITY: ELO parameter: {elo}")

    # Excluir filas donde la evaluación del motor no está disponible
    original_count = len(game_df)
    valid_mask = pd.Series(True, index=game_df.index)

    # La fuente de verdad es 'delta_eval' si existe, si no, las evaluaciones
    if 'delta_eval' in game_df.columns:
        valid_mask = pd.to_numeric(game_df['delta_eval'], errors='coerce').notna()
    elif 'eval_cp_before' in game_df.columns and 'eval_cp_after' in game_df.columns:
        valid_mask = pd.to_numeric(game_df['eval_cp_before'], errors='coerce').notna() & \
                     pd.to_numeric(game_df['eval_cp_after'], errors='coerce').notna()

    if (~valid_mask).any():
        game_df = game_df[valid_mask]
        excluded_count = original_count - len(game_df)
        logger.info(f"DEBUG QUALITY: Excluded {excluded_count} of {original_count} rows due to missing/invalid engine evaluations.")


    # Check for effective depth and warn if below target
    TARGET_DEPTH = 12
    if 'depth' in game_df.columns:
        # Ensure depth column is numeric and handle non-numeric gracefully
        depth_series = pd.to_numeric(game_df['depth'], errors='coerce').dropna()

        if not depth_series.empty:
            avg_effective_depth = depth_series.mean()
            logger.info(f"DEBUG QUALITY: Effective analysis depth found. Average: {avg_effective_depth:.2f} over {len(depth_series)} moves.")

            if avg_effective_depth < TARGET_DEPTH:
                logger.warning(
                    f"Shallow analysis warning: "
                    f"Average effective depth ({avg_effective_depth:.2f}) is below target depth ({TARGET_DEPTH}). "
                    f"Results may be less reliable."
                )
        else:
            logger.info("DEBUG QUALITY: 'depth' column found, but contains no valid numeric data.")

    match_rate = (
        game_df["is_engine_best"].mean() if "is_engine_best" in game_df else 0.0
    )
    logger.info(f"DEBUG QUALITY: Match rate: {match_rate}")

    acpl_val = acpl(game_df, player_color)
    logger.info(f"DEBUG QUALITY: ACPL value: {acpl_val}")

    weighted_match = complexity_weighted_match(game_df)
    logger.info(f"DEBUG QUALITY: Weighted match rate: {weighted_match}")

    ipr_val = intrinsic_performance_rating(match_rate, acpl_val)
    logger.info(f"DEBUG QUALITY: IPR value: {ipr_val}")

    wdl_loss_val = wdl_loss(game_df, player_color)
    logger.info(f"DEBUG QUALITY: WDL loss value: {wdl_loss_val}")

    feats = {
        "acpl"               : acpl_val,
        "wdl_loss"           : wdl_loss_val,
        "match_rate"         : match_rate,
        "weighted_match_rate": weighted_match,
        "ipr"                : ipr_val,
        # ipr_z_score con valor neutro por defecto
        "ipr_z_score"        : 0.0,
    }

    if elo is not None:
        ipr_z = ipr_z_score(feats["ipr"], elo)
        feats["ipr_z_score"] = ipr_z
        logger.info(f"DEBUG QUALITY: IPR Z-score: {ipr_z}")
    else:
        logger.info("DEBUG QUALITY: No ELO provided, IPR Z-score remains 0.0")

    # --- Quality Score ---
    # El quality_score es un indicador sintético (0-100) que combina:
    # 1. ACPL (pérdida media de centipawns): Aportando un 40%.
    #    - Se normaliza: un ACPL de 0 es 1.0, y un ACPL >= 100 es 0.0.
    #    - Un ACPL bajo (ej. < 20) es típico de maestros.
    #    - Un ACPL alto (ej. > 100) es típico de principiantes.
    # 2. Tasa de coincidencias con el motor (match_rate): Aportando un 30%.
    # 3. Tasa de coincidencias ponderada por complejidad: Aportando un 30%.

    # Normaliza el ACPL a un rango [0, 1] para el score.
    # Un ACPL de 100 o más se considera de calidad mínima (0 puntos).
    acpl_scaled = 1 - min(max(acpl_val, 0), 100) / 100

    quality_score = (
        40 * acpl_scaled +        # Menos ACPL es mejor
        30 * match_rate +         # Más jugadas exactas es mejor
        30 * weighted_match      # Precisión ponderada por complejidad
    )
    feats["quality_score"] = quality_score
    logger.info(f"DEBUG QUALITY: Quality score (acpl_scaled={acpl_scaled:.2f}): {quality_score:.2f}")

    burst_count = len(precision_bursts(game_df))
    feats["precision_burst_count"] = burst_count
    logger.info(f"DEBUG QUALITY: Precision burst count: {burst_count}")

    if "phase" in game_df.columns and "delta_eval" in game_df.columns:
        pacpl = phase_acpl_single(game_df)
        if pacpl:
            feats.update(pacpl)
            logger.info(f"DEBUG QUALITY: Phase ACPL added: {pacpl}")

        blunder_rates = phase_blunder_rate_single(game_df)
        if blunder_rates:
            feats.update(blunder_rates)
            logger.info(f"DEBUG QUALITY: Phase blunder rates added: {blunder_rates}")

    logger.info(f"DEBUG QUALITY: Final quality features: {feats}")
    return feats

###############################################################################
# 6.  Uso de ejemplo ##########################################################
###############################################################################

if __name__ == "__main__":
    # ── Ejemplo mínimo con un DataFrame ficticio ──────────────────────────
    df_moves = pd.DataFrame({
        'move_number': np.arange(1, 41),
        'eval_cp_before': np.random.randint(-200, 200, 40),
        'eval_cp_after': np.random.randint(-200, 200, 40),
        'legal_moves': np.random.randint(5, 40, 40),
        'is_engine_best': np.random.rand(40) < 0.35,
    })

    logger.info("ACPL partida: %s", acpl(df_moves))
    logger.info("Match ponderado: %s", complexity_weighted_match(df_moves))
    logger.info("Bursts: %s", precision_bursts(df_moves))

# How to implement
# Group by player and calculate
# player_stats = df_moves.groupby('player').apply(
#     lambda g: pd.Series({
#         'acpl': acpl(g),
#         'match_w': complexity_weighted_match(g),
#         'games': g.game_id.nunique(),
#         # etc.
#     })
# )
# Adjusts the ACPL_expected(ELO)rinse_quality curve.
# model = ACPLModel().fit(reference_stats)    # reference_stats: elo, acpl
# player_stats['acpl_z'] = player_stats.apply(
#     lambda r: model.z_score(r['elo'], r['acpl']), axis=1
# )
# Calculate IPR and z-score:Group by player and calculate
# player_stats['ipr'] = intrinsic_performance_rating(
#     player_stats['match_w'], player_stats['acpl']
# )
# player_stats['ipr_z'] = ipr_z_score(player_stats['ipr'], player_stats['elo'])
