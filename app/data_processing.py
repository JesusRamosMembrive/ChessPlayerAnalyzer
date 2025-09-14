"""
Utilidades de procesamiento de datos optimizadas.
Reemplaza operaciones pandas básicas con NumPy para mejor performance.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


def prepare_moves_array(game, username: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Versión optimizada de prepare_moves_dataframe usando NumPy arrays.
    Reemplaza pandas DataFrame con arrays estructurados para mejor performance.

    Returns:
        Dict con arrays NumPy para cada campo de movimiento
    """
    if not game.moves:
        return {
            "move_number": np.array([], dtype=np.int32),
            "played": np.array([], dtype=object),
            "best_rank": np.array([], dtype=np.float32),
            "cp_loss": np.array([], dtype=np.float32),
            "eval_cp_before": np.array([], dtype=np.float32),
            "eval_cp_after": np.array([], dtype=np.float32),
            "move_time": np.array([], dtype=np.float32),
            "legal_moves": np.array([], dtype=np.int32),
            "player_clock_before": np.array([], dtype=np.float32),
            "is_engine_best": np.array([], dtype=bool),
            "player_color": np.array([], dtype=object),
            "phase": np.array([], dtype=object),
        }

    player_color = None
    if username:
        if game.white_username == username:
            player_color = 'white'
        elif game.black_username == username:
            player_color = 'black'

    # Filtrar movimientos si es necesario
    moves = []
    initial_time = 600.0  # 10 minutos

    for i, m in enumerate(game.moves):
        if player_color and (i % 2 == 0) != (player_color == 'white'):
            continue

        # Calcular tiempo restante
        current_clock = initial_time
        if game.move_times and len(game.move_times) > 0:
            accumulated_time = 0.0
            for j, time_change in enumerate(game.move_times):
                if (player_color == 'white' and j % 2 == 0) or (player_color == 'black' and j % 2 == 1):
                    if j < i:
                        accumulated_time += abs(time_change)
            current_clock = max(0.0, initial_time - accumulated_time)

        moves.append({
            "move_number": m.move_number,
            "played": m.played,
            "best_rank": m.best_rank or np.nan,
            "cp_loss": m.cp_loss or np.nan,
            "eval_cp_before": m.eval_before or np.nan,
            "eval_cp_after": m.eval_after or np.nan,
            "move_time": m.time_spent or 0.0,
            "legal_moves": m.legal_moves_count or 0,
            "player_clock_before": current_clock,
            "is_engine_best": (m.best_rank or 1) == 0,
            "player_color": player_color,
        })

    if not moves:
        return prepare_moves_array(game, None)  # Return empty structure

    # Convertir a arrays NumPy
    n_moves = len(moves)
    result = {}

    # Arrays numéricos
    result["move_number"] = np.array([m["move_number"] for m in moves], dtype=np.int32)
    result["best_rank"] = np.array([m["best_rank"] for m in moves], dtype=np.float32)
    result["cp_loss"] = np.array([m["cp_loss"] for m in moves], dtype=np.float32)
    result["eval_cp_before"] = np.array([m["eval_cp_before"] for m in moves], dtype=np.float32)
    result["eval_cp_after"] = np.array([m["eval_cp_after"] for m in moves], dtype=np.float32)
    result["move_time"] = np.array([m["move_time"] for m in moves], dtype=np.float32)
    result["legal_moves"] = np.array([m["legal_moves"] for m in moves], dtype=np.int32)
    result["player_clock_before"] = np.array([m["player_clock_before"] for m in moves], dtype=np.float32)
    result["is_engine_best"] = np.array([m["is_engine_best"] for m in moves], dtype=bool)

    # Arrays de strings/objetos
    result["played"] = np.array([m["played"] for m in moves], dtype=object)
    result["player_color"] = np.array([m["player_color"] for m in moves], dtype=object)

    # Calcular fases de juego
    total_moves = n_moves
    opening_cut = int(total_moves * 0.25)
    endgame_cut = int(total_moves * 0.80)

    phases = np.empty(n_moves, dtype=object)
    phases[:opening_cut] = "opening"
    phases[opening_cut:endgame_cut] = "middlegame"
    phases[endgame_cut:] = "endgame"
    result["phase"] = phases

    # Agregar delta_eval
    valid_cp_loss = ~np.isnan(result["cp_loss"])
    if np.any(valid_cp_loss):
        result["delta_eval"] = result["cp_loss"].copy()
    else:
        # Calcular desde eval_before/after si está disponible
        before_valid = ~np.isnan(result["eval_cp_before"])
        after_valid = ~np.isnan(result["eval_cp_after"])
        if np.any(before_valid & after_valid):
            result["delta_eval"] = np.abs(result["eval_cp_before"] - result["eval_cp_after"])
        else:
            result["delta_eval"] = np.full(n_moves, np.nan, dtype=np.float32)

    return result


def aggregate_basic_metrics(values: np.ndarray, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calcula métricas básicas de agregación sin pandas.
    Reemplaza operaciones pandas básicas como mean(), std(), etc.
    """
    if len(values) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "count": 0,
        }

    # Filtrar valores válidos (no NaN)
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]

    if len(valid_values) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "count": 0,
        }

    result = {
        "count": len(valid_values),
        "min": float(np.min(valid_values)),
        "max": float(np.max(valid_values)),
        "median": float(np.median(valid_values)),
    }

    if weights is not None and len(weights) == len(values):
        valid_weights = weights[valid_mask]
        if len(valid_weights) > 0 and np.sum(valid_weights) > 0:
            result["mean"] = float(np.average(valid_values, weights=valid_weights))
            # Weighted standard deviation
            weighted_mean = result["mean"]
            weighted_var = np.average((valid_values - weighted_mean) ** 2, weights=valid_weights)
            result["std"] = float(np.sqrt(weighted_var))
        else:
            result["mean"] = float(np.mean(valid_values))
            result["std"] = float(np.std(valid_values))
    else:
        result["mean"] = float(np.mean(valid_values))
        result["std"] = float(np.std(valid_values))

    return result


def compute_percentiles(values: np.ndarray, percentiles: List[float] = [25, 50, 75, 90, 95]) -> Dict[str, float]:
    """
    Calcula percentiles sin pandas.
    """
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return {f"p{int(p)}": np.nan for p in percentiles}

    return {
        f"p{int(p)}": float(np.percentile(valid_values, p))
        for p in percentiles
    }


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """
    Calcula media móvil sin pandas.
    """
    if len(values) < window:
        return np.full_like(values, np.nan, dtype=np.float32)

    result = np.full_like(values, np.nan, dtype=np.float32)

    for i in range(window - 1, len(values)):
        window_values = values[i - window + 1:i + 1]
        valid_mask = ~np.isnan(window_values)
        if np.any(valid_mask):
            result[i] = np.mean(window_values[valid_mask])

    return result


def group_by_phase(moves_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Agrupa los datos de movimientos por fase de juego.
    Reemplaza DataFrame.groupby() con operaciones NumPy.
    """
    if "phase" not in moves_data or len(moves_data["phase"]) == 0:
        return {}

    phases = ["opening", "middlegame", "endgame"]
    result = {}

    for phase in phases:
        phase_mask = moves_data["phase"] == phase
        if not np.any(phase_mask):
            continue

        phase_data = {}
        for key, values in moves_data.items():
            if isinstance(values, np.ndarray):
                phase_data[key] = values[phase_mask]

        result[phase] = phase_data

    return result


def safe_divide(numerator: Union[np.ndarray, float], denominator: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    División segura que maneja división por cero.
    """
    if isinstance(denominator, np.ndarray):
        result = np.full_like(denominator, np.nan, dtype=np.float32)
        valid_mask = (denominator != 0) & ~np.isnan(denominator)
        if isinstance(numerator, np.ndarray):
            valid_mask &= ~np.isnan(numerator)
            result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        else:
            if not np.isnan(numerator):
                result[valid_mask] = numerator / denominator[valid_mask]
        return result
    else:
        if denominator == 0 or np.isnan(denominator):
            return np.nan
        return numerator / denominator


def correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula coeficiente de correlación sin pandas.
    """
    if len(x) != len(y):
        return np.nan

    # Filtrar valores válidos en ambos arrays
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(valid_mask) < 2:
        return np.nan

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calcula promedio ponderado manajando casos edge.
    """
    valid_mask = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
    if not np.any(valid_mask):
        return np.nan

    valid_values = values[valid_mask]
    valid_weights = weights[valid_mask]

    return float(np.average(valid_values, weights=valid_weights))


def find_outliers_iqr(values: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Detecta outliers usando el método IQR.
    """
    valid_values = values[~np.isnan(values)]
    if len(valid_values) < 4:
        return np.zeros_like(values, dtype=bool)

    q1 = np.percentile(valid_values, 25)
    q3 = np.percentile(valid_values, 75)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    return (values < lower_bound) | (values > upper_bound)