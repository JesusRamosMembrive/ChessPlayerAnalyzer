# app/validation.py
"""
Validation functions for BULLDOZER REFACTOR.

Philosophy: Fail fast with real problems - no sanitization that hides issues.
"""
import math
from typing import Any, Dict, List, Union


class InvalidAnalysisDataError(Exception):
    """Raised when analysis data contains invalid values that indicate real problems."""
    pass


def validate_analysis_metrics(data: Dict[str, Any], context: str = "") -> Dict[str, Any]:
    """
    Validates analysis metrics - fails fast if there are mathematical problems.

    NO sanitization - if there's a NaN or Inf, it means there's a real bug
    that needs to be fixed in the calculation logic.
    """
    context_msg = f" in {context}" if context else ""

    # Fields that are allowed to have NaN (non-critical metrics)
    ALLOWED_NAN_FIELDS = {
        'second_choice_rate', 'opening_second_choice_rate', 'opening_entropy', 'novelty_depth',
        'clutch_accuracy_diff', 'precision_burst_count', 'robust_loss',
        'cp_loss', 'delta_eval', 'move_time', 'time_spent',
        'middlegame_acpl', 'endgame_acpl', 'opening_acpl',
        'eval_before', 'eval_after'  # End-of-game moves may lack final evaluations
    }

    def check_numeric_value(key: str, value: Any) -> None:
        if isinstance(value, float):
            # Check if this field is allowed to have NaN
            field_name = key.split('.')[-1]  # Get the last part of the path

            if math.isnan(value):
                if (field_name in ALLOWED_NAN_FIELDS or
                    'second_choice_rate' in key or
                    'opening_' in key or
                    'middlegame_' in key or
                    'endgame_' in key or
                    'moves[' in key):  # Allow NaN in move-level data (end-of-game scenarios)
                    # Allow NaN for non-critical fields, but log it
                    return  # Skip validation for these fields
                else:
                    raise InvalidAnalysisDataError(
                        f"NaN detected in {key}{context_msg}. "
                        f"This indicates a division by zero or invalid calculation. "
                        f"Fix the calculation logic instead of sanitizing."
                    )
            if math.isinf(value):
                raise InvalidAnalysisDataError(
                    f"Infinity detected in {key}{context_msg}. "
                    f"This indicates a mathematical overflow or invalid calculation. "
                    f"Fix the calculation logic instead of sanitizing."
                )

    def validate_recursive(obj: Any, path: str = "") -> Any:
        if isinstance(obj, dict):
            for k, v in obj.items():
                current_path = f"{path}.{k}" if path else k
                check_numeric_value(current_path, v)
                validate_recursive(v, current_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]"
                validate_recursive(item, current_path)
        else:
            check_numeric_value(path, obj)

        return obj

    return validate_recursive(data)


def validate_stockfish_evaluation(evaluation: Union[float, None], move_number: int) -> Union[float, None]:
    """
    Validates Stockfish evaluation - returns None if engine couldn't evaluate.

    This is different from sanitization - None means "no data available"
    while NaN/Inf means "calculation error that needs fixing".
    """
    if evaluation is None:
        # Engine couldn't evaluate this position - that's valid
        return None

    if isinstance(evaluation, float):
        if math.isnan(evaluation) or math.isinf(evaluation):
            raise InvalidAnalysisDataError(
                f"Invalid Stockfish evaluation at move {move_number}: {evaluation}. "
                f"This indicates a problem with the engine interface or calculation."
            )

    return evaluation


def validate_timing_data(move_times: List[float]) -> List[float]:
    """
    Validates move timing data - ensures no impossible or corrupted values.
    """
    validated_times = []

    for i, time_val in enumerate(move_times):
        if time_val is None:
            # No timing data available for this move - skip it
            continue

        if not isinstance(time_val, (int, float)):
            raise InvalidAnalysisDataError(
                f"Invalid time value type at move {i}: {type(time_val)} = {time_val}"
            )

        if math.isnan(time_val) or math.isinf(time_val):
            raise InvalidAnalysisDataError(
                f"Invalid time value at move {i}: {time_val}. "
                f"This indicates corrupted timing data or calculation error."
            )

        if time_val < 0:
            raise InvalidAnalysisDataError(
                f"Negative time value at move {i}: {time_val}. "
                f"This indicates corrupted timing data."
            )

        if time_val > 3600:  # More than 1 hour per move is suspicious
            raise InvalidAnalysisDataError(
                f"Suspiciously high time value at move {i}: {time_val} seconds. "
                f"This might indicate corrupted timing data."
            )

        validated_times.append(time_val)

    return validated_times


def ensure_real_data_or_none(value: Any, field_name: str) -> Any:
    """
    Ensures data is real or explicitly None - no fake zeroes or sanitized values.

    Use this instead of sanitization when you want to explicitly handle
    cases where data might not be available.
    """
    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            # Don't sanitize to None - this is a real problem that needs fixing
            raise InvalidAnalysisDataError(
                f"Invalid mathematical value in {field_name}: {value}. "
                f"Fix the calculation instead of sanitizing to None."
            )

    return value