import numpy as np
import pandas as pd
from typing import Dict, Iterable, Any


def difference_in_differences(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    outcome_col: str,
    treated_value: Any,
    pre_period: Any,
    post_period: Any,
) -> float:
    """Compute classic difference-in-differences estimator.

    Parameters
    ----------
    df : DataFrame
        Dataset containing group, time and outcome columns.
    group_col : str
        Column indicating treatment vs control groups.
    time_col : str
        Column with time period labels.
    outcome_col : str
        Outcome variable column.
    treated_value : Any
        Value in ``group_col`` that identifies the treated group.
    pre_period, post_period : Any
        Values in ``time_col`` defining the pre and post periods.

    Returns
    -------
    float
        Estimated causal effect using difference-in-differences.
    """

    treated_pre = df[(df[group_col] == treated_value) & (df[time_col] == pre_period)][
        outcome_col
    ].mean()
    treated_post = df[(df[group_col] == treated_value) & (df[time_col] == post_period)][
        outcome_col
    ].mean()
    control_pre = df[(df[group_col] != treated_value) & (df[time_col] == pre_period)][
        outcome_col
    ].mean()
    control_post = df[(df[group_col] != treated_value) & (df[time_col] == post_period)][
        outcome_col
    ].mean()
    return float((treated_post - treated_pre) - (control_post - control_pre))


def demographic_parity(y_pred: Iterable[int], sensitive: Iterable[Any]) -> Dict[str, Any]:
    """Demographic parity difference across sensitive groups.

    Returns a dict with per-group positive prediction rates and the
    maximum absolute difference between groups.
    """
    y_pred_arr = np.asarray(list(y_pred))
    sensitive_arr = np.asarray(list(sensitive))
    groups = np.unique(sensitive_arr)
    positive_rates = {g: float(y_pred_arr[sensitive_arr == g].mean()) for g in groups}
    diff = max(positive_rates.values()) - min(positive_rates.values())
    return {"positive_rates": positive_rates, "parity_diff": float(diff)}


def equalized_odds(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    sensitive: Iterable[Any],
) -> Dict[str, Any]:
    """Equalized odds differences across groups.

    Computes true positive rate (TPR) and false positive rate (FPR) for
    each sensitive group and returns the maximum difference between
    groups for both metrics.
    """
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    sensitive_arr = np.asarray(list(sensitive))
    groups = np.unique(sensitive_arr)
    metrics: Dict[Any, Dict[str, float]] = {}
    tprs = []
    fprs = []
    for g in groups:
        mask = sensitive_arr == g
        tp = np.sum((y_pred_arr == 1) & (y_true_arr == 1) & mask)
        fn = np.sum((y_pred_arr == 0) & (y_true_arr == 1) & mask)
        fp = np.sum((y_pred_arr == 1) & (y_true_arr == 0) & mask)
        tn = np.sum((y_pred_arr == 0) & (y_true_arr == 0) & mask)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics[g] = {"tpr": float(tpr), "fpr": float(fpr)}
        tprs.append(tpr)
        fprs.append(fpr)
    return {
        "group_metrics": metrics,
        "tpr_diff": float(max(tprs) - min(tprs)),
        "fpr_diff": float(max(fprs) - min(fprs)),
    }


def check_fairness_thresholds(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    sensitive: Iterable[Any],
    thresholds: Dict[str, float],
) -> Dict[str, float]:
    """Validate fairness metrics against provided thresholds.

    Parameters
    ----------
    thresholds : Dict[str, float]
        Expected keys are ``"demographic_parity"`` and
        ``"equalized_odds"`` representing maximum allowed differences.

    Returns
    -------
    Dict[str, float]
        Calculated fairness metrics.

    Raises
    ------
    ValueError
        If any metric exceeds its threshold.
    """
    dp = demographic_parity(y_pred, sensitive)["parity_diff"]
    eo = equalized_odds(y_true, y_pred, sensitive)
    max_eo = max(abs(eo["tpr_diff"]), abs(eo["fpr_diff"]))
    dp_thr = thresholds.get("demographic_parity", 0.1)
    eo_thr = thresholds.get("equalized_odds", 0.1)
    if abs(dp) > dp_thr or max_eo > eo_thr:
        raise ValueError("Fairness thresholds exceeded")
    return {
        "demographic_parity": dp,
        "equalized_odds_tpr_diff": eo["tpr_diff"],
        "equalized_odds_fpr_diff": eo["fpr_diff"],
    }
