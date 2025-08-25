import os
import pathlib
import sys

import pandas as pd
import pytest

analysis_dir = pathlib.Path(__file__).resolve().parents[1] / "app" / "analysis"
sys.path.insert(0, str(analysis_dir))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("DB_MAX_RETRIES", "1")

from causal_fairness import demographic_parity, equalized_odds
from ml_classifier import MLSuspicionClassifier


def test_fairness_metrics_basic():
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    sensitive = ["A", "A", "A", "A", "B", "B", "B", "B"]

    dp = demographic_parity(y_pred, sensitive)
    eo = equalized_odds(y_true, y_pred, sensitive)

    assert dp["parity_diff"] >= 0
    assert "tpr_diff" in eo and "fpr_diff" in eo


def test_training_pipeline_checks_fairness():
    X = pd.DataFrame({
        "f1": [0, 1, 0, 1, 0, 1, 0, 1],
        "f2": [1, 0, 1, 0, 1, 0, 1, 0],
    })
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    sensitive = [0, 0, 0, 0, 1, 1, 1, 1]

    clf = MLSuspicionClassifier()
    # Should pass with relaxed thresholds
    stats = clf.train(
        X,
        y,
        model_type="gradient_boosting",
        cv=2,
        sensitive_features=sensitive,
        fairness_thresholds={"demographic_parity": 1.0, "equalized_odds": 1.0},
    )
    assert "demographic_parity" in stats
    from causal_fairness import check_fairness_thresholds

    # Artificially biased predictions to trigger failure
    biased_preds = [1, 1, 1, 1, 0, 0, 0, 0]

    with pytest.raises(ValueError):
        check_fairness_thresholds(
            y,
            biased_preds,
            sensitive,
            {"demographic_parity": 0.0},
        )
