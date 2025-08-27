import os
import pathlib
import sys

import numpy as np
import pandas as pd

analysis_dir = pathlib.Path(__file__).resolve().parents[1] / "app" / "analysis"
sys.path.insert(0, str(analysis_dir))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("DB_MAX_RETRIES", "1")

from performance_model import fit_performance_model, predict_performance


def test_garch_prediction_conf_int():
    np.random.seed(0)
    series = pd.Series(np.random.normal(scale=1.5, size=200))
    model = fit_performance_model(series, model="garch")
    preds, lower, upper = predict_performance(model, steps=5, return_conf_int=True)
    assert preds.shape == (5,)
    assert lower.shape == (5,)
    assert upper.shape == (5,)
    assert np.all(upper > lower)


def test_auto_model_selection():
    np.random.seed(1)
    high_vol = pd.Series(np.random.normal(scale=5.0, size=100))
    low_vol = pd.Series(np.linspace(1, 1.1, 100))

    model_high = fit_performance_model(high_vol, model="auto")
    model_low = fit_performance_model(low_vol, model="auto")

    assert model_high["model"] == "garch"
    assert model_low["model"] == "kalman"
