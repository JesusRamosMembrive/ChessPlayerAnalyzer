import os, sys, importlib.util, pathlib
import pandas as pd

# Import the module directly from its file path to avoid importing the entire
# application package, which attempts to connect to external services during
# import time.
module_path = pathlib.Path(__file__).with_name('backend') / 'app' / 'analysis' / 'longitudinal.py'
spec = importlib.util.spec_from_file_location('longitudinal', module_path)
longitudinal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(longitudinal)  # type: ignore[attr-defined]

aggregate_roi = longitudinal.aggregate_roi


def test_empty_dataframe_returns_sd_key():
    df = pd.DataFrame()
    result = aggregate_roi(df)
    assert 'roi_sd' in result
    assert result['roi_sd'] == 0.0
    assert 'roi_std' not in result
