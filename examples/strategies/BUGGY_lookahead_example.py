"""DELIBERATELY BROKEN: shows a classic lookahead bug that check_lookahead catches.
The z-score uses the mean/std of the WHOLE series, including future data."""
import numpy as np
import pandas as pd

PARAMS = {"lookback": 20}


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    mom = df["close"].pct_change(params["lookback"])
    z = (mom - mom.mean()) / mom.std()   # <-- future information leaks in here
    return np.sign(z).fillna(0)
