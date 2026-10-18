"""momentum_z v1
Hypothesis: Medium-term momentum persists due to investor under-reaction.
"""
import numpy as np
import pandas as pd

PARAMS = {"lookback": 20}

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    mom = df["close"].pct_change(params["lookback"])
    z = (mom - mom.mean()) / mom.std()   # BUG: full-sample mean/std uses future data
    return np.sign(z).clip(-1, 1)
