"""trend_filter v1
Hypothesis: Markets move in persistent regimes; being long only in uptrends (fast MA above slow MA) avoids most of the high-volatility down regimes.
"""
import numpy as np
import pandas as pd

PARAMS = {"fast": 20, "slow": 150}
PARAM_GRID = {"fast": [10, 20, 50], "slow": [100, 150, 200]}

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    close = df["close"]
    fast = close.rolling(params["fast"]).mean()
    slow = close.rolling(params["slow"]).mean()
    signal = (fast > slow).astype(float)          # long in uptrends, flat otherwise
    return signal.where(slow.notna(), 0.0)
