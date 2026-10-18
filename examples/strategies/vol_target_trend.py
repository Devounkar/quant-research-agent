"""Trend filter with volatility targeting: size the position so expected volatility is ~constant."""
import numpy as np
import pandas as pd

PARAMS = {"slow": 150, "vol_window": 20, "target_vol": 0.12}
PARAM_GRID = {"slow": [100, 200], "target_vol": [0.10, 0.15]}


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    close = df["close"]
    trend = (close > close.rolling(params["slow"]).mean()).astype(float)
    vol = close.pct_change().rolling(params["vol_window"]).std() * np.sqrt(252)
    size = (params["target_vol"] / vol).clip(upper=1.0)
    return (trend * size).fillna(0.0)
