"""Long-only trend filter: long when the fast moving average is above the slow one."""
import pandas as pd

PARAMS = {"fast": 20, "slow": 150}
PARAM_GRID = {"fast": [10, 20, 50], "slow": [100, 150, 200]}


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    close = df["close"]
    fast = close.rolling(params["fast"]).mean()
    slow = close.rolling(params["slow"]).mean()
    return (fast > slow).astype(float).where(slow.notna(), 0.0)
