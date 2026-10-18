"""Short-term mean reversion: buy oversold dips (RSI) only while the long-term trend is up."""
import pandas as pd

PARAMS = {"rsi_window": 3, "entry": 15, "exit": 60, "trend": 200}
PARAM_GRID = {"entry": [10, 20], "exit": [50, 70]}


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    close = df["close"]
    rsi = _rsi(close, params["rsi_window"])
    uptrend = close > close.rolling(params["trend"]).mean()
    pos, state = [], 0.0
    for r, up in zip(rsi.to_numpy(), uptrend.to_numpy()):
        if state == 0 and up and r < params["entry"]:
            state = 1.0
        elif state == 1 and (r > params["exit"] or not up):
            state = 0.0
        pos.append(state)
    return pd.Series(pos, index=df.index)
