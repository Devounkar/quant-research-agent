"""Scripted 'LLM' for offline demos and tests (no API key needed).

It replays a fixed research session that shows the full loop, including the agent writing a
strategy with a lookahead bug, the check catching it, and the agent fixing it.
"""
from __future__ import annotations

import json

from .llm import BaseLLM

_BUGGY = '''import numpy as np
import pandas as pd

PARAMS = {"lookback": 20}

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    mom = df["close"].pct_change(params["lookback"])
    z = (mom - mom.mean()) / mom.std()   # BUG: full-sample mean/std uses future data
    return np.sign(z).clip(-1, 1)
'''

_TREND = '''import numpy as np
import pandas as pd

PARAMS = {"fast": 20, "slow": 150}
PARAM_GRID = {"fast": [10, 20, 50], "slow": [100, 150, 200]}

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    close = df["close"]
    fast = close.rolling(params["fast"]).mean()
    slow = close.rolling(params["slow"]).mean()
    signal = (fast > slow).astype(float)          # long in uptrends, flat otherwise
    return signal.where(slow.notna(), 0.0)
'''

_ANALYSIS = '''ret = df["close"].pct_change().dropna()
for lag in (1, 5, 20):
    print(f"autocorr lag {lag}: {ret.autocorr(lag):.3f}")
ma = df["close"].rolling(200).mean()
above = df["close"] > ma
fwd = ret.shift(-1)  # fine in exploratory analysis, never in a strategy
print("mean next-day return above 200d MA:", round(fwd[above].mean(), 5))
print("mean next-day return below 200d MA:", round(fwd[~above].mean(), 5))
print("vol above / below:", round(fwd[above].std(), 4), round(fwd[~above].std(), 4))
'''


def _reply(thought: str, tool: str, args: dict | None = None, code: str | None = None) -> str:
    s = f"THOUGHT: {thought}\nACTION: {tool}\nARGS: {json.dumps(args or {})}\n"
    if code:
        s += f"CODE:\n```python\n{code}```\n"
    return s


SCRIPT = [
    _reply("First I need to understand the data.", "data_info"),
    _reply("Check whether returns trend or mean-revert, and how they differ above/below the 200-day average.",
           "run_analysis", code=_ANALYSIS),
    _reply("Try a z-scored momentum signal.", "write_strategy",
           {"name": "momentum_z", "hypothesis": "Medium-term momentum persists due to investor under-reaction."}, _BUGGY),
    _reply("Before trusting any backtest, verify there is no lookahead.", "check_lookahead", {"name": "momentum_z"}),
    _reply("Lookahead failed: the z-score used the full-sample mean/std. The analysis also showed returns are better "
           "and calmer above the long moving average, so test a simple causal trend filter instead.",
           "write_strategy",
           {"name": "trend_filter", "hypothesis": "Markets move in persistent regimes; being long only in uptrends "
                                                  "(fast MA above slow MA) avoids most of the high-volatility down regimes."},
           _TREND),
    _reply("Quick in-sample backtest for intuition.", "backtest", {"name": "trend_filter"}),
    _reply("Confirm it is causal.", "check_lookahead", {"name": "trend_filter"}),
    _reply("Now the real test: walk-forward with parameters chosen on past data only.", "walk_forward", {"name": "trend_filter"}),
    _reply("Get an independent critique before submitting.", "request_critique", {"name": "trend_filter"}),
    _reply("Try to submit; the gates will reject it if the evidence is too weak.", "submit",
           {"name": "trend_filter", "summary": "Long-only trend filter; reduces drawdowns by sitting out down regimes."}),
    _reply("Submission was rejected by the gates, so I report an honest null result.", "finish",
           {"summary": "Trend filter was causal and positive out of sample but did not clear every gate."}),
]


class MockResearcher(BaseLLM):
    def __init__(self, model="mock", **kw):
        super().__init__(model)
        self.i = 0

    def _chat(self, system, messages):
        reply = SCRIPT[min(self.i, len(SCRIPT) - 1)]
        self.i += 1
        return reply


class MockCritic(BaseLLM):
    def _chat(self, system, messages):
        return json.dumps({
            "verdict": "pass",
            "issues": ["(mock critic) Long-only filter can lag sharp reversals; parameters chosen from a small grid."],
            "suggestions": ["Test sensitivity to transaction costs and to the slow window."],
        })
