"""Vectorised daily backtest engine.

Convention (important for avoiding lookahead):
  * generate_signals() returns a target position for each bar, decided using data up to
    and including that bar's close.
  * The engine trades at the next bar: position held during bar t = signal at bar t-1.
  * Costs are charged on every change in position (turnover), in basis points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import compute_metrics


@dataclass
class BacktestResult:
    returns: pd.Series
    positions: pd.Series
    metrics: dict
    warnings: list = field(default_factory=list)


def clean_signals(signals, df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    warnings: list[str] = []
    if isinstance(signals, pd.DataFrame):
        if signals.shape[1] != 1:
            raise ValueError("generate_signals must return a pandas Series (one value per row), got a DataFrame")
        signals = signals.iloc[:, 0]
    if not isinstance(signals, pd.Series):
        arr = np.asarray(signals, dtype=float)
        if arr.shape != (len(df),):
            raise ValueError(f"generate_signals returned shape {arr.shape}, expected ({len(df)},)")
        signals = pd.Series(arr, index=df.index)
    if not signals.index.equals(df.index):
        missing = (~df.index.isin(signals.index)).sum()
        if missing:
            warnings.append(f"signal index missing {missing} dates; filled with 0")
        signals = signals.reindex(df.index)
    s = pd.to_numeric(signals, errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    n_nan = int(s.isna().sum())
    if n_nan > 0.5 * len(s):
        warnings.append(f"{n_nan}/{len(s)} signals were NaN (treated as flat)")
    s = s.fillna(0.0)
    if (s.abs() > 1 + 1e-9).any():
        warnings.append("signals outside [-1, 1] were clipped")
        s = s.clip(-1, 1)
    return s, warnings


def run_backtest(df: pd.DataFrame, signals: pd.Series, cost_bps: float = 5.0, periods_per_year: int = 252) -> BacktestResult:
    ret = df["close"].pct_change().fillna(0.0)
    pos = signals.shift(1).fillna(0.0)
    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])
    net = pos * ret - turnover * cost_bps / 1e4
    return BacktestResult(returns=net, positions=pos, metrics=compute_metrics(net, pos, periods_per_year))


def buy_and_hold(df: pd.DataFrame, periods_per_year: int = 252) -> dict:
    ret = df["close"].pct_change().fillna(0.0)
    return compute_metrics(ret, None, periods_per_year)
