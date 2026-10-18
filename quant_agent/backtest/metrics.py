"""Performance metrics for a series of periodic (daily) strategy returns."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _f(x, nd: int = 4):
    """Convert to a JSON-friendly rounded float (None for NaN/inf)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd)


def compute_metrics(returns: pd.Series, positions: pd.Series | None = None, periods_per_year: int = 252) -> dict:
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 2:
        return {"n_periods": n, "error": "not enough data"}

    equity = (1 + r).cumprod()
    years = n / periods_per_year
    final = equity.iloc[-1]
    cagr = final ** (1 / years) - 1 if final > 0 else -1.0
    std = r.std(ddof=1)
    sharpe = r.mean() / std * math.sqrt(periods_per_year) if std > 0 else 0.0
    downside = r[r < 0].std(ddof=1)
    sortino = r.mean() / downside * math.sqrt(periods_per_year) if downside and downside > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    out = {
        "start": str(r.index[0].date()) if hasattr(r.index[0], "date") else str(r.index[0]),
        "end": str(r.index[-1].date()) if hasattr(r.index[-1], "date") else str(r.index[-1]),
        "n_periods": n,
        "total_return": _f(final - 1),
        "cagr": _f(cagr),
        "ann_vol": _f(std * math.sqrt(periods_per_year)),
        "sharpe": _f(sharpe, 3),
        "sortino": _f(sortino, 3),
        "max_drawdown": _f(max_dd),
        "calmar": _f(cagr / abs(max_dd)) if max_dd < 0 else None,
        "skew": _f(r.skew(), 3),
        "kurtosis": _f(r.kurt() + 3, 3),  # non-excess kurtosis
    }

    if positions is not None:
        pos = pd.Series(positions).reindex(r.index).fillna(0.0)
        changes = pos.diff().abs()
        changes.iloc[0] = abs(pos.iloc[0])
        active = r[pos.abs() > 1e-9]
        out.update({
            "n_trades": int((changes > 1e-9).sum()),
            "turnover_per_year": _f(changes.sum() / years, 2),
            "exposure": _f((pos.abs() > 1e-9).mean(), 3),
            "avg_abs_position": _f(pos.abs().mean(), 3),
            "hit_rate": _f((active > 0).mean(), 3) if len(active) else None,
        })
    return out


def yearly_returns(returns: pd.Series) -> dict:
    r = pd.Series(returns).dropna()
    if r.empty:
        return {}
    by_year = (1 + r).groupby(r.index.year).prod() - 1
    return {int(y): _f(v) for y, v in by_year.items()}
