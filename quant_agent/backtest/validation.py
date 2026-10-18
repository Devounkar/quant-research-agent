"""Robustness checks: lookahead detection, walk-forward testing, deflated Sharpe ratio."""
from __future__ import annotations

import itertools
import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .engine import clean_signals, run_backtest
from .metrics import compute_metrics

MAX_COMBOS = 16
EULER_GAMMA = 0.5772156649015329

SignalFn = Callable[[pd.DataFrame, dict], pd.Series]


# ---------------------------------------------------------------- lookahead
def lookahead_check(fn: SignalFn, df: pd.DataFrame, params: dict, n_checks: int = 6,
                    min_rows: int = 60, tol: float = 1e-6) -> dict:
    """Re-run the strategy on truncated history. A causal strategy must give the same
    signals for every past date no matter how much future data exists."""
    full, _ = clean_signals(fn(df.copy(), dict(params)), df)
    n = len(df)
    start = min(max(min_rows, n // 5), n - 2)
    cuts = sorted(set(np.linspace(start, n - 2, n_checks).astype(int).tolist()))
    failures = []
    for c in cuts:
        part_df = df.iloc[: c + 1]
        part, _ = clean_signals(fn(part_df.copy(), dict(params)), part_df)
        diff = (full.iloc[: c + 1] - part).abs()
        bad = diff[diff > tol]
        if len(bad):
            failures.append({
                "cut_date": str(df.index[c].date()),
                "n_mismatched": int(len(bad)),
                "first_mismatch": str(bad.index[0].date()),
                "max_abs_diff": round(float(bad.max()), 6),
            })
    return {"passed": not failures, "n_checks": len(cuts), "failures": failures[:5]}


# ---------------------------------------------------------------- walk-forward
def param_combos(params: dict, grid: dict | None) -> list[dict]:
    if not grid:
        return [dict(params)]
    keys = list(grid.keys())
    combos = [{**params, **dict(zip(keys, vals))} for vals in itertools.product(*[grid[k] for k in keys])]
    if len(combos) > MAX_COMBOS:
        raise ValueError(f"PARAM_GRID has {len(combos)} combinations; the limit is {MAX_COMBOS}")
    return combos


def _select_params(fn, train, combos, cost_bps, ppy):
    if len(combos) == 1:
        return combos[0], None
    best, best_sr = combos[0], -np.inf
    for c in combos:
        sig, _ = clean_signals(fn(train.copy(), dict(c)), train)
        sr = run_backtest(train, sig, cost_bps, ppy).metrics.get("sharpe") or -np.inf
        if sr > best_sr:
            best, best_sr = c, sr
    return best, best_sr


def walk_forward(fn: SignalFn, df: pd.DataFrame, params: dict, grid: dict | None,
                 n_folds: int = 5, cost_bps: float = 5.0, ppy: int = 252) -> dict:
    """Expanding-window walk-forward. For each fold, parameters are chosen ONLY on past data
    (if a PARAM_GRID is given), then the strategy is evaluated on the next unseen window."""
    combos = param_combos(params, grid)
    n = len(df)
    chunk = n // (n_folds + 1)
    if chunk < 60:
        raise ValueError(f"not enough data for {n_folds} folds ({n} rows)")

    folds, oos_r, oos_p = [], [], []
    for i in range(1, n_folds + 1):
        tr_end = i * chunk
        te_end = n if i == n_folds else (i + 1) * chunk
        train = df.iloc[:tr_end]
        best, in_sample_sr = _select_params(fn, train, combos, cost_bps, ppy)
        prefix = df.iloc[:te_end]  # full history up to test end, so indicators are warmed up
        sig, _ = clean_signals(fn(prefix.copy(), dict(best)), prefix)
        res = run_backtest(prefix, sig, cost_bps, ppy)
        test_idx = df.index[tr_end:te_end]
        r, p = res.returns.loc[test_idx], res.positions.loc[test_idx]
        m = compute_metrics(r, p, ppy)
        bh = compute_metrics(df["close"].pct_change().loc[test_idx], None, ppy)
        folds.append({
            "fold": i,
            "test_start": str(test_idx[0].date()),
            "test_end": str(test_idx[-1].date()),
            "params": {k: best[k] for k in (grid or {})} if grid else "fixed",
            "in_sample_sharpe": round(in_sample_sr, 3) if in_sample_sr is not None else None,
            "oos_sharpe": m.get("sharpe"),
            "oos_return": m.get("total_return"),
            "oos_max_dd": m.get("max_drawdown"),
            "buy_hold_sharpe": bh.get("sharpe"),
        })
        oos_r.append(r)
        oos_p.append(p)

    oos_returns = pd.concat(oos_r)
    oos_positions = pd.concat(oos_p)
    combined = compute_metrics(oos_returns, oos_positions, ppy)
    bh_all = compute_metrics(df["close"].pct_change().loc[oos_returns.index], None, ppy)
    fold_srs = [f["oos_sharpe"] for f in folds if f["oos_sharpe"] is not None]
    return {
        "folds": folds,
        "combined_oos": combined,
        "buy_hold_oos": bh_all,
        "n_positive_folds": int(sum(s > 0 for s in fold_srs)),
        "fold_sharpe_std": round(float(np.std(fold_srs)), 3) if fold_srs else None,
        "returns": oos_returns,
        "positions": oos_positions,
    }


# ---------------------------------------------------------------- deflated Sharpe
def probabilistic_sharpe(sr: float, sr_benchmark: float, n: int, skew: float, kurt: float) -> float:
    """PSR (Bailey & Lopez de Prado). All Sharpe ratios are per-period (not annualised);
    kurt is non-excess kurtosis."""
    if n < 3:
        return 0.0
    var_term = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    var_term = max(var_term, 1e-12)
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(var_term)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """Expected maximum Sharpe among n_trials unskilled strategies (False Strategy Theorem)."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    return math.sqrt(var_sr) * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe(returns: pd.Series, trial_sharpes: list[float]) -> dict:
    """Probability that the true Sharpe is > 0 after accounting for how many strategies were tried."""
    r = pd.Series(returns).dropna()
    t = len(r)
    if t < 30 or r.std(ddof=1) == 0:
        return {"dsr": 0.0, "n_trials": len(trial_sharpes), "note": "not enough data"}
    sr = r.mean() / r.std(ddof=1)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    n_trials = max(1, len(trial_sharpes))
    var_sr = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) >= 2 else 0.0
    sr0 = expected_max_sharpe(var_sr, n_trials)
    return {
        "dsr": round(probabilistic_sharpe(sr, sr0, t, skew, kurt), 4),
        "psr_vs_zero": round(probabilistic_sharpe(sr, 0.0, t, skew, kurt), 4),
        "n_trials": n_trials,
        "sharpe_hurdle_annual": round(float(sr0) * math.sqrt(252), 3),
    }
