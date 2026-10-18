import numpy as np
import pandas as pd
import pytest

from quant_agent.backtest import clean_signals, compute_metrics, run_backtest


def _df(prices):
    idx = pd.bdate_range("2020-01-01", periods=len(prices))
    return pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1.0}, index=idx)


def test_positions_are_shifted_one_bar():
    df = _df([100, 110, 121, 133.1])
    sig = pd.Series([1.0, 1.0, 1.0, 1.0], index=df.index)
    res = run_backtest(df, sig, cost_bps=0)
    # day 0 has no position; days 1..3 earn +10% each
    assert res.returns.iloc[0] == 0
    assert np.allclose(res.returns.iloc[1:], 0.10)


def test_signal_on_last_bar_earns_nothing():
    df = _df([100, 110, 121])
    sig = pd.Series([0.0, 0.0, 1.0], index=df.index)  # decided at the final close -> no trade yet
    assert run_backtest(df, sig, cost_bps=0).returns.sum() == 0


def test_costs_charged_on_turnover():
    df = _df([100.0] * 5)
    sig = pd.Series([1, 0, 1, 0, 0], index=df.index, dtype=float)
    res = run_backtest(df, sig, cost_bps=10)
    # positions: 0,1,0,1,0 -> 4 changes of size 1 -> 4 * 10bps
    assert res.returns.sum() == pytest.approx(-4 * 0.001)


def test_clean_signals_clips_and_fills():
    df = _df([1, 2, 3, 4])
    s, warns = clean_signals(pd.Series([np.nan, 2.0, -3.0, 0.5], index=df.index), df)
    assert list(s) == [0.0, 1.0, -1.0, 0.5]
    assert any("clipped" in w for w in warns)


def test_clean_signals_rejects_wrong_length():
    df = _df([1, 2, 3])
    with pytest.raises(ValueError):
        clean_signals(np.array([1.0, 0.0]), df)


def test_metrics_basic():
    idx = pd.bdate_range("2020-01-01", periods=252)
    r = pd.Series(0.001, index=idx)
    r.iloc[::2] = -0.0005
    m = compute_metrics(r, pd.Series(1.0, index=idx))
    assert m["sharpe"] > 0 and m["max_drawdown"] <= 0 and m["exposure"] == 1.0
