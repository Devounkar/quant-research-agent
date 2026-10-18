import numpy as np
import pandas as pd

from quant_agent.backtest import deflated_sharpe, lookahead_check, param_combos, walk_forward
from quant_agent.data import generate_synthetic

DF = generate_synthetic(n_days=1500)


def causal(df, p):
    return (df["close"] > df["close"].rolling(p["n"]).mean()).astype(float)


def full_sample_zscore(df, p):
    m = df["close"].pct_change(20)
    return np.sign((m - m.mean()) / m.std()).fillna(0)


def future_peek(df, p):
    return np.sign(df["close"].shift(-1) - df["close"]).fillna(0)


def centered_window(df, p):
    return (df["close"] > df["close"].rolling(21, center=True).mean()).astype(float)


def test_causal_strategy_passes():
    assert lookahead_check(causal, DF, {"n": 50})["passed"]


def test_lookahead_bugs_are_caught():
    for fn in (full_sample_zscore, future_peek, centered_window):
        assert not lookahead_check(fn, DF, {"n": 50})["passed"], fn.__name__


def test_param_grid_limit():
    assert len(param_combos({"a": 1}, {"a": [1, 2], "b": [3, 4]})) == 4
    try:
        param_combos({}, {"a": list(range(5)), "b": list(range(5))})
        assert False, "should reject >16 combos"
    except ValueError:
        pass


def test_walk_forward_shapes():
    wf = walk_forward(causal, DF, {"n": 50}, {"n": [50, 100]}, n_folds=4)
    assert len(wf["folds"]) == 4
    # OOS returns cover exactly the data after the first chunk, with no overlap
    assert wf["returns"].index.is_unique
    assert wf["returns"].index[0] == DF.index[len(DF) // 5]


def test_dsr_falls_with_more_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 1500))
    few = deflated_sharpe(r, [0.05, 0.0])["dsr"]
    many = deflated_sharpe(r, list(rng.normal(0, 0.03, 50)))["dsr"]
    assert many < few
