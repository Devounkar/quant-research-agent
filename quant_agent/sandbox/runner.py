"""Executes strategy / analysis code. Normally run in a separate subprocess by executor.py:

    python -m quant_agent.sandbox.runner <job.json>

The result is printed as a single line prefixed with __RESULT__.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from quant_agent.backtest import (
    clean_signals, compute_metrics, lookahead_check, param_combos, run_backtest,
    walk_forward, yearly_returns,
)
from quant_agent.backtest.engine import buy_and_hold
from quant_agent.sandbox.safety import check_code

MARKER = "__RESULT__"
MAX_STDOUT = 4000


def load_strategy(code: str, filename: str = "<strategy>"):
    errors = check_code(code, require_strategy=True)
    if errors:
        raise ValueError("Code failed safety check: " + "; ".join(errors))
    ns: dict = {}
    exec(compile(code, filename, "exec"), ns)  # noqa: S102 - code was statically checked
    fn = ns["generate_signals"]
    params = dict(ns.get("PARAMS", {}) or {})
    grid = ns.get("PARAM_GRID") or None
    return fn, params, grid


def _save_series(out_dir: str | None, name: str, returns: pd.Series, positions: pd.Series):
    if not out_dir:
        return None
    path = Path(out_dir) / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"returns": returns, "positions": positions}).to_csv(path)
    return str(path)


def evaluate(mode: str, df: pd.DataFrame, code: str, cfg: dict, out_dir: str | None = None,
             research_end: str | None = None, filename: str = "<strategy>") -> dict:
    cost, ppy = cfg.get("cost_bps", 5.0), cfg.get("periods_per_year", 252)

    if mode == "analysis":
        errors = check_code(code, require_strategy=False)
        if errors:
            raise ValueError("Code failed safety check: " + "; ".join(errors))
        buf = io.StringIO()
        ns = {"df": df.copy(), "np": np, "pd": pd}
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<analysis>", "exec"), ns)  # noqa: S102
        out = buf.getvalue()
        if len(out) > MAX_STDOUT:
            out = out[:MAX_STDOUT] + f"\n... [truncated, {len(out)} chars total]"
        return {"ok": True, "stdout": out or "(no output - use print())"}

    fn, params, grid = load_strategy(code, filename)

    if mode == "backtest":
        sig, warns = clean_signals(fn(df.copy(), dict(params)), df)
        res = run_backtest(df, sig, cost, ppy)
        return {
            "ok": True, "params": params, "metrics": res.metrics,
            "buy_hold": buy_and_hold(df, ppy), "yearly": yearly_returns(res.returns),
            "warnings": warns,
            "series_path": _save_series(out_dir, "backtest", res.returns, res.positions),
        }

    if mode == "lookahead":
        return {"ok": True, **lookahead_check(fn, df, params)}

    if mode == "walk_forward":
        wf = walk_forward(fn, df, params, grid, cfg.get("n_folds", 5), cost, ppy)
        path = _save_series(out_dir, "walk_forward", wf.pop("returns"), wf.pop("positions"))
        return {"ok": True, "grid_size": len(param_combos(params, grid)), **wf, "series_path": path}

    if mode == "holdout":
        if research_end is None:
            raise ValueError("holdout mode needs research_end")
        research = df.loc[:research_end]
        combos = param_combos(params, grid)
        best = combos[0]
        if len(combos) > 1:
            best_sr = -np.inf
            for c in combos:
                s, _ = clean_signals(fn(research.copy(), dict(c)), research)
                sr = run_backtest(research, s, cost, ppy).metrics.get("sharpe") or -np.inf
                if sr > best_sr:
                    best, best_sr = c, sr
        sig, warns = clean_signals(fn(df.copy(), dict(best)), df)
        res = run_backtest(df, sig, cost, ppy)
        idx = df.index[df.index > pd.Timestamp(research_end)]
        r, p = res.returns.loc[idx], res.positions.loc[idx]
        return {
            "ok": True, "params_used": best, "metrics": compute_metrics(r, p, ppy),
            "buy_hold": compute_metrics(df["close"].pct_change().loc[idx], None, ppy),
            "yearly": yearly_returns(r), "warnings": warns,
            "series_path": _save_series(out_dir, "holdout", r, p),
        }

    raise ValueError(f"unknown mode: {mode}")


def _limit_resources(cpu_seconds: int):
    try:
        import resource  # Unix only
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
    except Exception:
        pass


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    _limit_resources(int(job.get("timeout", 60)) + 5)
    try:
        df = pd.read_pickle(job["data_path"])
        code = job.get("code") or Path(job["strategy_path"]).read_text()
        result = evaluate(job["mode"], df, code, job.get("cfg", {}), job.get("out_dir"),
                          job.get("research_end"), job.get("strategy_path", "<strategy>"))
    except Exception as e:  # report the error back to the agent
        tb = traceback.format_exc().strip().splitlines()
        result = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": "\n".join(tb[-6:])}
    print(MARKER + json.dumps(result, default=str))


if __name__ == "__main__":
    main()
