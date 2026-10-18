#!/usr/bin/env python
"""Run the same validation pipeline on a strategy file by hand (no LLM involved).

    python evaluate.py examples/strategies/ma_trend.py
    python evaluate.py my_strategy.py --data-source yfinance --ticker ^NSEI
"""
import argparse
import json
from pathlib import Path

from quant_agent.config import load_config
from quant_agent.data import load_data, split_holdout
from quant_agent.sandbox.runner import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("strategy")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--data-source", choices=["synthetic", "csv", "yfinance"])
    p.add_argument("--ticker")
    p.add_argument("--holdout", action="store_true", help="also evaluate on the holdout (use sparingly!)")
    a = p.parse_args()
    ov = {"data": {}}
    if a.data_source: ov["data"]["source"] = a.data_source
    if a.ticker: ov["data"]["ticker"] = a.ticker
    cfg = load_config(a.config, ov)
    full = load_data(cfg["data"])
    research, _ = split_holdout(full, cfg["data"]["holdout_fraction"])
    code = Path(a.strategy).read_text()
    bt = cfg["backtest"]

    for mode in ("backtest", "lookahead", "walk_forward"):
        res = evaluate(mode, research, code, bt)
        res.pop("series_path", None)
        print(f"\n===== {mode} =====")
        if mode == "walk_forward":
            for f in res["folds"]:
                print(f"  fold {f['fold']}: {f['test_start']}..{f['test_end']} params={f['params']} OOS_SR={f['oos_sharpe']}")
            print("  combined OOS:", json.dumps(res["combined_oos"]))
            print("  buy & hold OOS sharpe:", res["buy_hold_oos"].get("sharpe"))
        else:
            print(json.dumps(res, indent=1, default=str)[:2500])
    if a.holdout:
        res = evaluate("holdout", full, code, bt, research_end=str(research.index[-1].date()))
        print("\n===== holdout =====\n", json.dumps(res["metrics"], indent=1))


if __name__ == "__main__":
    main()
