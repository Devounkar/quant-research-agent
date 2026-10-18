"""Configuration loading with sensible defaults."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "llm": {
        "provider": "anthropic",      # anthropic | openai | ollama | mock
        "model": "claude-sonnet-5-5",
        "critic_model": None,          # defaults to `model`
        "temperature": 0.4,
        "max_tokens": 3000,
        "base_url": None,              # for OpenAI-compatible APIs / custom Ollama host
    },
    "data": {
        "source": "synthetic",         # synthetic | csv | yfinance
        "ticker": "^NSEI",
        "csv_path": "data/nifty.csv",
        "start": None,
        "end": None,
        "holdout_fraction": 0.2,
        "seed": 11,
    },
    "backtest": {"cost_bps": 5.0, "periods_per_year": 252, "n_folds": 5},
    "agent": {"max_steps": 40, "max_strategies": 12, "context_turns": 10, "sandbox_timeout": 60},
    "gates": {"min_oos_sharpe": 0.5, "min_dsr": 0.9, "max_drawdown": 0.35, "min_trades": 10},
    "output_dir": "runs",
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path and Path(path).exists():
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    return _merge(cfg, overrides or {})
