"""Market data loading: synthetic (offline demo), CSV, or Yahoo Finance."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close", "volume"]


def generate_synthetic(n_days: int = 4000, seed: int = 11, start: str = "2010-01-01") -> pd.DataFrame:
    """Regime-switching random walk (persistent bull/bear regimes). Useful for testing the
    pipeline offline. It is NOT real market data - never draw trading conclusions from it."""
    rng = np.random.default_rng(seed)
    regimes = {0: (0.0007, 0.009), 1: (-0.0010, 0.017)}  # (daily drift, daily vol)
    p_switch = {0: 0.005, 1: 0.008}
    state, rets = 0, []
    for _ in range(n_days):
        if rng.random() < p_switch[state]:
            state = 1 - state
        mu, sigma = regimes[state]
        rets.append(rng.normal(mu, sigma))
    close = 5000 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range(start=start, periods=n_days)
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n_days))
    spread = np.abs(rng.normal(0, 0.006, n_days))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(18, 0.3, n_days)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    if "adj_close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj_close"]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    for c in ("open", "high", "low"):
        if c not in df.columns:
            df[c] = df["close"]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"data is missing columns: {missing}")
    df = df[REQUIRED].apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.strip().lower() in ("date", "datetime", "timestamp")), df.columns[0])
    df = df.set_index(date_col)
    return _standardise(df)


def load_yfinance(ticker: str, start: str, end: str | None, cache_dir: str | Path = "data") -> pd.DataFrame:
    cache = Path(cache_dir) / f"{ticker.replace('^', '').replace('.', '_')}.csv"
    if cache.exists():
        return load_csv(cache)
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("pip install yfinance  (or use data.source: csv / synthetic)") from e
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"no data downloaded for {ticker}")
    df = _standardise(raw)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index_label="date")
    return df


def load_data(cfg: dict) -> pd.DataFrame:
    src = cfg.get("source", "synthetic")
    if src == "synthetic":
        df = generate_synthetic(seed=cfg.get("seed", 11))
    elif src == "csv":
        df = load_csv(cfg["csv_path"])
    elif src == "yfinance":
        df = load_yfinance(cfg["ticker"], cfg.get("start", "2010-01-01"), cfg.get("end"))
    else:
        raise ValueError(f"unknown data source: {src}")
    if cfg.get("start"):
        df = df.loc[cfg["start"]:]
    if cfg.get("end"):
        df = df.loc[: cfg["end"]]
    if len(df) < 1000:
        raise ValueError(f"only {len(df)} rows of data; need at least ~1000 daily bars (4 years)")
    return df


def split_holdout(df: pd.DataFrame, holdout_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(df) * (1 - holdout_fraction))
    return df.iloc[:cut], df.iloc[cut:]


def describe(df: pd.DataFrame, name: str) -> str:
    ret = df["close"].pct_change().dropna()
    return (
        f"Instrument: {name}\n"
        f"Research period: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} daily bars)\n"
        f"Columns: {list(df.columns)} (index: DatetimeIndex)\n"
        f"Close: first {df['close'].iloc[0]:.2f}, last {df['close'].iloc[-1]:.2f}, "
        f"min {df['close'].min():.2f}, max {df['close'].max():.2f}\n"
        f"Daily return: mean {ret.mean():.5f}, std {ret.std():.5f}, "
        f"skew {ret.skew():.2f}, excess kurtosis {ret.kurt():.2f}\n"
        f"Lag-1 autocorrelation of daily returns: {ret.autocorr(1):.3f}"
    )
