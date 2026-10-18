"""Writes a human-readable markdown report (and equity-curve chart) for a finished run."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .memory import ResearchLog


def _equity_chart(run_dir: Path, wf_path: str | None, holdout_path: str | None, full_path: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    parts = [p for p in (wf_path, holdout_path) if p and Path(p).exists()]
    if not parts:
        return None
    strat = pd.concat([pd.read_csv(p, index_col=0, parse_dates=True)["returns"] for p in parts])
    full = pd.read_pickle(full_path)
    bh = full["close"].pct_change().reindex(strat.index).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot((1 + strat).cumprod(), label="strategy (out-of-sample only)")
    ax.plot((1 + bh).cumprod(), label="buy & hold", alpha=0.7)
    if holdout_path and Path(holdout_path).exists():
        h_start = pd.read_csv(holdout_path, index_col=0, parse_dates=True).index[0]
        ax.axvspan(h_start, strat.index[-1], color="orange", alpha=0.12, label="hidden holdout")
    ax.set_title("Out-of-sample equity: walk-forward folds + holdout")
    ax.set_ylabel("growth of 1")
    ax.legend()
    ax.grid(alpha=0.3)
    out = run_dir / "equity_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out.name


def write_report(run_dir: Path, cfg: dict, log: ResearchLog, final: dict, full_path: Path, research_end: str) -> Path:
    lines = ["# Quant Research Agent - Run Report", ""]
    lines += [f"- **LLM:** {cfg['llm']['provider']} / {cfg['llm']['model']}",
              f"- **Data:** {cfg['data']['source']} ({cfg['data'].get('ticker')}), research data up to {research_end}, "
              f"last {cfg['data']['holdout_fraction']:.0%} hidden as holdout",
              f"- **Steps used:** {final.get('steps_used')}, **LLM calls:** {final.get('llm_calls')}, "
              f"**time:** {final.get('elapsed_sec')}s", ""]

    if final.get("submitted"):
        h = final["holdout"]
        m, bh = h["metrics"], h["buy_hold"]
        lines += [f"## Result: submitted `{final['strategy']}`", "", f"**Agent summary:** {final.get('summary', '')}", "",
                  "### One-time holdout evaluation", "",
                  "| metric | strategy | buy & hold |", "|---|---|---|"]
        for k in ["sharpe", "cagr", "ann_vol", "max_drawdown", "total_return"]:
            lines.append(f"| {k} | {m.get(k)} | {bh.get(k)} |")
        lines += ["", f"Trades: {m.get('n_trades')}, exposure: {m.get('exposure')}, params used: `{h.get('params_used')}`",
                  f"Deflated Sharpe at submission: {final.get('dsr')}", ""]
        trial = log.get(final["name"])
        chart = _equity_chart(run_dir, trial.walk_forward.get("series_path") if trial and trial.walk_forward else None,
                              h.get("series_path"), full_path)
        if chart:
            lines += [f"![equity curve]({chart})", ""]
        if trial:
            lines += ["### Final strategy code", "", "```python", Path(trial.path).read_text().strip(), "```", ""]
    else:
        lines += ["## Result: no strategy submitted", "", f"**Agent summary:** {final.get('summary', '')}", ""]

    lines += ["## All strategy versions tried", "", "```", log.summary_table(), "```", ""]
    for t in log.trials.values():
        if t.critique:
            lines += [f"**Critic on {t.label}: {t.critique['verdict']}**", ""]
            lines += [f"- {i}" for i in t.critique.get("issues", [])] + [""]
    lines += ["---", "Full step-by-step reasoning: `transcript.md`. Strategy files: `strategies/`."]
    out = run_dir / "report.md"
    out.write_text("\n".join(lines))
    return out
