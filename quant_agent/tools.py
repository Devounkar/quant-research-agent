"""Tools available to the Researcher agent.

Each tool returns a ToolResult whose text becomes the agent's next OBSERVATION. All LLM-written
code runs in a separate sandboxed subprocess (see sandbox/), never in this process.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .backtest.validation import MAX_COMBOS, deflated_sharpe
from .critic import Critic
from .data import describe
from .memory import ResearchLog, Trial
from .parsing import Action
from .sandbox import check_code, run_job

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
MAX_OBS_CHARS = 3500


@dataclass
class ToolResult:
    text: str
    done: bool = False


@dataclass
class ToolSpec:
    name: str
    args: str
    takes_code: bool
    description: str


TOOL_SPECS = [
    ToolSpec("data_info", "{}", False, "Describe the research dataset and buy-and-hold performance."),
    ToolSpec("run_analysis", "{}", True,
             "Run exploratory Python on the research data. `df`, `np`, `pd` are pre-loaded. Use print() to see results."),
    ToolSpec("write_strategy", '{"name": "snake_case_name", "hypothesis": "why this edge should exist"}', True,
             "Save a strategy (CODE = full strategy file). Re-using a name creates a new version."),
    ToolSpec("read_strategy", '{"name": "..."}', False, "Show the latest version's code and hypothesis."),
    ToolSpec("backtest", '{"name": "..."}', False,
             "Full research-period backtest with default PARAMS (in-sample - optimistic; use for debugging/intuition)."),
    ToolSpec("check_lookahead", '{"name": "..."}', False,
             "Detect use of future data by re-running on truncated history. Must pass before submit."),
    ToolSpec("walk_forward", '{"name": "..."}', False,
             "Expanding-window walk-forward test (the real out-of-sample evidence). Reports per-fold results and DSR."),
    ToolSpec("research_log", "{}", False, "Table of every strategy version tried and its results."),
    ToolSpec("request_critique", '{"name": "..."}', False,
             "Send the latest version to the independent Critic agent. Needs lookahead + walk_forward done first."),
    ToolSpec("submit", '{"name": "...", "summary": "what you found and why it should work"}', False,
             "Final submission. Checks all gates, then evaluates ONCE on the hidden holdout. Ends the run."),
    ToolSpec("finish", '{"summary": "honest summary of what you tried and learned"}', False,
             "End the run without submitting (use if no strategy passes the gates)."),
]


def tool_docs() -> str:
    lines = []
    for s in TOOL_SPECS:
        code = " + CODE block" if s.takes_code else ""
        lines.append(f"- {s.name}: ARGS {s.args}{code}\n    {s.description}")
    return "\n".join(lines)


def _fmt(d: dict | None, keys: list[str] | None = None) -> str:
    if not d:
        return "n/a"
    items = d.items() if keys is None else ((k, d.get(k)) for k in keys)
    return ", ".join(f"{k}={v}" for k, v in items if v is not None)


KEY_METRICS = ["sharpe", "cagr", "ann_vol", "max_drawdown", "n_trades", "turnover_per_year", "exposure", "hit_rate"]


class ResearchTools:
    def __init__(self, cfg: dict, run_dir: Path, log: ResearchLog, critic: Critic,
                 research_df: pd.DataFrame, full_df: pd.DataFrame, instrument: str):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.log = log
        self.critic = critic
        self.instrument = instrument
        self.research_end = str(research_df.index[-1].date())
        self.research_desc = describe(research_df, instrument)
        data_dir = self.run_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.research_path = data_dir / "research.pkl"
        self.full_path = data_dir / "full.pkl"
        research_df.to_pickle(self.research_path)
        full_df.to_pickle(self.full_path)  # only the holdout evaluation ever reads this
        self.bt_cfg = dict(cfg["backtest"])
        self.timeout = cfg["agent"]["sandbox_timeout"]
        self.final: dict | None = None
        self.step = 0
        self._handlers = {s.name: getattr(self, f"_t_{s.name}") for s in TOOL_SPECS}

    # ------------------------------------------------------------------ dispatch
    def call(self, action: Action, step: int) -> ToolResult:
        self.step = step
        handler = self._handlers.get(action.tool)
        if handler is None:
            return ToolResult(f"ERROR: unknown tool '{action.tool}'. Available: {', '.join(self._handlers)}")
        try:
            res = handler(action)
        except Exception as e:  # never crash the agent loop on a tool error
            res = ToolResult(f"ERROR in {action.tool}: {type(e).__name__}: {e}")
        if len(res.text) > MAX_OBS_CHARS:
            res.text = res.text[:MAX_OBS_CHARS] + "\n... [observation truncated]"
        return res

    def _job(self, mode: str, trial: Trial | None = None, code: str | None = None, data: Path | None = None,
             research_end: str | None = None) -> dict:
        out_dir = self.run_dir / "results" / (f"{trial.name}_v{trial.version}" if trial else "analysis")
        return run_job(mode, self.run_dir, data or self.research_path, self.timeout, code=code,
                       strategy_path=Path(trial.path) if trial else None, cfg=self.bt_cfg,
                       research_end=research_end, out_dir=out_dir)

    def _trial(self, action: Action) -> Trial:
        name = action.args.get("name")
        if not name:
            raise ValueError("missing required arg 'name'")
        trial = self.log.get(name)
        if trial is None:
            known = ", ".join(self.log.latest) or "none"
            raise ValueError(f"no strategy named '{name}' (known: {known})")
        return trial

    @staticmethod
    def _error(res: dict) -> str:
        tb = f"\n{res['traceback']}" if res.get("traceback") else ""
        return f"FAILED: {res.get('error')}{tb}"

    def _current_dsr(self, trial: Trial) -> dict | None:
        wf = trial.walk_forward
        if not wf or not wf.get("series_path"):
            return None
        returns = pd.read_csv(wf["series_path"], index_col=0, parse_dates=True)["returns"]
        return deflated_sharpe(returns, self.log.trial_sharpes())

    # ------------------------------------------------------------------ tools
    def _t_data_info(self, action: Action) -> ToolResult:
        holdout = self.cfg["data"]["holdout_fraction"]
        return ToolResult(
            f"{self.research_desc}\n"
            f"Hidden holdout: the {holdout:.0%} of data after {self.research_end} is reserved and not available.\n"
            f"Costs: {self.bt_cfg['cost_bps']} bps per unit position change. Walk-forward folds: {self.bt_cfg['n_folds']}."
        )

    def _t_run_analysis(self, action: Action) -> ToolResult:
        if not action.code:
            return ToolResult("ERROR: run_analysis needs a CODE block.")
        errors = check_code(action.code, require_strategy=False)
        if errors:
            return ToolResult("Code rejected by safety check:\n- " + "\n- ".join(errors))
        res = self._job("analysis", code=action.code)
        return ToolResult(res["stdout"] if res.get("ok") else self._error(res))

    def _t_write_strategy(self, action: Action) -> ToolResult:
        name, hyp = action.args.get("name", ""), action.args.get("hypothesis", "").strip()
        if not NAME_RE.match(name):
            return ToolResult("ERROR: name must be snake_case (lowercase letters, digits, underscores), 2-41 chars.")
        if not hyp:
            return ToolResult("ERROR: a 'hypothesis' explaining why the edge should exist is required.")
        if not action.code:
            return ToolResult("ERROR: write_strategy needs a CODE block with the full strategy file.")
        errors = check_code(action.code, require_strategy=True)
        if errors:
            return ToolResult("Strategy rejected by safety/format check:\n- " + "\n- ".join(errors))
        if self.log.n_strategies() >= self.cfg["agent"]["max_strategies"] and \
                self.log.hash_code(action.code) not in self.log.trials:
            return ToolResult("ERROR: strategy budget exhausted. Submit an existing strategy or finish.")
        h = self.log.hash_code(action.code)
        version = 1 + sum(t.name == name for t in self.log.trials.values())
        path = self.run_dir / "strategies" / f"{name}_v{version}.py"
        trial, is_new = self.log.add_version(name, action.code, hyp, str(path), self.step)
        if not is_new:
            return ToolResult(f"This exact code already exists as {trial.label}; nothing new was saved.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'"""{trial.label}\nHypothesis: {hyp}\n"""\n' + action.code + "\n")
        used = self.log.n_strategies()
        return ToolResult(f"Saved {trial.label} (code hash {h}). Strategy versions used: "
                          f"{used}/{self.cfg['agent']['max_strategies']}. Next: backtest / check_lookahead / walk_forward.")

    def _t_read_strategy(self, action: Action) -> ToolResult:
        t = self._trial(action)
        return ToolResult(f"{t.label}\nHypothesis: {t.hypothesis}\n\n{Path(t.path).read_text()}")

    def _t_backtest(self, action: Action) -> ToolResult:
        t = self._trial(action)
        res = self._job("backtest", t)
        if not res.get("ok"):
            return ToolResult(self._error(res))
        t.backtest = {**res["metrics"], "yearly": res["yearly"]}
        if t.sr_per_period is None and res["metrics"].get("sharpe") is not None:
            t.sr_per_period = res["metrics"]["sharpe"] / (self.bt_cfg["periods_per_year"] ** 0.5)
        self.log.save()
        warn = ("\nWARNINGS: " + "; ".join(res["warnings"])) if res["warnings"] else ""
        return ToolResult(
            f"Backtest of {t.label} on research period (IN-SAMPLE, params={res['params']}):\n"
            f"  strategy:   {_fmt(res['metrics'], KEY_METRICS)}\n"
            f"  buy & hold: {_fmt(res['buy_hold'], ['sharpe', 'cagr', 'ann_vol', 'max_drawdown'])}\n"
            f"  yearly returns: {res['yearly']}{warn}"
        )

    def _t_check_lookahead(self, action: Action) -> ToolResult:
        t = self._trial(action)
        res = self._job("lookahead", t)
        if not res.get("ok"):
            return ToolResult(self._error(res))
        t.lookahead = {"passed": res["passed"], "failures": res["failures"]}
        self.log.save()
        if res["passed"]:
            return ToolResult(f"Lookahead check PASSED for {t.label} ({res['n_checks']} truncation points; "
                              "signals for past dates never changed when future data was removed).")
        return ToolResult(
            f"Lookahead check FAILED for {t.label}: signals for past dates CHANGE when future data is removed, "
            f"so the strategy is using future information.\nDetails: {json.dumps(res['failures'])}\n"
            "Common causes: full-sample mean/std/quantiles, shift(-n), centered rolling windows, bfill."
        )

    def _t_walk_forward(self, action: Action) -> ToolResult:
        t = self._trial(action)
        res = self._job("walk_forward", t)
        if not res.get("ok"):
            return ToolResult(self._error(res))
        comb = res["combined_oos"]
        if comb.get("sharpe") is not None:
            t.sr_per_period = comb["sharpe"] / (self.bt_cfg["periods_per_year"] ** 0.5)
        t.walk_forward = {"oos_sharpe": comb.get("sharpe"), "combined_oos": comb, "folds": res["folds"],
                          "buy_hold_oos": res["buy_hold_oos"], "n_positive_folds": res["n_positive_folds"],
                          "grid_size": res["grid_size"], "series_path": res["series_path"]}
        dsr = self._current_dsr(t)
        t.walk_forward["dsr"] = dsr["dsr"] if dsr else None
        self.log.save()
        fold_lines = "\n".join(
            f"  fold {f['fold']} [{f['test_start']} -> {f['test_end']}] params={f['params']} "
            f"IS_SR={f['in_sample_sharpe']} OOS_SR={f['oos_sharpe']} ret={f['oos_return']} "
            f"maxDD={f['oos_max_dd']} | B&H_SR={f['buy_hold_sharpe']}"
            for f in res["folds"])
        return ToolResult(
            f"Walk-forward for {t.label} (grid size {res['grid_size']}, params chosen on past data only):\n"
            f"{fold_lines}\n"
            f"Combined OOS: {_fmt(comb, KEY_METRICS)}\n"
            f"Buy & hold over same OOS period: {_fmt(res['buy_hold_oos'], ['sharpe', 'cagr', 'max_drawdown'])}\n"
            f"Positive folds: {res['n_positive_folds']}/{len(res['folds'])}; fold Sharpe std: {res['fold_sharpe_std']}\n"
            f"Deflated Sharpe: {_fmt(dsr)}  (probability the edge is real after {dsr['n_trials'] if dsr else '?'} trials)"
        )

    def _t_research_log(self, action: Action) -> ToolResult:
        return ToolResult(self.log.summary_table())

    def _t_request_critique(self, action: Action) -> ToolResult:
        t = self._trial(action)
        if not t.lookahead or not t.walk_forward:
            return ToolResult(f"ERROR: run check_lookahead and walk_forward on {t.label} first.")
        dsr = self._current_dsr(t)
        wf_view = {k: t.walk_forward[k] for k in ("combined_oos", "folds", "buy_hold_oos", "n_positive_folds", "grid_size")}
        wf_view["deflated_sharpe"] = dsr
        bt = dict(t.backtest or {})
        payload = {"hypothesis": t.hypothesis, "code": Path(t.path).read_text(), "backtest": bt,
                   "yearly": bt.pop("yearly", None), "lookahead": t.lookahead, "walk_forward": wf_view,
                   "n_trials": self.log.n_strategies()}
        verdict = self.critic.review(payload)
        t.critique = verdict
        self.log.save()
        issues = "\n".join(f"  - {i}" for i in verdict["issues"]) or "  (none)"
        sugg = "\n".join(f"  - {s}" for s in verdict["suggestions"]) or "  (none)"
        return ToolResult(f"CRITIC VERDICT for {t.label}: {verdict['verdict'].upper()}\nIssues:\n{issues}\nSuggestions:\n{sugg}")

    def gate_check(self, t: Trial) -> list[str]:
        g = self.cfg["gates"]
        fails = []
        if not t.lookahead or not t.lookahead.get("passed"):
            fails.append("lookahead check has not passed for this version")
        if not t.walk_forward:
            fails.append("walk_forward has not been run for this version")
        else:
            comb = t.walk_forward["combined_oos"]
            sr, dd, n_tr = comb.get("sharpe") or 0, abs(comb.get("max_drawdown") or 1), comb.get("n_trades") or 0
            dsr = (self._current_dsr(t) or {}).get("dsr", 0)
            if sr < g["min_oos_sharpe"]:
                fails.append(f"OOS Sharpe {sr} < {g['min_oos_sharpe']}")
            if dsr < g["min_dsr"]:
                fails.append(f"deflated Sharpe {dsr} < {g['min_dsr']} (with {self.log.n_strategies()} trials)")
            if dd > g["max_drawdown"]:
                fails.append(f"OOS max drawdown {dd:.1%} > {g['max_drawdown']:.0%}")
            if n_tr < g["min_trades"]:
                fails.append(f"only {n_tr} OOS trades (< {g['min_trades']})")
        if not t.critique or t.critique.get("verdict") != "pass":
            fails.append("critic has not passed this version")
        return fails

    def _t_submit(self, action: Action) -> ToolResult:
        t = self._trial(action)
        fails = self.gate_check(t)
        if fails:
            return ToolResult(f"SUBMISSION REJECTED for {t.label}:\n- " + "\n- ".join(fails) +
                              "\nFix these, try another idea, or call finish.")
        res = self._job("holdout", t, data=self.full_path, research_end=self.research_end)
        if not res.get("ok"):
            return ToolResult("Holdout evaluation crashed: " + self._error(res))
        self.final = {"submitted": True, "strategy": t.label, "name": t.name, "summary": action.args.get("summary", ""),
                      "holdout": res, "dsr": self._current_dsr(t)}
        m, bh = res["metrics"], res["buy_hold"]
        return ToolResult(
            f"SUBMITTED {t.label}. One-time HOLDOUT evaluation ({m.get('start')} -> {m.get('end')}, params={res['params_used']}):\n"
            f"  strategy:   {_fmt(m, KEY_METRICS)}\n"
            f"  buy & hold: {_fmt(bh, ['sharpe', 'cagr', 'max_drawdown'])}\nRun complete.",
            done=True)

    def _t_finish(self, action: Action) -> ToolResult:
        self.final = {"submitted": False, "summary": action.args.get("summary", "")}
        return ToolResult("Run finished without a submission.", done=True)
