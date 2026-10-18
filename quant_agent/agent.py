"""The Researcher agent loop: plan -> act (tool) -> observe -> revise."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from .backtest.validation import MAX_COMBOS
from .critic import Critic
from .data import load_data, split_holdout
from .llm import make_llm
from .memory import ResearchLog
from .parsing import ParseError, parse_action
from .prompts import RESEARCHER_SYSTEM
from .report import write_report
from .tools import ResearchTools, ToolResult, tool_docs

DEFAULT_TASK = ("Find a robust daily trading strategy for this instrument that beats buy-and-hold on a "
                "risk-adjusted basis out of sample. Start by understanding the data.")


class ResearchAgent:
    def __init__(self, cfg: dict, run_dir: str | Path | None = None, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(run_dir or Path(cfg["output_dir"]) / f"run-{stamp}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        dcfg = cfg["data"]
        full = load_data(dcfg)
        research, _ = split_holdout(full, dcfg["holdout_fraction"])
        self.instrument = "synthetic index (demo data)" if dcfg["source"] == "synthetic" else dcfg.get("ticker", "instrument")

        self.log = ResearchLog(self.run_dir / "research_log.json")
        self.llm = make_llm(cfg["llm"], "researcher")
        self.critic = Critic(make_llm(cfg["llm"], "critic"))
        self.tools = ResearchTools(cfg, self.run_dir, self.log, self.critic, research, full, self.instrument)

        g, a = cfg["gates"], cfg["agent"]
        self.system = RESEARCHER_SYSTEM.format(
            instrument=self.instrument, tool_docs=tool_docs(), max_combos=MAX_COMBOS,
            timeout=a["sandbox_timeout"], holdout_pct=int(dcfg["holdout_fraction"] * 100),
            cost_bps=cfg["backtest"]["cost_bps"], min_oos_sharpe=g["min_oos_sharpe"], min_dsr=g["min_dsr"],
            max_dd_pct=int(g["max_drawdown"] * 100), min_trades=g["min_trades"],
            max_steps=a["max_steps"], max_strategies=a["max_strategies"],
        )
        (self.run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    # ------------------------------------------------------------------ context
    def _messages(self, task: str, history: list[dict]) -> list[dict]:
        k = self.cfg["agent"]["context_turns"] * 2
        recent = history[-k:]
        if recent and recent[0]["role"] == "user":
            recent = recent[1:]
        first = f"TASK: {task}\n\nRESEARCH LOG (your long-term memory of every strategy tried):\n{self.log.summary_table()}"
        if len(recent) < len(history):
            first += "\n\n(Older turns were trimmed from the conversation. Use the research log above as your memory.)"
        return [{"role": "user", "content": first}] + recent

    def _status(self, step: int) -> str:
        a = self.cfg["agent"]
        left = a["max_steps"] - step
        s = f"[step {step}/{a['max_steps']} | strategy versions {self.log.n_strategies()}/{a['max_strategies']}]"
        if left <= 3:
            s += f" WARNING: only {left} steps left - submit a passing strategy or call finish."
        return s

    def _print(self, text: str):
        if self.verbose:
            print(text, flush=True)

    # ------------------------------------------------------------------ main loop
    def run(self, task: str = DEFAULT_TASK) -> dict:
        history: list[dict] = []
        transcript = self.run_dir / "transcript.md"
        transcript.write_text(f"# Agent transcript\n\n**Task:** {task}\n\n")
        max_steps = self.cfg["agent"]["max_steps"]
        t0 = time.time()
        self._print(f"Run directory: {self.run_dir}\nTask: {task}\n")

        for step in range(1, max_steps + 1):
            reply = self.llm.chat(self.system, self._messages(task, history))
            try:
                action = parse_action(reply)
                self._print(f"\n=== Step {step}: {action.tool} {json.dumps(action.args)[:150]}")
                if action.thought:
                    self._print(f"THOUGHT: {action.thought[:400]}")
                result = self.tools.call(action, step)
            except ParseError as e:
                self._print(f"\n=== Step {step}: (unparseable reply)")
                result = ToolResult(f"FORMAT ERROR: {e}. Reply again using exactly: THOUGHT / ACTION / ARGS / CODE.")

            obs = f"{result.text}\n\n{self._status(step)}"
            self._print(f"OBSERVATION:\n{result.text[:1200]}")
            history += [{"role": "assistant", "content": reply}, {"role": "user", "content": f"OBSERVATION:\n{obs}"}]
            with transcript.open("a") as f:
                f.write(f"## Step {step}\n\n### Agent\n\n{reply}\n\n### Observation\n\n```\n{obs}\n```\n\n")
            if result.done:
                break

        final = self.tools.final or {"submitted": False, "summary": "Step budget exhausted without submit/finish."}
        final["steps_used"] = step
        final["elapsed_sec"] = round(time.time() - t0, 1)
        final["llm_calls"] = self.llm.n_calls + self.critic.llm.n_calls
        (self.run_dir / "final.json").write_text(json.dumps(final, indent=2, default=str))
        report = write_report(self.run_dir, self.cfg, self.log, final, self.tools.full_path, self.tools.research_end)
        self._print(f"\nDone in {final['elapsed_sec']}s. Report: {report}")
        return final
