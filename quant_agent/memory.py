"""Research log: the agent's long-term memory of every strategy version it has tried.

It is persisted to disk and shown to the agent each turn, so older conversation turns can be
trimmed from the context without the agent forgetting what it already tested. It also counts
trials, which the deflated Sharpe ratio uses to penalise data snooping.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Trial:
    name: str
    version: int
    code_hash: str
    hypothesis: str
    path: str
    created_step: int
    backtest: dict | None = None
    lookahead: dict | None = None
    walk_forward: dict | None = None
    critique: dict | None = None
    sr_per_period: float | None = None  # best estimate: walk-forward OOS if available

    @property
    def label(self) -> str:
        return f"{self.name} v{self.version}"


@dataclass
class ResearchLog:
    path: Path
    trials: dict = field(default_factory=dict)   # code_hash -> Trial
    latest: dict = field(default_factory=dict)   # name -> code_hash

    @staticmethod
    def hash_code(code: str) -> str:
        return hashlib.sha1(code.strip().encode()).hexdigest()[:10]

    def add_version(self, name: str, code: str, hypothesis: str, path: str, step: int) -> tuple[Trial, bool]:
        h = self.hash_code(code)
        if h in self.trials:
            self.latest[name] = h
            return self.trials[h], False
        version = 1 + sum(t.name == name for t in self.trials.values())
        trial = Trial(name, version, h, hypothesis, path, step)
        self.trials[h] = trial
        self.latest[name] = h
        self.save()
        return trial, True

    def get(self, name: str) -> Trial | None:
        h = self.latest.get(name)
        return self.trials.get(h) if h else None

    def n_strategies(self) -> int:
        return len(self.trials)

    def trial_sharpes(self) -> list[float]:
        return [t.sr_per_period for t in self.trials.values() if t.sr_per_period is not None]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(t) for t in self.trials.values()], indent=2, default=str))

    def summary_table(self) -> str:
        if not self.trials:
            return "(no strategies tested yet)"
        rows = ["strategy | hypothesis | backtest SR | lookahead | WF OOS SR | DSR | critic"]
        for t in self.trials.values():
            bt = t.backtest.get("sharpe") if t.backtest else "-"
            la = "-" if not t.lookahead else ("pass" if t.lookahead.get("passed") else "FAIL")
            wf = t.walk_forward.get("oos_sharpe") if t.walk_forward else "-"
            dsr = t.walk_forward.get("dsr") if t.walk_forward else "-"
            cr = t.critique.get("verdict", "-") if t.critique else "-"
            hyp = (t.hypothesis[:70] + "...") if len(t.hypothesis) > 70 else t.hypothesis
            rows.append(f"{t.label} | {hyp} | {bt} | {la} | {wf} | {dsr} | {cr}")
        return "\n".join(rows)
