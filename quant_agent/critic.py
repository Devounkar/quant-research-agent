"""The Critic agent: an independent LLM call that tries to break a strategy before submission."""
from __future__ import annotations

import json

from .llm import BaseLLM
from .parsing import ParseError, extract_json
from .prompts import CRITIC_SYSTEM


class Critic:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def review(self, payload: dict) -> dict:
        user = (
            "Review this strategy submission.\n\n"
            f"HYPOTHESIS:\n{payload['hypothesis']}\n\n"
            f"CODE:\n```python\n{payload['code']}\n```\n\n"
            f"FULL-PERIOD BACKTEST (in-sample, for context only):\n{json.dumps(payload.get('backtest'), indent=1)}\n\n"
            f"YEARLY RETURNS (full-period backtest):\n{json.dumps(payload.get('yearly'))}\n\n"
            f"LOOKAHEAD CHECK:\n{json.dumps(payload.get('lookahead'))}\n\n"
            f"WALK-FORWARD (out-of-sample):\n{json.dumps(payload.get('walk_forward'), indent=1)}\n\n"
            f"NUMBER OF STRATEGY VERSIONS TRIED SO FAR: {payload.get('n_trials')}\n"
        )
        try:
            text = self.llm.chat(CRITIC_SYSTEM, [{"role": "user", "content": user}])
            verdict = extract_json(text)
        except (ParseError, Exception) as e:  # an unparseable review counts as a fail
            return {"verdict": "fail", "issues": [f"critic response could not be parsed: {e}"], "suggestions": []}
        v = str(verdict.get("verdict", "fail")).strip().lower()
        return {
            "verdict": "pass" if v == "pass" else "fail",
            "issues": [str(i) for i in verdict.get("issues", [])][:10],
            "suggestions": [str(s) for s in verdict.get("suggestions", [])][:10],
        }
