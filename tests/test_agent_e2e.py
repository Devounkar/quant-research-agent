"""End-to-end run with the scripted mock LLM (no API key needed)."""
import json

from quant_agent.agent import ResearchAgent
from quant_agent.config import load_config


def test_mock_run_completes(tmp_path):
    cfg = load_config(None, {"llm": {"provider": "mock"}, "data": {"source": "synthetic"}})
    agent = ResearchAgent(cfg, run_dir=tmp_path / "run", verbose=False)
    final = agent.run()
    run = tmp_path / "run"
    assert (run / "report.md").exists() and (run / "transcript.md").exists()
    log = json.loads((run / "research_log.json").read_text())
    names = {t["name"]: t for t in log}
    # the deliberately buggy strategy must have been caught
    assert names["momentum_z"]["lookahead"]["passed"] is False
    assert names["trend_filter"]["lookahead"]["passed"] is True
    assert names["trend_filter"]["walk_forward"] is not None
    assert "submitted" in final


def test_submit_blocked_without_checks(tmp_path):
    from quant_agent.parsing import Action
    cfg = load_config(None, {"llm": {"provider": "mock"}})
    agent = ResearchAgent(cfg, run_dir=tmp_path / "r", verbose=False)
    code = 'PARAMS = {}\ndef generate_signals(df, params):\n    return (df["close"] > 0).astype(float)\n'
    agent.tools.call(Action("write_strategy", {"name": "always_long", "hypothesis": "equity premium"}, code), 1)
    res = agent.tools.call(Action("submit", {"name": "always_long", "summary": "x"}), 2)
    assert "REJECTED" in res.text and not res.done
    # the holdout must not have been touched
    assert agent.tools.final is None
