#!/usr/bin/env python
"""Run the autonomous quant research agent.

Examples:
    python run_agent.py --provider mock                       # offline demo, no API key
    python run_agent.py                                        # uses config.yaml (Claude by default)
    python run_agent.py --provider openai --model gpt-4o
    python run_agent.py --data-source yfinance --ticker ^NSEI --task "Find a mean-reversion strategy"
"""
import argparse
import os
import sys

from quant_agent.agent import DEFAULT_TASK, ResearchAgent
from quant_agent.config import load_config


def _load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    p = argparse.ArgumentParser(description="Autonomous LLM quant research agent")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--provider", choices=["anthropic", "openai", "ollama", "mock"])
    p.add_argument("--model")
    p.add_argument("--critic-model")
    p.add_argument("--data-source", choices=["synthetic", "csv", "yfinance"])
    p.add_argument("--ticker")
    p.add_argument("--csv-path")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-strategies", type=int)
    p.add_argument("--run-dir")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    _load_dotenv()
    ov: dict = {"llm": {}, "data": {}, "agent": {}}
    if a.provider: ov["llm"]["provider"] = a.provider
    if a.model: ov["llm"]["model"] = a.model
    if a.critic_model: ov["llm"]["critic_model"] = a.critic_model
    if a.data_source: ov["data"]["source"] = a.data_source
    if a.ticker: ov["data"]["ticker"] = a.ticker
    if a.csv_path: ov["data"]["csv_path"] = a.csv_path
    if a.max_steps: ov["agent"]["max_steps"] = a.max_steps
    if a.max_strategies: ov["agent"]["max_strategies"] = a.max_strategies
    cfg = load_config(a.config, ov)

    if cfg["llm"]["provider"] == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to .env, or try: python run_agent.py --provider mock")
    if cfg["llm"]["provider"] == "openai" and not os.environ.get("OPENAI_API_KEY") and not cfg["llm"].get("base_url"):
        sys.exit("OPENAI_API_KEY is not set. Add it to .env.")

    agent = ResearchAgent(cfg, run_dir=a.run_dir, verbose=not a.quiet)
    final = agent.run(a.task)
    print("\nSubmitted:" if final.get("submitted") else "\nNo submission.", final.get("strategy", ""))


if __name__ == "__main__":
    main()
