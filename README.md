# Quant Research Agent

An **autonomous LLM agent** that does quant research on its own: it forms a hypothesis, writes a
trading strategy in Python, tests it, reads the results, and decides what to try next. A separate
**Critic agent** tries to break every strategy before it can be submitted.

The main focus is **not fooling itself**. Most backtests look great and fail live. This agent has
guardrails against the classic mistakes: lookahead bias, overfitting, and testing too many ideas.

---

## What makes it an agent (not a pipeline)

The LLM controls the loop. Nothing is hard-coded about which steps happen or in what order.

```
        ┌──────────────────────────────────────────────┐
        │              RESEARCHER (LLM)                │
        │  THOUGHT → picks ONE tool → reads OBSERVATION │
        └───────────────┬──────────────────────────────┘
                        │ tool call
   ┌──────────┬─────────┼──────────┬──────────────┬──────────────┐
   ▼          ▼         ▼          ▼              ▼              ▼
data_info  run_analysis write_   backtest /    request_       submit
           (explore)    strategy  lookahead /   critique  ──►  (gates → hidden
                                  walk_forward  (CRITIC LLM)    holdout, once)
                        │
                        ▼
              Sandboxed subprocess (AST-checked code, timeout)
                        │
                        ▼
        Research log (memory of every strategy + results)
```

- **Tools:** the agent explores data, writes strategy files, runs tests and asks for reviews.
- **Memory:** a research log records every strategy version tried. It is shown to the agent every
  turn, so old conversation turns can be dropped without the agent forgetting what it tested.
- **Multi-agent:** the Researcher and the Critic are separate LLM calls with different goals.
- **Self-correction:** failed checks come back as observations, and the agent must fix or move on.

---

## Safeguards against fooling yourself

| Problem | How this project handles it |
|---|---|
| **Lookahead bias** (using future data) | `check_lookahead` re-runs the strategy on cut-off history. If signals for past dates change when future data is removed, the strategy is peeking. Catches full-sample z-scores, `shift(-1)`, centered windows, etc. |
| **Overfitting parameters** | `walk_forward` uses expanding windows. Parameters are picked only on past data, then tested on the next unseen period. |
| **Trying too many ideas** | Every strategy version is counted as a trial. The **Deflated Sharpe Ratio** (Bailey & López de Prado) raises the bar as trials grow. |
| **Peeking at the final test** | The last 20% of data is a **hidden holdout**. The agent's tools never see it. It is used exactly once, at submission. |
| **Wishful thinking** | Submission needs a "pass" from the independent **Critic** plus hard gates (OOS Sharpe, DSR, max drawdown, min trades). |
| **Ignoring costs** | Transaction costs are charged on every position change. |
| **Unsafe LLM code** | Code is AST-checked (no file/network/OS access, allow-listed imports) and runs in a separate process with a timeout. |

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Offline demo, no API key needed (scripted "LLM", synthetic data)
python run_agent.py --provider mock

# 2. Real run with Claude
cp .env.example .env          # then paste your ANTHROPIC_API_KEY into .env
python run_agent.py

# 3. Real Nifty 50 data from Yahoo Finance
python run_agent.py --data-source yfinance --ticker ^NSEI

# 4. Your own research question
python run_agent.py --data-source yfinance --ticker ^NSEBANK \
  --task "Look for a short-term mean-reversion edge. Keep turnover reasonable."

# Other providers
python run_agent.py --provider openai --model gpt-4o
python run_agent.py --provider ollama --model qwen2.5-coder:14b      # free, local
```

Each run creates a folder in `runs/` with:

- `report.md`: result, holdout performance, final code, every strategy tried, critic notes
- `equity_curve.png`: out-of-sample equity vs buy-and-hold, with the holdout shaded
- `transcript.md`: the agent's full step-by-step reasoning
- `strategies/`: every strategy version it wrote
- `research_log.json`: structured results for every trial

See `examples/sample_run/` for the output of the offline demo.

### Test a strategy by hand (no LLM)

```bash
python evaluate.py examples/strategies/vol_target_trend.py
python evaluate.py examples/strategies/BUGGY_lookahead_example.py   # watch the lookahead check fail
```

### Run the tests

```bash
pytest -q
```

---

## Strategy format

The agent (or you) writes strategies like this:

```python
import pandas as pd

PARAMS = {"fast": 20, "slow": 150}                        # defaults
PARAM_GRID = {"fast": [10, 20, 50], "slow": [100, 200]}   # optional, max 16 combos

def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    # df has columns open, high, low, close, volume (daily)
    # return a target position in [-1, 1] for each day, using data up to that day's close
    fast = df["close"].rolling(params["fast"]).mean()
    slow = df["close"].rolling(params["slow"]).mean()
    return (fast > slow).astype(float)
```

The engine trades on the **next** day automatically, so a signal at today's close earns
tomorrow's return.

---

## Configuration

Everything lives in `config.yaml`: LLM provider and model, data source, cost (bps), number of
walk-forward folds, step and strategy budgets, and the submission gates. Any setting can also be
overridden from the command line (`python run_agent.py --help`).

Tip: set `critic_model` to a stronger model than the researcher for a tougher reviewer.

---

## Project structure

```
quant-research-agent/
├── run_agent.py              # CLI: run the agent
├── evaluate.py               # CLI: test a strategy file by hand
├── config.yaml
├── quant_agent/
│   ├── agent.py              # main loop: think → act → observe, context trimming
│   ├── tools.py              # the 11 tools + submission gates
│   ├── critic.py             # Critic agent
│   ├── prompts.py            # Researcher + Critic system prompts
│   ├── memory.py             # research log (long-term memory, trial counting)
│   ├── parsing.py            # parses THOUGHT / ACTION / ARGS / CODE replies
│   ├── llm.py                # Anthropic / OpenAI / Ollama clients
│   ├── mock_llm.py           # scripted LLM for offline demo + tests
│   ├── data.py               # synthetic / CSV / Yahoo Finance data, holdout split
│   ├── report.py             # markdown report + equity chart
│   ├── backtest/
│   │   ├── engine.py         # vectorised backtest with costs
│   │   ├── metrics.py        # Sharpe, drawdown, turnover, etc.
│   │   └── validation.py     # lookahead check, walk-forward, deflated Sharpe
│   └── sandbox/
│       ├── safety.py         # AST checks on LLM code
│       ├── executor.py       # runs jobs in a subprocess with timeout
│       └── runner.py         # code that executes inside the subprocess
├── examples/strategies/      # example strategies (incl. a deliberately buggy one)
├── examples/sample_run/      # output of the offline demo
└── tests/                    # 26 tests: engine, lookahead, safety, parser, end-to-end
```

---

## Design choices

- **Plain-text action format instead of native tool calling.** Works the same with any model,
  including small local ones. Code goes in a fenced block rather than inside JSON, because LLMs
  often break JSON escaping on multi-line code.
- **One action per turn.** Easier to debug and gives a clean, readable transcript.
- **The holdout lives only in the submit tool.** The agent cannot "accidentally" tune to it.
- **An honest null result is a valid outcome.** The agent is told to call `finish` if nothing
  clears the gates, rather than forcing a weak strategy through.

---

## Limitations

- Single instrument, daily bars, close-to-close fills. No intraday data, slippage model, or
  short-borrow costs.
- The AST check is a guardrail, not a security boundary. For untrusted setups, run inside Docker.
- The synthetic data is only for testing the pipeline. Never draw trading conclusions from it.
- Not financial advice. Results on historical data do not guarantee future performance.

---

## Ideas to extend it

- Multi-asset universe (e.g. Nifty 50 stocks) with cross-sectional strategies
- Add a third agent: a "Portfolio" agent that combines several passing strategies
- Use the Critic's past verdicts as few-shot examples so the Researcher learns faster
- Benchmark different LLMs on the same task: success rate, steps used, overfitting rate
- Cost-sensitivity and parameter-sensitivity tools the agent can call
