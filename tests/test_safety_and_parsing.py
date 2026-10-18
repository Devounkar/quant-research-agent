import pytest

from quant_agent.parsing import ParseError, extract_json, parse_action
from quant_agent.sandbox import check_code

GOOD = '''import pandas as pd
PARAMS = {}
def generate_signals(df, params):
    return df["close"].pct_change().gt(0).astype(float)
'''


def test_good_strategy_passes():
    assert check_code(GOOD) == []


@pytest.mark.parametrize("bad", [
    "import os\n" + GOOD,
    "import subprocess\n" + GOOD,
    GOOD + "\nx = open('/etc/passwd').read()\n",
    GOOD + "\ndf_ = None\nimport requests\n",
    GOOD.replace('df["close"]', 'df.__class__'),
    GOOD + "\nimport pandas as pd\npd.read_csv('x.csv')\n",
    "def f(df, params): return 1\n",          # missing generate_signals
    "def generate_signals(df: return 1\n",   # syntax error
])
def test_bad_code_rejected(bad):
    assert check_code(bad)


def test_parse_full_action():
    text = '''THOUGHT: test a trend idea
ACTION: write_strategy
ARGS: {"name": "trend", "hypothesis": "regimes persist"}
CODE:
```python
PARAMS = {}
def generate_signals(df, params):
    return df["close"] * 0
```'''
    a = parse_action(text)
    assert a.tool == "write_strategy"
    assert a.args == {"name": "trend", "hypothesis": "regimes persist"}
    assert "generate_signals" in a.code and a.thought == "test a trend idea"


def test_parse_markdown_bold_and_fenced_json():
    text = '**THOUGHT:** hmm\n**ACTION:** backtest\n**ARGS:**\n```json\n{"name": "x"}\n```'
    a = parse_action(text)
    assert a.tool == "backtest" and a.args == {"name": "x"} and a.code is None


def test_parse_errors():
    with pytest.raises(ParseError):
        parse_action("I think we should backtest")
    with pytest.raises(ParseError):
        parse_action('ACTION: backtest\nARGS: {"name": }')


def test_extract_json_from_chatter():
    assert extract_json('Sure! {"verdict": "fail", "issues": []} done')["verdict"] == "fail"
