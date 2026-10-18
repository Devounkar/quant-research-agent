"""Runs a sandbox job in a separate Python process with a timeout."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from .runner import MARKER

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_job(mode: str, run_dir: Path, data_path: Path, timeout: int = 60, *, code: str | None = None,
            strategy_path: Path | None = None, cfg: dict | None = None, research_end: str | None = None,
            out_dir: Path | None = None) -> dict:
    jobs = Path(run_dir) / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    job_path = jobs / f"{mode}_{uuid.uuid4().hex[:8]}.json"
    job = {
        "mode": mode, "data_path": str(data_path), "timeout": timeout, "code": code,
        "strategy_path": str(strategy_path) if strategy_path else None, "cfg": cfg or {},
        "research_end": research_end, "out_dir": str(out_dir) if out_dir else None,
    }
    job_path.write_text(json.dumps(job))

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "quant_agent.sandbox.runner", str(job_path)],
            capture_output=True, text=True, timeout=timeout, cwd=str(PROJECT_ROOT), env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout}s (make the code faster / vectorised)"}

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    err = (proc.stderr or proc.stdout or "no output").strip().splitlines()[-8:]
    return {"ok": False, "error": "sandbox process crashed", "traceback": "\n".join(err)}
