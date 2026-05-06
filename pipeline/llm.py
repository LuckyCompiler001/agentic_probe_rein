"""Subprocess wrappers for the `claude` CLI used as NLP and as a code agent.

Two flavors:
- nlp_call: short, JSON-returning, no tools, no session persistence
- agent_call: long-running, allowed to edit files in the workspace

Both register their Popen handle in a module-level slot so the API server
can cancel an in-flight stage action via cancel_current().
"""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

NLP_MODEL = "opus"
AGENT_MODEL = "opus"


# ── Cancellation registry ────────────────────────────────────────────────────
_current_proc: subprocess.Popen | None = None
_current_proc_lock = threading.Lock()


def _register(p: subprocess.Popen) -> None:
    global _current_proc
    with _current_proc_lock:
        _current_proc = p


def _unregister() -> None:
    global _current_proc
    with _current_proc_lock:
        _current_proc = None


def cancel_current() -> bool:
    """Kill the active subprocess if any. Returns True iff a process was killed."""
    with _current_proc_lock:
        p = _current_proc
    if p is None or p.poll() is not None:
        return False
    try:
        p.kill()
    except ProcessLookupError:
        return False
    return True


# ── Calls ────────────────────────────────────────────────────────────────────
def nlp_call(message: str, *, model: str = NLP_MODEL) -> dict:
    """Call the NLP model and parse its JSON response."""
    p = subprocess.Popen(
        [
            "claude",
            "-p",
            "--model",
            model,
            "--tools",
            "",
            "--no-session-persistence",
            message,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _register(p)
    try:
        stdout, stderr = p.communicate()
    finally:
        _unregister()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args, stdout, stderr)
    return json.loads(stdout)


def agent_call(
    prompt: str,
    *,
    cwd: Path,
    log_path: Path | None = None,
    model: str = AGENT_MODEL,
) -> None:
    """Run the code agent inside `cwd`. Streams stdout/stderr to log_path if given."""
    cmd = [
        "claude",
        "-p",
        "--dangerously-skip-permissions",
        "--model",
        model,
        prompt,
    ]
    if log_path is None:
        p = subprocess.Popen(cmd, cwd=str(cwd))
        _register(p)
        try:
            p.wait()
        finally:
            _unregister()
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, p.args)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as f:
        f.write(b"\n--- agent_call ---\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
        _register(p)
        try:
            p.wait()
        finally:
            _unregister()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args)
