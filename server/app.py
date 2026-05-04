"""FastAPI server for the agentic-probe pipeline.

Single-user. CORS open for localhost. The pipeline runs in a worker thread
managed by `server.session.SESSION`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from controller.config import (
    LEGACY_DEFAULT_PROJECT,
    PROJECTS_BASE,
    RUN_BASE,
    list_projects,
)
from server.session import SESSION, install_stdout_routing


install_stdout_routing()

app = FastAPI(title="agentic-probe")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _list_runs() -> list[str]:
    if not RUN_BASE.exists():
        return []
    return sorted(
        p.name for p in RUN_BASE.iterdir() if p.is_dir() and p.name.isdigit()
    )


def _run_dir(run_id: str) -> Path:
    p = RUN_BASE / run_id
    if not p.is_dir() or not run_id.isdigit():
        raise HTTPException(404, f"Run {run_id} not found")
    return p


def _resolve_working_dir(run_id: str) -> Path:
    """Look up the project chosen for `run_id` and resolve to its workspace.

    Falls back to LEGACY_DEFAULT_PROJECT for legacy runs that pre-date the
    select_project step.
    """
    pb_path = _run_dir(run_id) / "progressbar.json"
    project = LEGACY_DEFAULT_PROJECT
    if pb_path.exists():
        try:
            data = json.loads(pb_path.read_text())
            for s in data.get("steps", []):
                if s.get("name") == "select_project" and s.get("done"):
                    if isinstance(s.get("answer"), str):
                        project = s["answer"]
                        break
        except Exception:
            pass
    return PROJECTS_BASE / project


def _read_json_safe(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return {"_error": f"failed to parse {path.name}: {e}"}


def _summarize_run(run_id: str) -> dict:
    rd = RUN_BASE / run_id
    pb_path = rd / "progressbar.json"
    summary: dict[str, Any] = {
        "run_id": run_id,
        "steps_done": 0,
        "current_step": None,
        "project": LEGACY_DEFAULT_PROJECT,
    }
    if pb_path.exists():
        try:
            data = json.loads(pb_path.read_text())
            steps = data.get("steps", [])
            summary["steps_done"] = sum(1 for s in steps if s.get("done"))
            summary["mode"] = next(
                ("auto_research" if s.get("answer") else "normal"
                 for s in steps if s["name"] == "auto_research_choice"),
                "unknown",
            )
            for s in steps:
                if s.get("name") == "select_project" and isinstance(s.get("answer"), str):
                    summary["project"] = s["answer"]
                    break
            mtime = pb_path.stat().st_mtime
            summary["last_activity"] = mtime
        except Exception:
            pass
    return summary


# ── routes: run inventory ─────────────────────────────────────────────────────

@app.get("/api/runs")
def list_runs():
    return {"runs": [_summarize_run(r) for r in _list_runs()]}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    rd = _run_dir(run_id)
    pb = rd / "progressbar.json"
    return {
        "run_id": run_id,
        "progressbar": _read_json_safe(pb) or {"steps": []},
        "files": {
            "probe_designs": _read_json_safe(rd / "probe_designs.json"),
            "probe_confidenced": _read_json_safe(rd / "probe_confidenced.json"),
            "dev_doc": _read_json_safe(rd / "dev_doc.json"),
            "dev_doc_confidenced": _read_json_safe(rd / "dev_doc_confidenced.json"),
        },
    }


# ── routes: file editing (cheap PATCH, no cascade) ────────────────────────────

class FileUpdate(BaseModel):
    content: dict | list


_EDITABLE_FILES = {
    "probe_confidenced",
    "dev_doc_confidenced",
    "probe_designs",
    "dev_doc",
}


@app.patch("/api/runs/{run_id}/files/{name}")
def patch_file(run_id: str, name: str, body: FileUpdate):
    if name not in _EDITABLE_FILES:
        raise HTTPException(400, f"File {name!r} is not editable")
    rd = _run_dir(run_id)
    target = rd / f"{name}.json"
    target.write_text(json.dumps(body.content, indent=2))
    return {"saved": str(target), "size": target.stat().st_size}


# ── routes: project inventory ────────────────────────────────────────────────

@app.get("/api/projects")
def list_projects_endpoint():
    return {"projects": list_projects()}


# ── routes: probe results (per-run workspace artifacts) ──────────────────────

@app.get("/api/runs/{run_id}/probe-results")
def list_probe_results(run_id: str):
    wd = _resolve_working_dir(run_id)
    metric_dir = wd / ".agent_probe" / "metric"
    if not metric_dir.exists():
        return {"results": []}
    out: list[dict] = []
    for p in sorted(metric_dir.glob("probe_result_*.json")):
        try:
            n = int(p.stem.rsplit("_", 1)[-1])
            out.append({"n": n, **json.loads(p.read_text())})
        except Exception:
            continue
    out.sort(key=lambda r: r["n"])
    return {"results": out}


@app.get("/api/runs/{run_id}/probe-plot/{n}")
def get_plot(run_id: str, n: int):
    wd = _resolve_working_dir(run_id)
    p = wd / ".agent_probe" / "plot" / f"probe_result_{n}.pdf"
    if not p.exists():
        raise HTTPException(404, "plot not found")
    return FileResponse(p, media_type="application/pdf")


# ── routes: session (pipeline thread control) ────────────────────────────────

class StartRequest(BaseModel):
    run_id: str | None = None


@app.get("/api/session")
def session_state():
    return SESSION.state


@app.post("/api/session/start")
def session_start(body: StartRequest):
    if body.run_id is not None and not (RUN_BASE / body.run_id).is_dir():
        raise HTTPException(404, f"Run {body.run_id} not found")
    try:
        rid = SESSION.start(body.run_id)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"started": rid}


class AnswerRequest(BaseModel):
    value: Any


@app.post("/api/session/answer")
def session_answer(body: AnswerRequest):
    try:
        SESSION.submit_answer(body.value)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/api/session/abort")
def session_abort():
    SESSION.abort()
    return {"ok": True}


# ── routes: threshold override (one-shot, runs in background thread) ─────────

class ThresholdRequest(BaseModel):
    run_id: str
    plan_idx: int | None = None
    new_threshold: str


@app.post("/api/threshold-override")
def threshold_override(body: ThresholdRequest):
    """Update threshold on the named run; if prober.py exists, dispatch the
    propagation agent in a worker thread (non-blocking).
    """
    rd = _run_dir(body.run_id)
    # Always update the plan json (cheap, sync)
    if body.plan_idx is not None:
        dev_doc = rd / "dev_doc_confidenced.json"
        if dev_doc.exists():
            data = json.loads(dev_doc.read_text())
            data["dev_plans"][body.plan_idx]["threshold"] = body.new_threshold
            dev_doc.write_text(json.dumps(data, indent=2))

    wd = _resolve_working_dir(body.run_id)
    prober_path = wd / "prober.py"
    if not prober_path.exists():
        return {"updated_dev_doc": True, "agent_dispatched": False}

    # Dispatch the agent propagation in a thread so the HTTP call returns fast.
    import threading
    from controller.actions import agent_call
    new_threshold = body.new_threshold

    def _dispatch():
        agent_call(
            "The probe threshold has been manually overridden. The new threshold is:\n"
            f"    {new_threshold}\n\n"
            "Tasks (do all of them, in order):\n"
            "1. Open prober.py and update every reference to the threshold value to the new "
            "value above. Keep the metric, direction, and PASS/FAIL semantics unchanged.\n"
            "2. If prober.py imports helpers that also hold a copy of the threshold, update them.\n"
            "3. For every existing file under .agent_probe/metric/probe_result_*.json, "
            "re-evaluate the 'status' field against the new threshold using the metric "
            "values already recorded, update the 'threshold' field, and rewrite the "
            "'conclusion' string. Do not change the 'values' arrays.\n"
            "4. Do not run training. Just save the file changes.",
            cwd=wd,
        )

    threading.Thread(target=_dispatch, daemon=True).start()
    return {"updated_dev_doc": True, "agent_dispatched": True}


# ── routes: log streaming (SSE) ──────────────────────────────────────────────

@app.get("/api/runs/{run_id}/log/stream")
async def stream_log(run_id: str, request: Request):
    """SSE stream of session.log for the given run.

    Sends current contents on connect, then tails new lines as they're written.
    """
    rd = _run_dir(run_id)
    log_path = rd / "session.log"

    async def event_stream():
        # Wait for the file to exist (worker may not have created it yet)
        while not log_path.exists():
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.5)

        with log_path.open("r") as f:
            # First, send all existing content
            initial = f.read()
            if initial:
                for chunk in initial.splitlines(keepends=False):
                    yield f"data: {json.dumps(chunk)}\n\n"

            # Then tail
            while True:
                if await request.is_disconnected():
                    return
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.3)
                    continue
                yield f"data: {json.dumps(line.rstrip())}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── routes: snapshot diff (per-run) ──────────────────────────────────────────

@app.get("/api/runs/{run_id}/snapshots")
def list_snapshots(run_id: str):
    wd = _resolve_working_dir(run_id)
    snap_dir = wd / ".agent_probe" / "snapshot"
    if not snap_dir.exists():
        return {"snapshots": []}
    out = []
    for p in sorted(snap_dir.glob("train_version_*.py")):
        try:
            n = int(p.stem.rsplit("_", 1)[-1])
            out.append({"n": n, "size": p.stat().st_size})
        except ValueError:
            continue
    out.sort(key=lambda x: x["n"])
    return {"snapshots": out}


@app.get("/api/runs/{run_id}/snapshots/{n}")
def get_snapshot(run_id: str, n: int):
    wd = _resolve_working_dir(run_id)
    p = wd / ".agent_probe" / "snapshot" / f"train_version_{n}.py"
    if not p.exists():
        raise HTTPException(404)
    return {"n": n, "content": p.read_text()}
