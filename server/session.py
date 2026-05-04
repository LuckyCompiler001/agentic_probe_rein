"""Single-user session manager.

Holds the currently-active pipeline thread, its frontend, and the per-run log
file. Only one pipeline can be running at a time (the workspace at
`mimic/` is shared, so concurrent runs would corrupt each other).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

from controller import run_pipeline, setup_run
from controller.actions import current_log_path
from server.web_frontend import WebFrontend


# ── Thread-aware stdout/stderr ────────────────────────────────────────────────
#
# print() calls in the pipeline thread are routed to the active run's log file
# without affecting prints from other threads (FastAPI, etc).

class _ThreadAwareStream:
    def __init__(self, original) -> None:
        self._original = original
        self._files: dict[int, object] = {}
        self._lock = threading.Lock()

    def register(self, thread_id: int, file_obj) -> None:
        with self._lock:
            self._files[thread_id] = file_obj

    def unregister(self, thread_id: int) -> None:
        with self._lock:
            self._files.pop(thread_id, None)

    def write(self, s: str):
        with self._lock:
            f = self._files.get(threading.get_ident())
        if f is not None:
            try:
                f.write(s)
            except Exception:
                self._original.write(s)
        else:
            self._original.write(s)

    def flush(self):
        with self._lock:
            f = self._files.get(threading.get_ident())
        if f is not None:
            try:
                f.flush()
            except Exception:
                self._original.flush()
        else:
            self._original.flush()


_stdout_router = _ThreadAwareStream(sys.stdout)
_stderr_router = _ThreadAwareStream(sys.stderr)


def install_stdout_routing() -> None:
    sys.stdout = _stdout_router  # type: ignore[assignment]
    sys.stderr = _stderr_router  # type: ignore[assignment]


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    """One-at-a-time pipeline runner."""

    def __init__(self) -> None:
        self.frontend: Optional[WebFrontend] = None
        self.thread: Optional[threading.Thread] = None
        self.run_dir: Optional[Path] = None
        self.log_path: Optional[Path] = None
        self.error: Optional[str] = None
        self.finished: bool = False
        self._lock = threading.Lock()

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self.thread is not None and self.thread.is_alive()

    @property
    def state(self) -> dict:
        with self._lock:
            return {
                "active": self.thread is not None and self.thread.is_alive(),
                "run_id": self.run_dir.name if self.run_dir else None,
                "pending": self.frontend.pending_question if self.frontend else None,
                "finished": self.finished,
                "error": self.error,
            }

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self, run_id: str | None) -> str:
        """Start a new pipeline thread targeting `run_id` (or new if None)."""
        with self._lock:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError("A pipeline is already running. Abort it first.")
            self.frontend = WebFrontend()
            self.run_dir = None
            self.log_path = None
            self.error = None
            self.finished = False

        # Pre-push the select_run answer so the worker doesn't block on it.
        # setup_run() consumes this immediately when it calls frontend.select_run().
        self.frontend._answer_q.put(run_id)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        return run_id or "<new>"

    def _worker(self) -> None:
        assert self.frontend is not None
        try:
            ctx = setup_run(self.frontend)
            with self._lock:
                self.run_dir = ctx.response_dir
                self.log_path = ctx.response_dir / "session.log"
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                # open append; route subprocess and prints through this fd
                log_fd = self.log_path.open("a", buffering=1)
            _stdout_router.register(threading.get_ident(), log_fd)
            _stderr_router.register(threading.get_ident(), log_fd)
            current_log_path.set(self.log_path)
            try:
                run_pipeline(ctx, self.frontend)
            finally:
                _stdout_router.unregister(threading.get_ident())
                _stderr_router.unregister(threading.get_ident())
                log_fd.close()
            with self._lock:
                self.finished = True
        except Exception as e:
            with self._lock:
                self.error = f"{type(e).__name__}: {e}"
                self.finished = True

    def submit_answer(self, value) -> None:
        if self.frontend is None:
            raise RuntimeError("No active session.")
        self.frontend.submit_answer(value)

    def abort(self) -> None:
        # Best-effort: push None into the queue to unstick a blocked ask, then
        # wait briefly. Real abort would need subprocess kill; out of scope MVP.
        if self.frontend is not None:
            try:
                self.frontend._answer_q.put_nowait(None)
            except Exception:
                pass


# Singleton session
SESSION = Session()
