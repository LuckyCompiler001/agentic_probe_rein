"""Frontend driver for the FastAPI server.

The pipeline runs in a worker thread. Whenever it calls one of the `ask_*`
methods, the thread blocks on `_answer_q`. The HTTP layer:
  - inspects `pending_question` to render the active prompt in the UI
  - calls `submit_answer()` to release the worker

`show()` and `show_artifact()` append to the run-wide event buffer so the
log viewer can stream them via SSE.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PendingQuestion:
    kind: str  # "yn" | "int_range" | "pos_int" | "text" | "select_run" | "select_project"
    question: str
    default: Any = None
    lo: int | None = None
    hi: int | None = None
    existing: list[str] | None = None
    available: list[str] | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class WebFrontend:
    """Thread-safe Frontend implementation that pairs with the HTTP layer."""

    def __init__(self) -> None:
        self._answer_q: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._pending: PendingQuestion | None = None

    # ── public inspection (HTTP layer) ────────────────────────────────────────

    @property
    def pending_question(self) -> dict | None:
        with self._lock:
            return self._pending.to_dict() if self._pending else None

    def submit_answer(self, value: Any) -> None:
        """Push an answer to release the worker thread. Validation happens here
        — invalid answers raise so the HTTP layer can return a 400."""
        with self._lock:
            pending = self._pending
        if pending is None:
            raise RuntimeError("No pending question to answer.")
        validated = self._validate(pending, value)
        self._answer_q.put(validated)

    @staticmethod
    def _validate(pending: PendingQuestion, value: Any) -> Any:
        kind = pending.kind
        if kind == "yn":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.upper() in ("Y", "N"):
                return value.upper() == "Y"
            raise ValueError(f"Expected boolean for yn, got {value!r}")
        if kind == "int_range":
            v = int(value)
            if pending.lo is not None and v < pending.lo:
                raise ValueError(f"Below lo={pending.lo}")
            if pending.hi is not None and v > pending.hi:
                raise ValueError(f"Above hi={pending.hi}")
            return v
        if kind == "pos_int":
            v = int(value)
            if v <= 0:
                raise ValueError("Must be positive")
            return v
        if kind == "text":
            return str(value).strip()
        if kind == "select_run":
            if value is None or value == "":
                return None  # new run
            if pending.existing and value not in pending.existing:
                raise ValueError(f"Run id {value!r} not found")
            return value
        if kind == "select_project":
            if not isinstance(value, str) or not value:
                raise ValueError("Expected a project name (string)")
            if pending.available and value not in pending.available:
                raise ValueError(f"Project {value!r} not available")
            return value
        raise ValueError(f"Unknown kind: {kind}")

    # ── Frontend protocol (called by pipeline thread) ─────────────────────────

    def _ask(self, q: PendingQuestion) -> Any:
        with self._lock:
            self._pending = q
        try:
            return self._answer_q.get()
        finally:
            with self._lock:
                self._pending = None

    def ask_yn(self, question: str, default: bool | None = None) -> bool:
        return self._ask(PendingQuestion(kind="yn", question=question, default=default))

    def ask_int_range(self, question: str, lo: int, hi: int) -> int:
        return self._ask(PendingQuestion(kind="int_range", question=question, lo=lo, hi=hi))

    def ask_pos_int(self, question: str, default: int = 3) -> int:
        return self._ask(PendingQuestion(kind="pos_int", question=question, default=default))

    def ask_text(self, question: str) -> str:
        return self._ask(PendingQuestion(kind="text", question=question))

    def show(self, message: str) -> None:
        # Pipeline modules use plain print() heavily; show() is rarely used.
        # Route through stdout — the worker's stdout is redirected to the run log.
        print(message)

    def show_artifact(self, name: str, content: str) -> None:
        # The frontend already renders the JSON file from /runs/{id}, so we
        # don't need to push the content. A short marker keeps the log readable.
        print(f"[show_artifact] {name}")

    def select_run(self, existing: list[str]) -> str | None:
        return self._ask(
            PendingQuestion(kind="select_run", question="Select run", existing=existing)
        )

    def select_project(self, available: list[str], default: str | None = None) -> str:
        return self._ask(
            PendingQuestion(
                kind="select_project",
                question="Pick a project workspace for this run.",
                available=available,
                default=default,
            )
        )
