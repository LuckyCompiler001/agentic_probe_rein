from typing import Protocol


class Frontend(Protocol):
    """I/O abstraction for the pipeline.

    The terminal driver implements this with input()/print(); the web driver
    implements it with a queue fed by HTTP handlers. The pipeline never
    imports input() or print() directly — everything goes through Frontend.
    """

    def ask_yn(self, question: str, default: bool | None = None) -> bool: ...

    def ask_int_range(self, question: str, lo: int, hi: int) -> int: ...

    def ask_pos_int(self, question: str, default: int = 3) -> int: ...

    def ask_text(self, question: str) -> str: ...

    def show(self, message: str) -> None: ...

    def show_artifact(self, name: str, content: str) -> None: ...

    def select_run(self, existing: list[str]) -> str | None:
        """Return an existing run_id to resume, or None to start a new run."""
        ...

    def select_project(self, available: list[str], default: str | None = None) -> str:
        """Return the project folder name to use as the workspace for this run."""
        ...
