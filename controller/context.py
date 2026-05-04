from dataclasses import dataclass, field
from pathlib import Path

from controller.progressbar import Progressbar


@dataclass
class RunContext:
    """Per-run mutable state, threaded through the pipeline.

    `working_dir` is set once per run from the `select_project` answer.
    All workspace operations (agent_call cwd, .agent_probe artifacts,
    train.py, prober.py) resolve against it.
    """

    response_dir: Path
    progressbar: Progressbar
    working_dir: Path | None = None
    project_context: str = ""
    selected_probe_idx: int = 0
    selected_plan_idx: int = 0
    iteration_count: int = 3
