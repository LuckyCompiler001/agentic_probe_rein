from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent

NLP_MODEL: Final = "opus"
AGENT_MODEL: Final = "opus"

# Each run targets exactly one subfolder of PROJECTS_BASE as its workspace.
# The choice is persisted as the `select_project` answer in progressbar.json.
PROJECTS_BASE: Final = PROJECT_ROOT / "project"
LEGACY_DEFAULT_PROJECT: Final = "mimic"

RUN_BASE: Final = PROJECT_ROOT / "response"

MAX_FIX_RETRIES: Final = 5


def list_projects() -> list[str]:
    """Folders inside PROJECTS_BASE that look like real workspaces."""
    if not PROJECTS_BASE.exists():
        return []
    return sorted(
        p.name
        for p in PROJECTS_BASE.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
