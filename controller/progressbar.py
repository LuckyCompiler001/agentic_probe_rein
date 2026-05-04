import json
from pathlib import Path


class Progressbar:
    """Tracks completed steps and stores question answers for resume."""

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "progressbar.json"
        self._steps: dict[str, dict] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._steps = {s["name"]: s for s in data["steps"]}

    def is_done(self, name: str) -> bool:
        return self._steps.get(name, {}).get("done", False)

    def get_answer(self, name: str):
        return self._steps.get(name, {}).get("answer")

    def mark(self, name: str, answer=None) -> None:
        entry: dict = {"name": name, "done": True}
        if answer is not None:
            entry["answer"] = answer
        self._steps[name] = entry
        self._save()

    def _save(self) -> None:
        data = {"steps": list(self._steps.values())}
        self.path.write_text(json.dumps(data, indent=2))
