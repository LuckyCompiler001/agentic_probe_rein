from hard_prompt.nlp_prober_gen import PROMPT_ONE
from hard_prompt.nlp_prober_confi_comput import PROMPT_TWO
from hard_prompt.nlp_dev_doc_gen import PROMPT_THREE
from hard_prompt.nlp_dd_confi_comput import PROMPT_FOUR
from hard_prompt.agent_dd_implement import PROMPT_FIVE
from hard_prompt.agent_improve_commentor import PROMPT_SIX
from hard_prompt.agent_iterat_improver import PROMPT_SEVEN
from hard_prompt.agent_exception_catcher import PROMPT_EIGHT
from hard_prompt.auto_research_prompt_patch import (
    PROMPT_AUTO_RESEARCH_PATCH_PERFORMANCE_PROBE_IMPLEMENTATION_AND_INTEGRATION,
    PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT,
)
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Final

from Questions import (
    QUESTION_ZERO,
    QUESTION_ONE,
    QUESTION_TWO,
    QUESTION_THREE,
    QUESTION_FOUR,
    QUESTION_FIVE,
    QUESTION_SIX,
)

# config
NLP_MODEL = "opus"
AGENT_MODEL = "opus"
WROKING_SPACE: Final = "/home/xuanhe_linux_001/agentic_probe_rein/CelebFaces_Attributes_Classification"
RUN_BASE = Path(__file__).parent / "response"

# updated per run inside main()
RESPONSE_DIR: Path = RUN_BASE


# ── Progressbar ───────────────────────────────────────────────────────────────

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


# ── Input helpers ─────────────────────────────────────────────────────────────

def get_input_placeholder(text: str) -> str:
    print(f"\n{text}")
    return input(">> ")


def _ask_yn(text: str, default: bool | None = None) -> bool:
    while True:
        raw = get_input_placeholder(text).strip().upper()
        if raw == "" and default is not None:
            return default
        if raw in ("Y", "N"):
            return raw == "Y"
        print("  Invalid input — please enter Y or N.")


def _ask_int_range(text: str, lo: int, hi: int) -> int:
    while True:
        raw = get_input_placeholder(text).strip()
        if raw.isdigit():
            value = int(raw)
            if lo <= value <= hi:
                return value
        print(f"  Invalid input — please enter a number between {lo} and {hi}.")


def _ask_pos_int(text: str, default: int = 3) -> int:
    while True:
        raw = get_input_placeholder(text).strip()
        if raw == "":
            return default
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print(f"  Invalid input — please enter a positive integer, or press Enter for default ({default}).")


# ── State ─────────────────────────────────────────────────────────────────────

USER_ANSWER_ONE: str = ""
USER_ANSWER_TWO: int = 0
USER_ANSWER_THREE: int = 0
USER_ANSWER_FOUR: int = 3

probe_template = """
{
    "probe_designs": [
        { "probe_type": "string", "probe_name": "string", \
            "content": "string", "possible_sources": ["string"], \
                "confidence": "float between 0 and 1" },
        { "probe_type": "string", "probe_name": "string", \
            "content": "string", "possible_sources": ["string"], \
                "confidence": "float between 0 and 1" },
        ...
    ]
}"""

dev_plan_template = """
{
    "dev_plans": [
        { "content": "string", "metric": "string", "threshold": "string", "confidence": 0.0 },
        { "content": "string", "metric": "string", "threshold": "string", "confidence": 0.0 },
        { "content": "string", "metric": "string", "threshold": "string", "confidence": 0.0 }
    ]
}
"""


# ── Claude helpers ─────────────────────────────────────────────────────────────

def _nlp_call(message: str) -> dict:
    result = subprocess.run(
        ["claude", "-p", "--model", NLP_MODEL, "--tools", "", "--no-session-persistence", message],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _agent_call(prompt: str) -> None:
    subprocess.run(
        ["claude", "-p", "--dangerously-skip-permissions", "--model", AGENT_MODEL, prompt],
        cwd=WROKING_SPACE,
        check=True,
    )


def _probe_passed() -> bool:
    metric_dir = Path(WROKING_SPACE) / ".agent_probe" / "metric"
    if not metric_dir.exists():
        return False
    nums = []
    for p in metric_dir.glob("probe_result_*.json"):
        try:
            nums.append(int(p.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    if not nums:
        return False
    latest = metric_dir / f"probe_result_{max(nums)}.json"
    try:
        data = json.loads(latest.read_text())
        return data.get("status") == "PASS"
    except Exception:
        return False


# ── Actions ───────────────────────────────────────────────────────────────────

def action_1_probe_generation_from_context():
    RESPONSE_DIR.mkdir(exist_ok=True)
    result = _nlp_call(f"{PROMPT_ONE}\n\n{USER_ANSWER_ONE}")
    out = RESPONSE_DIR / "probe_designs.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


def action_1a_probe_confidence():
    RESPONSE_DIR.mkdir(exist_ok=True)
    probe_designs = (RESPONSE_DIR / "probe_designs.json").read_text()
    result = _nlp_call(f"{PROMPT_TWO}\n\n{probe_designs}")
    out = RESPONSE_DIR / "probe_confidenced.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


def action_2_dev_doc_generation_from_probe():
    RESPONSE_DIR.mkdir(exist_ok=True)
    probe_data = json.loads((RESPONSE_DIR / "probe_confidenced.json").read_text())
    selected = probe_data["probe_designs"][USER_ANSWER_TWO - 1]
    result = _nlp_call(f"{PROMPT_THREE}\n\n{json.dumps(selected, indent=2)}")
    out = RESPONSE_DIR / "dev_doc.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


def action_2a_dev_doc_confidence():
    RESPONSE_DIR.mkdir(exist_ok=True)
    dev_doc = (RESPONSE_DIR / "dev_doc.json").read_text()
    result = _nlp_call(f"{PROMPT_FOUR}\n\n{dev_doc}")
    out = RESPONSE_DIR / "dev_doc_confidenced.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


def action_3_agent_implementation():
    dev_doc_data = json.loads((RESPONSE_DIR / "dev_doc_confidenced.json").read_text())
    selected = dev_doc_data["dev_plans"][USER_ANSWER_THREE - 1]
    _agent_call(f"{PROMPT_FIVE}\n\nWrite prober.py and integrate it into train.py.\n\n{selected}")


def action_4_agent_improvement():
    _agent_call(f"{PROMPT_SIX}\n\nTarget file: train.py")


def action_4_iterate():
    _agent_call(PROMPT_SEVEN)


def action_auto_research_probe_setup():
    _agent_call(PROMPT_AUTO_RESEARCH_PATCH_PERFORMANCE_PROBE_IMPLEMENTATION_AND_INTEGRATION)


def action_auto_research_iterate():
    _agent_call(PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT)


def action_run_training() -> tuple[bool, str]:
    result = subprocess.run(
        ["python", "train.py"],
        cwd=WROKING_SPACE,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stderr


MAX_FIX_RETRIES = 5


def _probe_artifact_nums(directory: Path, glob: str) -> set[int]:
    nums: set[int] = set()
    if not directory.exists():
        return nums
    for p in directory.glob(glob):
        try:
            nums.add(int(p.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return nums


def _purge_new_probe_artifacts(
    metric_dir: Path,
    plot_dir: Path,
    existing_nums: set[int],
) -> None:
    """Delete metric JSON and plot PDF files produced by a failed run."""
    for p in metric_dir.glob("probe_result_*.json"):
        try:
            if int(p.stem.rsplit("_", 1)[-1]) not in existing_nums:
                p.unlink(missing_ok=True)
        except ValueError:
            continue
    for p in plot_dir.glob("probe_result_*.pdf"):
        try:
            if int(p.stem.rsplit("_", 1)[-1]) not in existing_nums:
                p.unlink(missing_ok=True)
        except ValueError:
            continue


def action_x_agentic_exception_catcher():
    metric_dir = Path(WROKING_SPACE) / ".agent_probe" / "metric"
    plot_dir = Path(WROKING_SPACE) / ".agent_probe" / "plot"
    existing_nums = _probe_artifact_nums(metric_dir, "probe_result_*.json")

    success, error = action_run_training()
    retries = 0
    while not success:
        if retries >= MAX_FIX_RETRIES:
            print(f"Could not fix after {MAX_FIX_RETRIES} attempts. Last error:\n{error}")
            raise RuntimeError("Exception catcher exceeded max retries.")
        retries += 1
        _purge_new_probe_artifacts(metric_dir, plot_dir, existing_nums)
        print(f"Error detected (attempt {retries}/{MAX_FIX_RETRIES}):\n{error}\nAsking agent to fix...")
        _agent_call(f"{PROMPT_EIGHT}\n\nError output:\n{error}")
        success, error = action_run_training()
    print("Training and probe ran successfully.")


# ── Run setup (new / resume) ──────────────────────────────────────────────────

def _setup_run() -> tuple[Path, Progressbar]:
    RUN_BASE.mkdir(exist_ok=True)
    existing = sorted(
        p.name for p in RUN_BASE.iterdir()
        if p.is_dir() and p.name.isdigit()
    )

    if existing:
        print(f"\nExisting runs: {', '.join(existing)}")
        if _ask_yn("Resume a previous run? (Y/N)"):
            while True:
                run_id = get_input_placeholder(
                    "Enter the run ID (the number shown above) to resume:"
                ).strip()
                run_dir = RUN_BASE / run_id
                if run_dir.is_dir():
                    pb = Progressbar(run_dir)
                    print(f"[Resume] Loaded run {run_id}.")
                    return run_dir, pb
                print(f"  Run '{run_id}' not found. Please enter one of: {', '.join(existing)}")

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = RUN_BASE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pb = Progressbar(run_dir)
    print(f"[New run] Started run {run_id}.")
    return run_dir, pb


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global RESPONSE_DIR

    run_dir, pb = _setup_run()
    RESPONSE_DIR = run_dir

    # ── Step 1: confirm setup ──────────────────────────────────────────────────
    if not pb.is_done("setup_confirm"):
        answer = _ask_yn(QUESTION_ZERO)
        pb.mark("setup_confirm", answer)
    else:
        answer = pb.get_answer("setup_confirm")
        print("[Resume] Setup already confirmed.")

    if not answer:
        print("Please complete the setup before proceeding. Exiting.")
        return

    # Save the original train.py before any agent touches it.
    snapshot_dir = Path(WROKING_SPACE) / ".agent_probe" / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_v0 = snapshot_dir / "train_version_0.py"
    if not snapshot_v0.exists():
        snapshot_v0.write_text((Path(WROKING_SPACE) / "train.py").read_text())

    # ── Step 2: auto-research feature choice (one upfront switch) ──────────────
    if not pb.is_done("auto_research_choice"):
        use_auto_research = _ask_yn(QUESTION_SIX, default=False)
        pb.mark("auto_research_choice", use_auto_research)
    else:
        use_auto_research = pb.get_answer("auto_research_choice")
        print(f"[Resume] Auto-research: {'enabled' if use_auto_research else 'skipped'}.")

    if use_auto_research:
        _run_auto_research_pipeline(pb)
    else:
        _run_normal_pipeline(pb)


def _run_normal_pipeline(pb: Progressbar) -> None:
    global USER_ANSWER_ONE, USER_ANSWER_TWO, USER_ANSWER_THREE, USER_ANSWER_FOUR

    # ── Project context ───────────────────────────────────────────────────────
    if not pb.is_done("project_context"):
        USER_ANSWER_ONE = get_input_placeholder(QUESTION_ONE).strip()
        pb.mark("project_context", USER_ANSWER_ONE)
    else:
        USER_ANSWER_ONE = pb.get_answer("project_context")
        print("[Resume] Project context restored.")

    if not pb.is_done("probe_generation"):
        action_1_probe_generation_from_context()
        pb.mark("probe_generation")

    if not pb.is_done("probe_confidence"):
        action_1a_probe_confidence()
        pb.mark("probe_confidence")

    # ── Main loop (each outer iteration = one "cycle") ─────────────────────────
    cycle = 0
    while True:
        cp = f"cycle_{cycle}"

        # Show probes and let user select one
        if not pb.is_done(f"{cp}/probe_select"):
            print("\n" + (RESPONSE_DIR / "probe_confidenced.json").read_text())
            USER_ANSWER_TWO = _ask_int_range(QUESTION_TWO, lo=1, hi=10)
            pb.mark(f"{cp}/probe_select", USER_ANSWER_TWO)
        else:
            USER_ANSWER_TWO = pb.get_answer(f"{cp}/probe_select")
            print(f"[Resume] Probe {USER_ANSWER_TWO} already selected.")

        if not pb.is_done(f"{cp}/dev_doc_generation"):
            action_2_dev_doc_generation_from_probe()
            pb.mark(f"{cp}/dev_doc_generation")

        if not pb.is_done(f"{cp}/dev_doc_confidence"):
            action_2a_dev_doc_confidence()
            pb.mark(f"{cp}/dev_doc_confidence")

        # Show dev docs and let user select one
        if not pb.is_done(f"{cp}/plan_select"):
            print("\n" + (RESPONSE_DIR / "dev_doc_confidenced.json").read_text())
            USER_ANSWER_THREE = _ask_int_range(QUESTION_THREE, lo=1, hi=3)
            pb.mark(f"{cp}/plan_select", USER_ANSWER_THREE)
        else:
            USER_ANSWER_THREE = pb.get_answer(f"{cp}/plan_select")
            print(f"[Resume] Plan {USER_ANSWER_THREE} already selected.")

        prober_path = Path(WROKING_SPACE) / "prober.py"
        if not pb.is_done(f"{cp}/implementation") or not prober_path.exists():
            action_3_agent_implementation()
            pb.mark(f"{cp}/implementation")

        if not pb.is_done(f"{cp}/exception_check_1"):
            action_x_agentic_exception_catcher()
            pb.mark(f"{cp}/exception_check_1")

        # Iteration count
        if not pb.is_done(f"{cp}/iter_count"):
            USER_ANSWER_FOUR = _ask_pos_int(QUESTION_FOUR, default=3)
            pb.mark(f"{cp}/iter_count", USER_ANSWER_FOUR)
        else:
            USER_ANSWER_FOUR = pb.get_answer(f"{cp}/iter_count")
            print(f"[Resume] Iteration count: {USER_ANSWER_FOUR}.")

        agent_probe_dir = Path(WROKING_SPACE) / ".agent_probe"
        # change_log_1 is the baseline marker (no agent changes for the first run).
        change_log_1 = agent_probe_dir / "change_log_1.txt"
        change_log_1.parent.mkdir(parents=True, exist_ok=True)
        if not change_log_1.exists():
            change_log_1.write_text("")

        if _probe_passed():
            print("Probe already passed before iteration loop — skipping improvement iterations.")
        else:
            remaining = USER_ANSWER_FOUR
            iter_idx = 0
            while remaining > 0:
                ip = f"{cp}/iter_{iter_idx}"
                if not pb.is_done(f"{ip}/improve"):
                    # Snapshot train.py as train_version_{iter_idx+1} before the agent
                    # modifies it. Numbering is 1-based and aligns with probe_result_N.
                    train_src = Path(WROKING_SPACE) / "train.py"
                    snapshot_dst = agent_probe_dir / "snapshot" / f"train_version_{iter_idx + 1}.py"
                    snapshot_dst.parent.mkdir(parents=True, exist_ok=True)
                    snapshot_dst.write_text(train_src.read_text())
                    action_4_iterate()
                    pb.mark(f"{ip}/improve")
                if not pb.is_done(f"{ip}/exception_check"):
                    action_x_agentic_exception_catcher()
                    pb.mark(f"{ip}/exception_check")
                remaining -= 1
                iter_idx += 1
                if _probe_passed():
                    print("Probe status: PASS — stopping iterations early.")
                    break

        # Continue or exit
        if not pb.is_done(f"{cp}/continue_confirm"):
            do_continue = _ask_yn(QUESTION_FIVE)
            pb.mark(f"{cp}/continue_confirm", do_continue)
        else:
            do_continue = pb.get_answer(f"{cp}/continue_confirm")
            print(f"[Resume] Continue: {'yes' if do_continue else 'no'}.")

        if not do_continue:
            print("Goodbye!")
            return

        # Clean up before next probe cycle: remove prober.py and restore train.py
        prober_path = Path(WROKING_SPACE) / "prober.py"
        if prober_path.exists():
            prober_path.unlink()
            print("Removed prober.py.")
        snapshot_v0 = agent_probe_dir / "snapshot" / "train_version_0.py"
        if snapshot_v0.exists():
            train_py = Path(WROKING_SPACE) / "train.py"
            train_py.write_text(snapshot_v0.read_text())
            print("Reverted train.py to train_version_0.py.")
        else:
            print("Warning: train_version_0.py not found — train.py not reverted.")

        cycle += 1


def _run_auto_research_pipeline(pb: Progressbar) -> None:
    global USER_ANSWER_FOUR

    agent_probe_dir = Path(WROKING_SPACE) / ".agent_probe"
    prober_path = Path(WROKING_SPACE) / "prober.py"

    # Agent picks a standard performance metric, writes prober.py, integrates with train.py.
    if not pb.is_done("ar/probe_setup") or not prober_path.exists():
        action_auto_research_probe_setup()
        pb.mark("ar/probe_setup")

    # Run training to validate the integration and produce probe_result_1.
    if not pb.is_done("ar/exception_check_1"):
        action_x_agentic_exception_catcher()
        pb.mark("ar/exception_check_1")

    # Commentor adds 10 # potential_improvement_N: comments to train.py.
    if not pb.is_done("ar/comment"):
        action_4_agent_improvement()
        pb.mark("ar/comment")

    # Comments alone shouldn't break training, but keep parity with the normal flow.
    if not pb.is_done("ar/exception_check_2"):
        action_x_agentic_exception_catcher()
        pb.mark("ar/exception_check_2")

    # Iteration count
    if not pb.is_done("ar/iter_count"):
        USER_ANSWER_FOUR = _ask_pos_int(QUESTION_FOUR, default=3)
        pb.mark("ar/iter_count", USER_ANSWER_FOUR)
    else:
        USER_ANSWER_FOUR = pb.get_answer("ar/iter_count")
        print(f"[Resume] Iteration count: {USER_ANSWER_FOUR}.")

    # change_log_1 is the baseline marker (no agent changes for the first run).
    change_log_1 = agent_probe_dir / "change_log_1.txt"
    change_log_1.parent.mkdir(parents=True, exist_ok=True)
    if not change_log_1.exists():
        change_log_1.write_text("")

    if _probe_passed():
        print("Probe already passed before iteration loop — skipping improvement iterations.")
    else:
        remaining = USER_ANSWER_FOUR
        iter_idx = 0
        while remaining > 0:
            ip = f"ar/iter_{iter_idx}"
            if not pb.is_done(f"{ip}/improve"):
                # Snapshot train.py as train_version_{iter_idx+1} before the agent
                # modifies it. Numbering aligns with probe_result_N (same convention as normal mode).
                train_src = Path(WROKING_SPACE) / "train.py"
                snapshot_dst = agent_probe_dir / "snapshot" / f"train_version_{iter_idx + 1}.py"
                snapshot_dst.parent.mkdir(parents=True, exist_ok=True)
                snapshot_dst.write_text(train_src.read_text())
                action_auto_research_iterate()
                pb.mark(f"{ip}/improve")
            if not pb.is_done(f"{ip}/exception_check"):
                action_x_agentic_exception_catcher()
                pb.mark(f"{ip}/exception_check")
            remaining -= 1
            iter_idx += 1
            if _probe_passed():
                print("Probe status: PASS — stopping iterations early.")
                break

    print("Auto-research pipeline finished. Goodbye!")


if __name__ == "__main__":
    main()
