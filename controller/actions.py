import contextvars
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from hard_prompt.agent_dd_implement import PROMPT_FIVE
from hard_prompt.agent_exception_catcher import PROMPT_EIGHT
from hard_prompt.agent_improve_commentor import PROMPT_SIX
from hard_prompt.agent_iterat_improver import PROMPT_SEVEN
from hard_prompt.auto_research_prompt_patch import (
    PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT,
    PROMPT_AUTO_RESEARCH_PATCH_PERFORMANCE_PROBE_IMPLEMENTATION_AND_INTEGRATION,
)
from hard_prompt.nlp_dd_confi_comput import PROMPT_FOUR
from hard_prompt.nlp_dev_doc_gen import PROMPT_THREE
from hard_prompt.nlp_prober_confi_comput import PROMPT_TWO
from hard_prompt.nlp_prober_gen import PROMPT_ONE
from Questions import QUESTION_SEVEN, QUESTION_SEVEN_VALUE

from controller.config import AGENT_MODEL, MAX_FIX_RETRIES, NLP_MODEL
from controller.context import RunContext
from controller.frontend import Frontend


# ── Subprocess helpers ─────────────────────────────────────────────────────────

current_log_path: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "current_log_path", default=None
)


def _open_log() -> object | None:
    log_path = current_log_path.get()
    if log_path is None:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("a", buffering=1)
    f.write(f"\n--- {datetime.now().isoformat(timespec='seconds')} ---\n")
    return f


def _spawn_capture(cmd: list[str]) -> str:
    log_file = _open_log()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if log_file is not None:
            log_file.write(f"$ {' '.join(cmd[:3])} ...\n")
            if result.stderr:
                log_file.write("--- stderr ---\n")
                log_file.write(result.stderr)
            log_file.write("--- stdout ---\n")
            log_file.write(result.stdout)
        return result.stdout
    finally:
        if log_file is not None:
            log_file.close()


def _spawn_stream(cmd: list[str], cwd: Path | None = None) -> int:
    log_file = _open_log()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        bufsize=1,
    )
    try:
        if log_file is not None:
            log_file.write(f"$ {' '.join(cmd[:3])} ... (cwd={cwd})\n")
        assert proc.stdout is not None
        for line in proc.stdout:
            if log_file is not None:
                log_file.write(line)
            else:
                sys.stdout.write(line)
                sys.stdout.flush()
    finally:
        if log_file is not None:
            log_file.close()
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


# ── Claude helpers ─────────────────────────────────────────────────────────────

def nlp_call(message: str) -> dict:
    out = _spawn_capture(
        ["claude", "-p", "--model", NLP_MODEL, "--tools", "",
         "--no-session-persistence", message]
    )
    return json.loads(out)


def agent_call(prompt: str, cwd: Path) -> None:
    _spawn_stream(
        ["claude", "-p", "--dangerously-skip-permissions",
         "--model", AGENT_MODEL, prompt],
        cwd=cwd,
    )


# ── Probe artifact inspection ──────────────────────────────────────────────────

def probe_passed(working_dir: Path) -> bool:
    metric_dir = working_dir / ".agent_probe" / "metric"
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


def probe_artifact_nums(directory: Path, glob: str) -> set[int]:
    nums: set[int] = set()
    if not directory.exists():
        return nums
    for p in directory.glob(glob):
        try:
            nums.add(int(p.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return nums


def purge_new_probe_artifacts(
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


# ── Pipeline actions ───────────────────────────────────────────────────────────
#
# Every action that touches the workspace takes the working_dir from `ctx`.

def _wd(ctx: RunContext) -> Path:
    if ctx.working_dir is None:
        raise RuntimeError("RunContext.working_dir is not set yet.")
    return ctx.working_dir


def action_1_probe_generation_from_context(ctx: RunContext) -> None:
    ctx.response_dir.mkdir(exist_ok=True)
    result = nlp_call(f"{PROMPT_ONE}\n\n{ctx.project_context}")
    out = ctx.response_dir / "probe_designs.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


def action_1a_probe_confidence(ctx: RunContext) -> None:
    ctx.response_dir.mkdir(exist_ok=True)
    probe_designs = (ctx.response_dir / "probe_designs.json").read_text()
    result = nlp_call(f"{PROMPT_TWO}\n\n{probe_designs}")
    out = ctx.response_dir / "probe_confidenced.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


def action_2_dev_doc_generation_from_probe(ctx: RunContext) -> None:
    ctx.response_dir.mkdir(exist_ok=True)
    probe_data = json.loads((ctx.response_dir / "probe_confidenced.json").read_text())
    selected = probe_data["probe_designs"][ctx.selected_probe_idx - 1]
    result = nlp_call(f"{PROMPT_THREE}\n\n{json.dumps(selected, indent=2)}")
    out = ctx.response_dir / "dev_doc.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


def action_2a_dev_doc_confidence(ctx: RunContext) -> None:
    ctx.response_dir.mkdir(exist_ok=True)
    dev_doc = (ctx.response_dir / "dev_doc.json").read_text()
    result = nlp_call(f"{PROMPT_FOUR}\n\n{dev_doc}")
    out = ctx.response_dir / "dev_doc_confidenced.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")


def action_3_agent_implementation(ctx: RunContext) -> None:
    dev_doc_data = json.loads((ctx.response_dir / "dev_doc_confidenced.json").read_text())
    selected = dev_doc_data["dev_plans"][ctx.selected_plan_idx - 1]
    agent_call(
        f"{PROMPT_FIVE}\n\nWrite prober.py and integrate it into train.py.\n\n{selected}",
        cwd=_wd(ctx),
    )


def action_4_agent_improvement(ctx: RunContext) -> None:
    agent_call(f"{PROMPT_SIX}\n\nTarget file: train.py", cwd=_wd(ctx))


def action_4_iterate(ctx: RunContext) -> None:
    agent_call(PROMPT_SEVEN, cwd=_wd(ctx))


def action_auto_research_probe_setup(ctx: RunContext) -> None:
    agent_call(
        PROMPT_AUTO_RESEARCH_PATCH_PERFORMANCE_PROBE_IMPLEMENTATION_AND_INTEGRATION,
        cwd=_wd(ctx),
    )


def action_auto_research_iterate(ctx: RunContext) -> None:
    agent_call(PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT, cwd=_wd(ctx))


def action_run_training(ctx: RunContext) -> tuple[bool, str]:
    log_file = _open_log()
    try:
        result = subprocess.run(
            ["python", "train.py"],
            cwd=_wd(ctx),
            capture_output=True,
            text=True,
        )
        if log_file is not None:
            log_file.write(f"$ python train.py (cwd={_wd(ctx)})\n")
            if result.stdout:
                log_file.write("--- stdout ---\n")
                log_file.write(result.stdout)
            if result.stderr:
                log_file.write("--- stderr ---\n")
                log_file.write(result.stderr)
            log_file.write(f"--- exit {result.returncode} ---\n")
        return result.returncode == 0, result.stderr
    finally:
        if log_file is not None:
            log_file.close()


def action_threshold_override(
    ctx: RunContext,
    frontend: Frontend,
    plan_idx: int | None = None,
) -> None:
    """Ask whether to manually override the probe threshold; if yes, propagate.

    Decides between code-only and agent-driven update based on whether prober.py
    already exists. When plan_idx is given, also rewrites the threshold field in
    dev_doc_confidenced.json so re-runs and resumes see the new value. Auto-
    research mode has no dev plan, so plan_idx is omitted there.
    """
    if not frontend.ask_yn(QUESTION_SEVEN, default=False):
        return

    new_threshold = frontend.ask_text(QUESTION_SEVEN_VALUE)
    if not new_threshold:
        print("  Empty threshold — keeping the original value.")
        return

    if plan_idx is not None:
        dev_doc_path = ctx.response_dir / "dev_doc_confidenced.json"
        if dev_doc_path.exists():
            data = json.loads(dev_doc_path.read_text())
            old = data["dev_plans"][plan_idx].get("threshold", "<unset>")
            data["dev_plans"][plan_idx]["threshold"] = new_threshold
            dev_doc_path.write_text(json.dumps(data, indent=2))
            print(f"  Updated dev_doc_confidenced.json threshold: {old!r} -> {new_threshold!r}")

    prober_path = _wd(ctx) / "prober.py"
    if prober_path.exists():
        print("  prober.py exists — using agent to propagate threshold to code + existing results.")
        agent_call(
            "The probe threshold has been manually overridden. The new threshold is:\n"
            f"    {new_threshold}\n\n"
            "Tasks (do all of them, in order):\n"
            "1. Open prober.py and update every reference to the threshold value to the new "
            "value above. Keep the metric, direction, and PASS/FAIL semantics unchanged — only "
            "the numerical / expression threshold changes.\n"
            "2. If prober.py imports helpers that also hold a copy of the threshold, update "
            "those too.\n"
            "3. For every existing file under .agent_probe/metric/probe_result_*.json, "
            "re-evaluate the 'status' field (PASS/FAIL) against the new threshold using the "
            "metric values already recorded, update the 'threshold' field to the new value, "
            "and rewrite the 'conclusion' string to reflect the re-evaluation. Do not change "
            "the 'values' arrays.\n"
            "4. Do not run training. Just save the file changes.",
            cwd=_wd(ctx),
        )
        print(f"  Agent finished propagating threshold {new_threshold!r}.")


def action_x_agentic_exception_catcher(ctx: RunContext) -> None:
    metric_dir = _wd(ctx) / ".agent_probe" / "metric"
    plot_dir = _wd(ctx) / ".agent_probe" / "plot"
    existing_nums = probe_artifact_nums(metric_dir, "probe_result_*.json")

    success, error = action_run_training(ctx)
    retries = 0
    while not success:
        if retries >= MAX_FIX_RETRIES:
            print(f"Could not fix after {MAX_FIX_RETRIES} attempts. Last error:\n{error}")
            raise RuntimeError("Exception catcher exceeded max retries.")
        retries += 1
        purge_new_probe_artifacts(metric_dir, plot_dir, existing_nums)
        print(f"Error detected (attempt {retries}/{MAX_FIX_RETRIES}):\n{error}\nAsking agent to fix...")
        agent_call(f"{PROMPT_EIGHT}\n\nError output:\n{error}", cwd=_wd(ctx))
        success, error = action_run_training(ctx)
    print("Training and probe ran successfully.")
