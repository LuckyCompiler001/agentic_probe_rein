from datetime import datetime
from pathlib import Path

from Questions import (
    QUESTION_FIVE,
    QUESTION_FOUR,
    QUESTION_ONE,
    QUESTION_SIX,
    QUESTION_THREE,
    QUESTION_TWO,
    QUESTION_ZERO,
)

from controller.actions import (
    action_1_probe_generation_from_context,
    action_1a_probe_confidence,
    action_2_dev_doc_generation_from_probe,
    action_2a_dev_doc_confidence,
    action_3_agent_implementation,
    action_4_agent_improvement,
    action_4_iterate,
    action_auto_research_iterate,
    action_auto_research_probe_setup,
    action_threshold_override,
    action_x_agentic_exception_catcher,
    probe_passed,
)
from controller.config import (
    LEGACY_DEFAULT_PROJECT,
    PROJECTS_BASE,
    RUN_BASE,
    list_projects,
)
from controller.context import RunContext
from controller.frontend import Frontend
from controller.progressbar import Progressbar


# ── Run setup ─────────────────────────────────────────────────────────────────

def setup_run(frontend: Frontend) -> RunContext:
    """Pick existing or new run, build a RunContext."""
    RUN_BASE.mkdir(exist_ok=True)
    existing = sorted(
        p.name for p in RUN_BASE.iterdir()
        if p.is_dir() and p.name.isdigit()
    )

    chosen = frontend.select_run(existing)
    if chosen is not None:
        run_dir = RUN_BASE / chosen
        pb = Progressbar(run_dir)
        print(f"[Resume] Loaded run {chosen}.")
        return RunContext(response_dir=run_dir, progressbar=pb)

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = RUN_BASE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pb = Progressbar(run_dir)
    print(f"[New run] Started run {run_id}.")
    return RunContext(response_dir=run_dir, progressbar=pb)


# ── Project resolution ───────────────────────────────────────────────────────

def _resolve_project(ctx: RunContext, frontend: Frontend) -> str:
    """Resolve the project name for the run. Asks once on a fresh run; reads
    from progressbar on resume; falls back to LEGACY_DEFAULT_PROJECT for old
    runs that pre-date the select_project step.
    """
    pb = ctx.progressbar
    if pb.is_done("select_project"):
        name = pb.get_answer("select_project")
        if isinstance(name, str) and name:
            return name

    available = list_projects()
    has_other_done = any(
        s.get("done") for n, s in pb._steps.items() if n != "select_project"
    )
    if has_other_done:
        # Legacy run: progressbar exists and has done steps but no select_project.
        legacy = LEGACY_DEFAULT_PROJECT
        pb.mark("select_project", legacy)
        print(f"[Legacy run] Defaulting to project: {legacy}")
        return legacy

    if not available:
        raise RuntimeError(
            f"No project folders found under {PROJECTS_BASE}. "
            "Add at least one workspace folder before starting a run."
        )

    default = LEGACY_DEFAULT_PROJECT if LEGACY_DEFAULT_PROJECT in available else available[0]
    name = frontend.select_project(available, default=default)
    pb.mark("select_project", name)
    return name


# ── Top-level orchestrator ────────────────────────────────────────────────────

def run_pipeline(ctx: RunContext, frontend: Frontend) -> None:
    pb = ctx.progressbar

    # ── Step 0: pick the project workspace ─────────────────────────────────────
    project_name = _resolve_project(ctx, frontend)
    ctx.working_dir = PROJECTS_BASE / project_name
    if not ctx.working_dir.is_dir():
        raise RuntimeError(f"Selected project folder does not exist: {ctx.working_dir}")
    print(f"Workspace: {ctx.working_dir}")

    # ── Step 1: confirm setup ──────────────────────────────────────────────────
    if not pb.is_done("setup_confirm"):
        answer = frontend.ask_yn(QUESTION_ZERO)
        pb.mark("setup_confirm", answer)
    else:
        answer = pb.get_answer("setup_confirm")
        print("[Resume] Setup already confirmed.")

    if not answer:
        print("Please complete the setup before proceeding. Exiting.")
        return

    # Save the original train.py before any agent touches it.
    snapshot_dir = ctx.working_dir / ".agent_probe" / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_v0 = snapshot_dir / "train_version_0.py"
    if not snapshot_v0.exists():
        train_py = ctx.working_dir / "train.py"
        if train_py.exists():
            snapshot_v0.write_text(train_py.read_text())

    # ── Step 2: auto-research feature choice (one upfront switch) ──────────────
    if not pb.is_done("auto_research_choice"):
        use_auto_research = frontend.ask_yn(QUESTION_SIX, default=False)
        pb.mark("auto_research_choice", use_auto_research)
    else:
        use_auto_research = pb.get_answer("auto_research_choice")
        print(f"[Resume] Auto-research: {'enabled' if use_auto_research else 'skipped'}.")

    if use_auto_research:
        _run_auto_research_pipeline(ctx, frontend)
    else:
        _run_normal_pipeline(ctx, frontend)


# ── Normal pipeline ───────────────────────────────────────────────────────────

def _run_normal_pipeline(ctx: RunContext, frontend: Frontend) -> None:
    pb = ctx.progressbar
    wd = ctx.working_dir
    assert wd is not None

    # ── Project context ───────────────────────────────────────────────────────
    if not pb.is_done("project_context"):
        ctx.project_context = frontend.ask_text(QUESTION_ONE)
        pb.mark("project_context", ctx.project_context)
    else:
        ctx.project_context = pb.get_answer("project_context")
        print("[Resume] Project context restored.")

    if not pb.is_done("probe_generation"):
        action_1_probe_generation_from_context(ctx)
        pb.mark("probe_generation")

    if not pb.is_done("probe_confidence"):
        action_1a_probe_confidence(ctx)
        pb.mark("probe_confidence")

    # ── Main loop (each outer iteration = one "cycle") ─────────────────────────
    cycle = 0
    while True:
        cp = f"cycle_{cycle}"

        if not pb.is_done(f"{cp}/probe_select"):
            frontend.show_artifact(
                "probe_confidenced.json",
                (ctx.response_dir / "probe_confidenced.json").read_text(),
            )
            ctx.selected_probe_idx = frontend.ask_int_range(QUESTION_TWO, lo=1, hi=10)
            pb.mark(f"{cp}/probe_select", ctx.selected_probe_idx)
        else:
            ctx.selected_probe_idx = pb.get_answer(f"{cp}/probe_select")
            print(f"[Resume] Probe {ctx.selected_probe_idx} already selected.")

        if not pb.is_done(f"{cp}/dev_doc_generation"):
            action_2_dev_doc_generation_from_probe(ctx)
            pb.mark(f"{cp}/dev_doc_generation")

        if not pb.is_done(f"{cp}/dev_doc_confidence"):
            action_2a_dev_doc_confidence(ctx)
            pb.mark(f"{cp}/dev_doc_confidence")

        if not pb.is_done(f"{cp}/plan_select"):
            frontend.show_artifact(
                "dev_doc_confidenced.json",
                (ctx.response_dir / "dev_doc_confidenced.json").read_text(),
            )
            ctx.selected_plan_idx = frontend.ask_int_range(QUESTION_THREE, lo=1, hi=3)
            pb.mark(f"{cp}/plan_select", ctx.selected_plan_idx)
        else:
            ctx.selected_plan_idx = pb.get_answer(f"{cp}/plan_select")
            print(f"[Resume] Plan {ctx.selected_plan_idx} already selected.")

        action_threshold_override(ctx, frontend, plan_idx=ctx.selected_plan_idx - 1)

        prober_path = wd / "prober.py"
        if not pb.is_done(f"{cp}/implementation") or not prober_path.exists():
            action_3_agent_implementation(ctx)
            pb.mark(f"{cp}/implementation")

        if not pb.is_done(f"{cp}/exception_check_1"):
            action_x_agentic_exception_catcher(ctx)
            pb.mark(f"{cp}/exception_check_1")

        if not pb.is_done(f"{cp}/iter_count"):
            ctx.iteration_count = frontend.ask_pos_int(QUESTION_FOUR, default=3)
            pb.mark(f"{cp}/iter_count", ctx.iteration_count)
        else:
            ctx.iteration_count = pb.get_answer(f"{cp}/iter_count")
            print(f"[Resume] Iteration count: {ctx.iteration_count}.")

        agent_probe_dir = wd / ".agent_probe"
        change_log_1 = agent_probe_dir / "change_log_1.txt"
        change_log_1.parent.mkdir(parents=True, exist_ok=True)
        if not change_log_1.exists():
            change_log_1.write_text("")

        if probe_passed(wd):
            print("Probe already passed before iteration loop — skipping improvement iterations.")
        else:
            remaining = ctx.iteration_count
            iter_idx = 0
            while remaining > 0:
                ip = f"{cp}/iter_{iter_idx}"
                if not pb.is_done(f"{ip}/improve"):
                    train_src = wd / "train.py"
                    snapshot_dst = agent_probe_dir / "snapshot" / f"train_version_{iter_idx + 1}.py"
                    snapshot_dst.parent.mkdir(parents=True, exist_ok=True)
                    snapshot_dst.write_text(train_src.read_text())
                    action_4_iterate(ctx)
                    pb.mark(f"{ip}/improve")
                if not pb.is_done(f"{ip}/exception_check"):
                    action_x_agentic_exception_catcher(ctx)
                    pb.mark(f"{ip}/exception_check")
                remaining -= 1
                iter_idx += 1
                if probe_passed(wd):
                    print("Probe status: PASS — stopping iterations early.")
                    break

        if not pb.is_done(f"{cp}/continue_confirm"):
            do_continue = frontend.ask_yn(QUESTION_FIVE)
            pb.mark(f"{cp}/continue_confirm", do_continue)
        else:
            do_continue = pb.get_answer(f"{cp}/continue_confirm")
            print(f"[Resume] Continue: {'yes' if do_continue else 'no'}.")

        if not do_continue:
            print("Goodbye!")
            return

        # Clean up before next probe cycle
        prober_path = wd / "prober.py"
        if prober_path.exists():
            prober_path.unlink()
            print("Removed prober.py.")
        snapshot_v0 = agent_probe_dir / "snapshot" / "train_version_0.py"
        if snapshot_v0.exists():
            train_py = wd / "train.py"
            train_py.write_text(snapshot_v0.read_text())
            print("Reverted train.py to train_version_0.py.")
        else:
            print("Warning: train_version_0.py not found — train.py not reverted.")

        cycle += 1


# ── Auto-research pipeline ────────────────────────────────────────────────────

def _run_auto_research_pipeline(ctx: RunContext, frontend: Frontend) -> None:
    pb = ctx.progressbar
    wd = ctx.working_dir
    assert wd is not None

    agent_probe_dir = wd / ".agent_probe"
    prober_path = wd / "prober.py"

    if not pb.is_done("ar/probe_setup") or not prober_path.exists():
        action_auto_research_probe_setup(ctx)
        pb.mark("ar/probe_setup")

    action_threshold_override(ctx, frontend)

    if not pb.is_done("ar/exception_check_1"):
        action_x_agentic_exception_catcher(ctx)
        pb.mark("ar/exception_check_1")

    if not pb.is_done("ar/comment"):
        action_4_agent_improvement(ctx)
        pb.mark("ar/comment")

    if not pb.is_done("ar/exception_check_2"):
        action_x_agentic_exception_catcher(ctx)
        pb.mark("ar/exception_check_2")

    if not pb.is_done("ar/iter_count"):
        ctx.iteration_count = frontend.ask_pos_int(QUESTION_FOUR, default=10)
        pb.mark("ar/iter_count", ctx.iteration_count)
    else:
        ctx.iteration_count = pb.get_answer("ar/iter_count")
        print(f"[Resume] Iteration count: {ctx.iteration_count}.")

    change_log_1 = agent_probe_dir / "change_log_1.txt"
    change_log_1.parent.mkdir(parents=True, exist_ok=True)
    if not change_log_1.exists():
        change_log_1.write_text("")

    remaining = ctx.iteration_count
    iter_idx = 0
    while remaining > 0:
        ip = f"ar/iter_{iter_idx}"
        if not pb.is_done(f"{ip}/improve"):
            train_src = wd / "train.py"
            snapshot_dst = agent_probe_dir / "snapshot" / f"train_version_{iter_idx + 1}.py"
            snapshot_dst.parent.mkdir(parents=True, exist_ok=True)
            snapshot_dst.write_text(train_src.read_text())
            action_auto_research_iterate(ctx)
            pb.mark(f"{ip}/improve")
        if not pb.is_done(f"{ip}/exception_check"):
            action_x_agentic_exception_catcher(ctx)
            pb.mark(f"{ip}/exception_check")
        remaining -= 1
        iter_idx += 1

    print("Auto-research pipeline finished. Goodbye!")
