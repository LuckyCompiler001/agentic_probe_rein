# Agentic Probe Rein

An agentic pipeline that automatically designs, implements, and iteratively improves evaluation probes for ML/DL training pipelines. Given a project description, it generates quantitative probes grounded in peer literature, implements them as runnable code, and uses AI agents to drive a training metric toward a target threshold.

There are **two ways to drive the pipeline**: a web UI and a terminal CLI. Both share the same controller code, so a session started in one can be inspected (or even resumed) by the other.

---

## Quick start

```bash
# 1. one-time setup
make install                  # installs Python deps + Node deps for the web UI

# 2. start everything
make dev                      # runs FastAPI on :8765 + Next.js on :3000
                              # Ctrl-C stops both
```

Then open **http://localhost:3000** in a browser.

If you only want the terminal experience, skip `make dev` and run:

```bash
venv/bin/python main.py
```

The first time you start a new run (web or terminal), the pipeline will ask you to **pick a project workspace** from the `project/` folder — that's the directory whose `train.py` the agents will modify. Drop your own project under `project/<your-name>/` to make it selectable.

---

## How it works

The system chains two types of AI calls:

- **NLP calls** — Claude (no tools, no session) for structured JSON generation: probe design, confidence scoring, and implementation planning.
- **Agent calls** — Claude Code CLI (full tools, full filesystem access) for code generation, code modification, crash fixing, and iterative improvement.

Each call is fully isolated: no shared history, no persistent context.

### Workflow

```
Pick a project workspace (from project/<…>)
        │
        ▼
Confirm setup, choose mode (normal vs auto-research)
        │
        ▼
[NLP] Generate 10 probes          ← PROMPT_ONE
[NLP] Score probe confidence      ← PROMPT_TWO
        │
User selects a probe (1–10)
        │
[NLP] Generate 3 dev plans        ← PROMPT_THREE
[NLP] Score plan practicality     ← PROMPT_FOUR
        │
User selects a plan (1–3)
        │
(Optional) Override the threshold for the chosen plan
        │
[Agent] Implement probe           ← PROMPT_FIVE
[Run train.py] ─crash─► [Agent] Fix  ← PROMPT_EIGHT (up to 5 retries)
        │
User sets iteration count (default: 3)
        │
┌─────────────────────────────────────────────────┐
│ Snapshot train.py                               │
│ [Agent] Improve training pipeline               │ ← PROMPT_SEVEN
│ [Run train.py + auto-fix]                       │ ← PROMPT_EIGHT
│ Probe status == PASS? → stop early              │
└─────────────────────────────────────────────────┘
        │
User: try another probe or exit
```

### What each probe run produces

Inside the chosen workspace (`project/<name>/.agent_probe/`):

| Artifact | Path | Contents |
|---|---|---|
| Metric JSON | `metric/probe_result_N.json` | metric name, per-epoch values, stats, threshold, status (PASS/FAIL), conclusion |
| Plot PDF | `plot/probe_result_N.pdf` | line chart with threshold reference and stats box |
| Train snapshot | `snapshot/train_version_N.py` | `train.py` before iteration N (for revert) |
| Change log | `change_log_N.txt` | summary of what the agent changed in iteration N |
| Axis cache | `_axis_range.json` | y-range cached so plots stay visually comparable across runs |

Per-run NLP outputs (probes, dev plans, progressbar) live in `response/<run_id>/`.

---

## Web UI tour

When you open **http://localhost:3000** you land on the **dashboard**:

- A list of every run in `response/`, with project name, mode, step count, and last-activity timestamp.
- A "Begin a new run" button that starts a fresh pipeline. The first thing it'll ask you (on the run page) is to pick a project workspace from `project/`.
- A "Session live" banner if a pipeline is currently running — only one can run at a time because the workspace is shared.

Clicking a run takes you to the **run page** (`/runs/<id>`):

| Region | What it does |
|---|---|
| Header | Run id, formatted date, mode badge, workspace badge, step counter, **Resume** button (or "Live" if a session is active here). |
| Stepper (left) | Vertical timeline of every progressbar step, grouped by Preamble / Cycle N / Auto-research. Solid black dot = done; ringed dot = pending; ghost dot = future. |
| Active prompt (top of right column) | Whatever the pipeline is asking for *right now* — Y/N, an integer, free text, or a project picker. Submit advances the pipeline. If no session is active, this card just says "Idle — press Resume". |
| Threshold override | Collapsible widget. Edits the dev plan's `threshold` field; if `prober.py` already exists, dispatches a background agent to propagate the new value into the code and re-evaluate existing `probe_result_*.json` PASS/FAIL. |
| Probe trajectory | Live Recharts line chart overlaying every iteration's per-epoch metric. The dashed red line is the threshold. Stat cards underneath show μ/σ/Δ and PASS/FAIL per iteration. |
| Probe designs editor | Monaco JSON editor on `probe_confidenced.json` — edit any probe's content/confidence/sources. Save writes the file; the pipeline picks up the new content next time it reads it. |
| Development plans editor | Monaco JSON editor on `dev_doc_confidenced.json` — same, for the 3 plan candidates. |
| Log tail | SSE-streamed view of `response/<run_id>/session.log` (subprocess stdout + agent output). Auto-scrolls; toggle "Follow" to pause. |

Editing JSON files is intentionally **cheap** — no cascading re-runs. If you rewrite a probe after the implementation step ran, the existing `prober.py` is still based on the old probe; that's by design. Re-run the relevant step manually if you want it regenerated.

---

## Terminal mode

Same code path, just driven from `stdin/stdout`:

```bash
venv/bin/python main.py
```

The terminal walks through the same questions (project pick, setup confirm, probe pick, threshold override, plan pick, iteration count, continue or exit). Pressing **Ctrl-C** mid-run prints a friendly resume hint with the run id; running `main.py` again offers to resume.

---

## Project structure

```
agentic_probe_rein/
├── main.py                       # Terminal entry point
├── Questions.py                  # All user-facing prompt strings
├── test.py                       # Smoke tests for NLP + agent reachability
├── Makefile                      # `make dev`, `make backend`, `make frontend`, `make install`
│
├── controller/                   # Pipeline state machine (shared by terminal + web)
│   ├── config.py                 #   PROJECTS_BASE, RUN_BASE, model names
│   ├── progressbar.py            #   resumable step tracker (progressbar.json)
│   ├── context.py                #   RunContext (working_dir, selections, …)
│   ├── frontend.py               #   I/O abstraction (Protocol)
│   ├── terminal.py               #   stdin/stdout driver
│   ├── actions.py                #   action_* + subprocess helpers (Popen + log tee)
│   └── pipeline.py               #   orchestrator (normal + auto-research modes)
│
├── server/                       # FastAPI backend
│   ├── app.py                    #   18 routes (runs, files, projects, session, SSE log…)
│   ├── session.py                #   one-at-a-time pipeline thread + per-thread stdout
│   └── web_frontend.py           #   queue-based Frontend impl
│
├── web/                          # Next.js 16 (App Router) frontend
│   ├── src/app/page.tsx          #   dashboard
│   ├── src/app/runs/[id]/page.tsx#   run detail
│   ├── src/components/           #   Stepper, ActivePrompt, JsonEditor, ThresholdWidget, ProbeChart, LogViewer, …
│   ├── src/lib/api.ts            #   typed client for the FastAPI surface
│   └── src/lib/steps.ts          #   parses progressbar steps into UI-friendly groups
│
├── hard_prompt/                  # System prompts (one file per agent)
│   ├── nlp_prober_gen.py         #   PROMPT_ONE   — generate 10 probe designs
│   ├── nlp_prober_confi_comput.py#   PROMPT_TWO   — score probe confidence
│   ├── nlp_dev_doc_gen.py        #   PROMPT_THREE — generate 3 implementation plans
│   ├── nlp_dd_confi_comput.py    #   PROMPT_FOUR  — score plan practicality
│   ├── agent_dd_implement.py     #   PROMPT_FIVE  — implement prober.py + integrate
│   ├── agent_improve_commentor.py#   PROMPT_SIX   — annotate train.py with 10 ideas
│   ├── agent_iterat_improver.py  #   PROMPT_SEVEN — iteratively improve train.py
│   ├── agent_exception_catcher.py#   PROMPT_EIGHT — fix crashed train.py
│   └── auto_research_prompt_patch.py # auto-research-mode patches for FIVE+SEVEN
│
├── project/                      # Candidate workspaces — pipeline targets one of these per run
│   ├── mimic/                    #   MIMIC-III mortality (TF-IDF + logistic regression)
│   ├── home_credit/              #   Home Credit Default Risk
│   ├── ieee_cis_fraud_detection/
│   ├── m5_forecast/
│   ├── rossmann/
│   ├── CelebFaces_Attributes_Classification/
│   ├── dummy_project/            #   minimal smoke-test target
│   └── user_project/             #   blank slot for your own project
│
└── response/                     # Run outputs (timestamped, auto-created)
    └── YYYYMMDDHHMMSS/
        ├── probe_designs.json
        ├── probe_confidenced.json
        ├── dev_doc.json
        ├── dev_doc_confidenced.json
        ├── progressbar.json      # resume state (steps + answers, including select_project)
        └── session.log           # captured stdout / agent output
```

---

## Setup

### 1. Python (3.10+)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install everything

```bash
make install
```

This:
- pip-installs `fastapi`, `uvicorn[standard]`, `python-multipart` (for the web backend);
- installs Node deps via `pnpm install` in `web/`.

The orchestration code itself only uses the standard library; the **target workspaces** under `project/` need their own ML dependencies — for the included `mimic/`:

```bash
pip install torch torchvision scipy scikit-learn numpy plotly kaleido matplotlib
```

For your own project, install whatever it needs.

> `kaleido` is the Plotly static image export backend. If unavailable, the probe falls back to matplotlib; if that also fails, it writes a placeholder PDF.

### 3. Node.js + Claude Code CLI

Node 18+ is required. Install Claude Code globally:

```bash
npm install -g @anthropic-ai/claude-code
claude --version
```

### 4. API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Both NLP and agent calls go through the Claude Code CLI and use this same key. Add it to your shell profile to persist.

### 5. Authenticate Claude Code

```bash
claude
# follow the OAuth flow, then Ctrl-C
```

### 6. Drop your project under `project/`

```
project/
└── my_project/
    └── train.py        # runs end-to-end with `python train.py`,
                        # exits 0 on success, prints traceback + nonzero on failure
```

The agent reads your existing code before making any changes. It needs `train.py` to be **self-contained and runnable**.

---

## Running

### Web UI + backend together

```bash
make dev
```

| Service | Port | What |
|---|---|---|
| FastAPI backend | 8765 | `server.app:app`, served by `uvicorn` |
| Next.js frontend | 3000 | `pnpm dev` (Turbopack) |

`Ctrl-C` stops both. Use `make backend` or `make frontend` to run them individually.

Production-style frontend (`next build && next start`) isn't wired into the Makefile yet; for personal/local use, `make dev` is the recommended path.

### Terminal only

```bash
venv/bin/python main.py
```

A run is **stateful**: each one creates a timestamped directory under `response/`. If you quit mid-run, re-running `main.py` lists existing runs and offers to resume. The same is true via the web UI.

### Smoke test (optional)

```bash
venv/bin/python test.py
```

All three checks should print `PASS`:

```
── NLP model (Claude, no tools) ────────
  PASS — got: {'status': 'ok', 'model': 'nlp'}
── Agent (Claude, full tools) ──────────
  PASS — got: 'PONG'
── Web search (NLP, CRWV stock price) ──
  PASS — CRWV price: …
```

### Interactive questions

| # | Where it appears | Question | Input |
|---|---|---|---|
| 0 | Run page (first time) | Pick a project workspace | one of `project/<name>` |
| 1 | Run page | Confirm dependencies installed | Y / N |
| 2 | Run page | Use auto-research mode? | Y / N (default N) |
| 3 | Run page | Describe the project + dataset | free text |
| 4 | Run page | Pick a probe | 1–10 |
| 5 | Run page | Pick a plan | 1–3 |
| 6 | Threshold widget *or* run page | Override threshold | optional |
| 7 | Run page | Iteration count | positive int (default 3 / 10) |
| 8 | Run page | Try another probe or exit | Y / N |

---

## Configuration

Common settings live in [controller/config.py](controller/config.py):

```python
NLP_MODEL   = "opus"   # generation + confidence scoring (--tools "" mode)
AGENT_MODEL = "opus"   # code gen + iteration (full tools mode)
PROJECTS_BASE = PROJECT_ROOT / "project"
RUN_BASE      = PROJECT_ROOT / "response"
MAX_FIX_RETRIES = 5
```

Any model id supported by Claude Code works — `"sonnet"`, `"haiku"`, full pinned ids like `"claude-opus-4-7"`.

The frontend's API base URL defaults to `http://localhost:8765` and can be overridden with `NEXT_PUBLIC_API_BASE` if you want to run the backend on a different host/port.

---

## Key design decisions

- **Per-run workspace** — every run picks one folder under `project/` and persists the choice in `progressbar.json` (`select_project` step). Resuming a legacy run with no such answer silently defaults to `mimic`.
- **Single shared workspace at a time** — only one pipeline can run at a time because every workspace owns the same `train.py` / `prober.py` / `.agent_probe/`. The backend enforces this with an in-memory lock.
- **Supervisor-scored confidence** — PROMPT_TWO and PROMPT_FOUR are separate agents from the generators. They fill in the confidence field independently, avoiding self-assessment bias.
- **Isolated calls** — every NLP call uses `--no-session-persistence`; every agent call is a new subprocess. No shared state between calls.
- **Frozen probe definition** — once `prober.py` is written (PROMPT_FIVE), the improvement agents (PROMPT_SIX/SEVEN) are instructed never to modify it. Only `train.py` and supporting files change.
- **Snapshot before each iteration** — `train.py` is saved to `.agent_probe/snapshot/train_version_N.py` before every agent modification, enabling manual revert.
- **Change log trail** — each iteration writes `change_log_N.txt` so the next iteration can see what was already tried and avoid repeating failed approaches.
- **Exception catcher cap** — up to 5 auto-fix retries per run. If all fail, the error is surfaced and execution halts.
- **Early stopping** — iterative improvement stops as soon as the latest `probe_result_N.json` reports `"status": "PASS"`, regardless of the configured iteration count.
- **Cheap edits** — editing JSON files in the web UI just rewrites the file; it does not retroactively re-run dependent steps. Re-run them manually if you want the changes to propagate.
- **Consistent chart axes** — the first probe run caches the y-range in `_axis_range.json`; subsequent runs reuse it so plots are visually comparable.
