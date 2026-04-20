# Agentic Probe Rein

An agentic pipeline that automatically designs, implements, and iteratively improves evaluation probes for ML/DL training pipelines. Given a project description, it generates quantitative probes grounded in peer literature, implements them as runnable code, and uses AI agents to drive the training quality metric toward a target threshold.

---

## Workflow

```
User describes project
        │
        ▼
[GPT] Generate 10 probes          ← PROMPT_ONE  (probe angles from peer literature)
        │
        ▼
[GPT] Score probe confidence      ← PROMPT_TWO  (web-search source verification)
        │
        ▼
User selects a probe (1–10)
        │
        ▼
[GPT] Generate 3 dev plans        ← PROMPT_THREE (concrete implementation plans with metric + threshold)
        │
        ▼
[GPT] Score plan practicality     ← PROMPT_FOUR  (engineering feasibility review)
        │
        ▼
User selects a dev plan (1–3)
        │
        ▼
[Claude agent] Implement          ← PROMPT_FIVE  (writes prober.py, integrates into train.py)
        │
        ▼
[Run train.py] ──fail──► [Claude agent] Fix  ← PROMPT_EIGHT  (exception catcher, up to 5 retries)
        │ success
        ▼
(Optional) User enables auto-research feature
        │  yes
        ├──► [Claude agent] Annotate train.py  ← PROMPT_SIX  (10 targeted improvement comments)
        │           │
        │    [Run + fix if needed]
        │
        ▼
User sets iteration count (default 3)
        │
        ▼
┌──────────────────────────────────────┐
│  [Claude agent] Improve workspace    │  ← PROMPT_SEVEN (modifies anything except prober.py)
│  [Run train.py + fix if needed]      │  ← PROMPT_EIGHT
│  repeat N times                      │
└──────────────────────────────────────┘
        │
        ▼
User chooses: try another probe or exit
```

### What each probe run produces

After each successful `train.py` run, the probe writes two artifacts into the workspace:

| Artifact | Path | Contents |
|---|---|---|
| Metric data | `WROKING_SPACE/.agent_probe/metric/probe_result.json` | metric name, per-epoch values, min/max/mean/std, delta, trend, threshold, status, conclusion |
| Plot | `WROKING_SPACE/.agent_probe/plot/probe_result.html` | Plotly line chart of metric over epochs with threshold line, stats box, and pass/fail coloring |

---

## Project Structure

```
agentic_probe_rein/
├── main.py                   # Entry point and full workflow orchestration
├── Questions.py              # All user-facing prompt strings
├── hard_prompt/              # System prompts for each agent
│   ├── nlp_prober_gen.py         # PROMPT_ONE  — probe generation
│   ├── nlp_prober_confi_comput.py # PROMPT_TWO  — probe confidence scoring
│   ├── nlp_dev_doc_gen.py        # PROMPT_THREE — dev plan generation
│   ├── nlp_dd_confi_comput.py    # PROMPT_FOUR  — dev plan confidence scoring
│   ├── agent_dd_implement.py     # PROMPT_FIVE  — probe implementation agent
│   ├── agent_improve_commentor.py # PROMPT_SIX  — code annotation agent
│   ├── agent_iterat_improver.py  # PROMPT_SEVEN — iterative improvement agent
│   └── agent_exception_catcher.py # PROMPT_EIGHT — crash fix agent
├── response/                 # Intermediate JSON outputs (auto-created)
│   ├── probe_designs.json
│   ├── probe_confidenced.json
│   ├── dev_doc.json
│   └── dev_doc_confidenced.json
└── dummy_project/            # Example target workspace
    ├── train.py
    └── data_process.py
```

---

## Setup

### 1. Dependencies

```bash
pip install openai
```

The Claude agent actions use the **Claude Code CLI** (`claude`), not the Python SDK. Install it separately:

```bash
npm install -g @anthropic-ai/claude-code
```

### 2. Environment variables

```bash
export OPENAI_API_KEY=sk-...        # for GPT NLP calls (PROMPT_ONE through FOUR)
export ANTHROPIC_API_KEY=sk-ant-... # for Claude Code CLI (PROMPT_FIVE through EIGHT)
```

### 3. Plug in your project

Set `WROKING_SPACE` in `main.py` to the absolute path of your project directory:

```python
WROKING_SPACE: Final = "/path/to/your/project"
```

Your project directory must contain a `train.py` that:
- Runs a complete training loop when executed with `python train.py`
- Exits with code 0 on success

The Claude agent will read your existing code to understand the pipeline before making any changes.

### 4. Models

| Variable | Default | Used for |
|---|---|---|
| `NLP_MODEL` | `gpt-5.4` | Probe generation, confidence scoring, dev plan generation |
| `AGENT_MODEL` | `claude-opus-4-7` | Code implementation, annotation, iterative improvement, crash fixing |

---

## Running

```bash
python main.py
```

Follow the interactive prompts. The session is stateful — intermediate JSON results are saved to `response/` so you can inspect what each agent produced at every stage.

---

## Key Design Decisions

- **Probe confidence is filled by a separate supervisor agent** (PROMPT_TWO / PROMPT_FOUR), not the agent that generated the content. This prevents self-assessment bias.
- **Each NLP call is a fresh, isolated conversation** — no shared history between agents.
- **Each Claude CLI call spawns a new subprocess** — agents are fully independent and sequential.
- **`prober.py` is never touched by the improvement agent** — the probe definition is frozen; only the training pipeline is modified to improve the metric.
- **Exception catcher has a 5-retry cap** — prevents infinite loops on unfixable errors.
- **Auto-research (PROMPT_SIX) is opt-in, default off** — adds annotation overhead only when the user wants it.
