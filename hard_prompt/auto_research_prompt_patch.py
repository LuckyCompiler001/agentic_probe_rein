PROMPT_AUTO_RESEARCH_PATCH_PERFORMANCE_PROBE_IMPLEMENTATION_AND_INTEGRATION = """
You are an expert ML/DL software engineer. The user has chosen auto-research mode. There is no probe design document for you to read — you must decide the metric yourself based on the project, then implement and integrate a performance probe.

Your task:
1. Read the project (especially `train.py`) to understand what the model is trying to do.
2. Pick ONE common, standard performance metric that reflects how well training is going on this task. A second complementary metric is acceptable only if it adds clear value, but one is preferred. Use widely-accepted choices for the task type — for example: validation loss for general supervised training, RMSE / MAE for regression and forecasting, ROC-AUC / accuracy / F1 for classification, mAP for detection, perplexity for language modelling. Do not invent novel metrics.
3. Implement `prober.py` and integrate it into `train.py` exactly as specified below.

Step 1 — Read the codebase
Read `train.py` carefully. Understand:
- The task type (classification / regression / forecasting / detection / etc.)
- The training loop structure and where per-epoch (or per-validation-step) state is available
- What variables, objects, or hooks are accessible at each stage

Step 2 — Decide the metric and threshold
Based on standard practice for the identified task type, choose the metric and note its direction (higher-is-better or lower-is-better). Pick a principled, sane threshold that represents "training is working well enough" — use a reference value from common practice on this kind of dataset / task. If no clear public reference exists, pick a defensible target informed by the data (e.g. a value modestly better than a trivial baseline). Record the chosen threshold so it can be passed to `conclude(...)`.

Step 3 — Implement `prober.py`
Write a self-contained `prober.py` that exposes two entry points:

`def record(epoch, ...)` — called once per epoch (or per validation step) during training to capture the metric value for that step. The exact signature beyond `epoch` is yours to design based on what `train.py` can naturally pass in.

`def conclude(threshold)` — called once after training ends. It MUST do the following without raising:

  A. Compute statistics over all recorded values:
     - min, max, mean, std of the metric series
     - delta: final_value minus first_value (positive = increasing over training, negative = decreasing)
     - tail_mean: the mean of the last 5 recorded values, or all values if fewer than 5 exist
     - status: "PASS" if tail_mean satisfies the threshold condition (in the correct direction for the chosen metric), "FAIL" otherwise
     - conclusion: a one-sentence plain-English summary of what the probe found (e.g. "Validation RMSE fell from 0.81 to 0.42, comfortably under the 0.50 threshold by epoch 9.")

  B. Save the following JSON to `WORKING_SPACE/.agent_probe/metric/probe_result_N.json`, where N is the next available integer (find the highest existing `probe_result_*.json` in that directory and add 1, or start at 1 if none exist):
     {
         "metric_name": "string",
         "threshold": float,
         "values": [{"epoch": int, "value": float}, ...],
         "min": float,
         "max": float,
         "mean": float,
         "std": float,
         "first_value": float,
         "final_value": float,
         "delta": float,
         "tail_mean": float,
         "status": "PASS" | "FAIL",
         "conclusion": "string"
     }

  C. Create the metric directory if it does not exist. The save is mandatory — `conclude()` must not return without writing this file.

  Important: in this auto-research mode there is NO plot output. Do not generate any chart, PDF, image, or plot file. The probe writes only the metric JSON. Do not import plotly / matplotlib for plotting purposes.

Step 4 — Integrate into `train.py`
Modify `train.py` to:
- Import `record` and `conclude` from `prober.py`
- Call `record(epoch, ...)` inside the training loop at the appropriate validation point each epoch
- Call `conclude(threshold)` exactly once after the training loop ends, passing the threshold value chosen in Step 2
- Do not alter any training logic — only add the import and the two calls

Constraints:
- The metric must be a standard, widely-used performance metric appropriate for the task type
- Save metric output to the same `.agent_probe/metric/probe_result_N.json` location used by the rest of the pipeline — do not invent a new path
- Do not write a plot of any kind
- Do not modify files other than `prober.py` and the integration points in `train.py`
"""


PROMPT_AUTO_RESEARCH_PATCH_ITERATION_IMPROVEMENT = """
You are an expert ML/DL optimization engineer working in auto-research mode. `train.py` already contains 10 inline comments labeled `# potential_improvement_1:` through `# potential_improvement_10:` — each one marks a specific place where a targeted change could move the probe metric in the desired direction.

Your job for THIS iteration is simple and strict: pick exactly ONE of those comments and apply only the change it suggests. Leave everything else in the file untouched.

Step 0 — Pass status and revert check
Count the files in `.agent_probe/metric/`. Call that count N (e.g. 2 files → N = 2).
The snapshot of `train.py` taken just before this iteration is `.agent_probe/snapshot/train_version_{N}.py`.
Read `.agent_probe/metric/probe_result_{N}.json`. If its `status` field is "PASS", the metric has already satisfied the threshold — do NOT make any changes and stop immediately.
If N >= 2, also read `probe_result_{N-1}.json` and compare their `tail_mean` values (the mean of the last 5 recorded values, representing stable end-of-training behaviour rather than a single noisy point).
If the most recent `tail_mean` is WORSE than the previous one (higher for a lower-is-better metric, lower for a higher-is-better metric), the previous iteration's change hurt the metric — restore `train.py` from `.agent_probe/snapshot/train_version_{N-1}.py` BEFORE doing anything else this iteration.

Step 1 — Read the probe and the script
- Read `prober.py` to understand which metric is tracked and the direction in which improvement means.
- Read `train.py` carefully. Locate every `# potential_improvement_N:` comment; each one describes a candidate change and the code line / block it applies to.
- Read all existing `.agent_probe/change_log_*.txt` files. Note which `potential_improvement_*` items have already been actioned and whether they helped or hurt the metric. Do not repeat an item that has already been tried in the same direction and made things worse.

Step 2 — Pick exactly ONE comment to action this iteration
- Choose the `potential_improvement_*` item most likely to push the metric the right way given the per-epoch values in the latest probe result and the change history.
- The chosen item must not duplicate a previous unsuccessful attempt at the same change.
- You must modify ONLY the code referenced by that single comment (the line it sits on, or the immediately adjacent code block it clearly refers to).
- Leave the other 9 comments and their referenced code untouched. Do not rewrite, reformat, or refactor unrelated regions.
- Leave the `# potential_improvement_N:` comment line itself in place — do not delete or renumber the comments. You may append a short trailing note like `# applied` to the chosen comment if helpful, but it is not required.

Step 3 — Apply the change
Make exactly one targeted edit. Do not refactor, rename, or improve anything else. Specifically:
- Do not touch `prober.py` under any circumstances
- Do not modify the `record(...)` or `conclude(...)` call sites in `train.py`
- Do not modify the threshold value passed to `conclude(...)`

Step 4 — Write the change log
After making your change, write a plain-text summary to `.agent_probe/change_log_{N+1}.txt`. Include:
- Which `potential_improvement_*` item you chose (by number)
- The exact change you made (one or two sentences naming the parameter, value, or pattern)
- One sentence on why this change is expected to move the metric in the right direction
- If you reverted in Step 0, note that as well

Step 5 — Verify integration integrity
Before finishing, re-read the `record(...)` and `conclude(...)` call sites in `train.py` and confirm:
- Both calls are still present and unmodified
- The arguments passed to `record()` still resolve to existing variables with the correct types
- The threshold value passed to `conclude()` is unchanged

Make exactly one targeted change now — no more, no less.
"""
