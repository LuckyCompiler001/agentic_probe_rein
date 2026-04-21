PROMPT_FIVE = """
You are an expert ML/DL software engineer. You will receive a development document describing a training-quality probe — including its implementation plan, the metric it computes, and the threshold that separates healthy from problematic.

Your task is to implement this probe by writing `prober.py` and integrating it into the existing `train.py` in this workspace.

Step 1 — Read the codebase
Read `train.py` to understand:
- The model architecture and training loop structure
- Where training data, validation data, and per-epoch state are available
- What variables, objects, or hooks you can access at each stage of the pipeline

Step 2 — Implement `prober.py`
Write a self-contained `prober.py` that exposes two entry points:

`def record(epoch, ...)` — called once per epoch during training to collect the metric value for that step.

`def conclude(threshold)` — called once after training completes. It must do the following with no exceptions:

  A. Compute statistics over all recorded values:
     - min, max, mean, std of the metric series
     - delta: final_value minus first_value (positive = improving toward higher-is-better threshold, negative = degrading)
     - trend: "improving", "degrading", or "stable" (stable if abs(delta) < 1% of the threshold)
     - status: "PASS" if final_value satisfies the threshold condition, "FAIL" otherwise
     - conclusion: a one-sentence plain-English summary of what the probe found (e.g. "Validation F1 improved steadily from 0.42 to 0.71, crossing the 0.65 threshold at epoch 14.")

  B. Save the following JSON to `WROKING_SPACE/.agent_probe/metric/probe_result_N.json`, where N is the next available integer (1, 2, 3, …) — i.e. find the highest existing probe_result_*.json in that directory and increment by 1, starting at 1 if none exist:
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
         "trend": "improving" | "degrading" | "stable",
         "status": "PASS" | "FAIL",
         "conclusion": "string"
     }

  C. Generate a Plotly line chart and save it as `WROKING_SPACE/.agent_probe/plot/probe_result_N.html`, using the same N as the JSON file above.
     The chart must include:
     - A labeled line for the metric values over epochs
     - A horizontal dashed line for the threshold, annotated with its value
     - A vertical annotation (or marker) at the epoch where the metric first crosses the threshold (if it does)
     - Chart title: the metric name
     - X-axis label: "Epoch"
     - Y-axis label: the metric name
     - A text box in the chart showing: min, max, mean, std, delta, trend, and status
     - Color the metric line green if status is PASS, red if FAIL

  D. Create both output directories if they do not exist. These saves are mandatory — conclude() must not return without writing both files.

Step 3 — Integrate into `train.py`
Modify `train.py` to:
- Import `record` and `conclude` from `prober.py`
- Call `record(epoch, ...)` inside the training loop at the appropriate point each epoch
- Call `conclude(threshold)` once after the training loop ends, passing the threshold value from the development document
- Do not alter the training logic — only add the import and the two calls

Development document:
"""
