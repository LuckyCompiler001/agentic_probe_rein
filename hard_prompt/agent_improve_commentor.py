PROMPT_SIX = """
You are an expert ML/DL code reviewer. Your job is to annotate a training script with improvement suggestions — but you must not change any code.

Step 1 — Understand the probe
Read `prober.py` to understand what aspect of training quality is being evaluated: what metric is tracked, what threshold is used, and what the probe considers healthy vs. problematic.

Step 2 — Review the training script
Read `train.py` carefully. With the probe's evaluation criteria in mind, identify exactly 10 places in the code that have meaningful room for improvement — spots where a change could positively move the probe's metric in the desired direction, or reduce the risk of the metric crossing into the problematic zone.

Step 3 — Add comments only
For each of the 10 places, insert an inline comment on the relevant line (or the line immediately above it if the line is too dense). Label them sequentially:

# potential_improvement_1: <concise explanation of what could be improved and why it matters for the probe metric>
# potential_improvement_2: ...
...
# potential_improvement_10: ...

Rules:
- Do not change any existing code — only add comment lines
- Each comment must be specific: name the technique, parameter, or pattern to change, and explain how it relates to the probe metric
- Spread the 10 comments across different parts of the file (data loading, model definition, optimizer, training loop, validation, etc.) — do not cluster them in one section
- Do not add comments to lines that are already obviously correct and leave no room for improvement

Modify only `train.py`. Do not touch `prober.py`.
"""
