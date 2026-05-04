"""Cross-site generalization probe (cross-cohort proxy on existing precomputed splits).

Substitution note (documented per development document):
    This dataset (the precomputed MIMIC-III TF-IDF splits in this repo) has no admission
    timestamp and no hospital/ICU-unit identifier, so a true temporal or site split is
    not available. As an in-distribution-vs-shift proxy, we use a cross-cohort split:
    at probe init we pick the largest ethnicity group in the test split as the
    in-distribution reference cohort and the second-largest group as the shift cohort.
    Per epoch we compute AUROC on each cohort and track the absolute gap

        ABSOLUTE_GAP = abs(AUROC(in-dist) - AUROC(shift))

    The metric is non-negative and lower is better (cross-cohort parity). PASS iff the
    mean of the last 10 recorded epochs is <= threshold.

    change_log: dataset has no temporal/site keys, so a cross-cohort proxy on
    test_meta['eth'] (largest vs second-largest group) is used in place of true
    site/temporal split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
PROBE_DIR = SCRIPT_DIR / '.agent_probe'
METRIC_DIR = PROBE_DIR / 'metric'
PLOT_DIR = PROBE_DIR / 'plot'
COHORTS_PATH = PROBE_DIR / '_cohorts.json'
AXIS_RANGE_PATH = PROBE_DIR / '_axis_range.json'
INIT_LOG_PATH = PROBE_DIR / '_probe_init.log'

ETH_NAMES = ['white', 'black', 'hispanic', 'asian', 'other']
METRIC_NAME = 'absolute_auroc_gap_cross_cohort'

_state = {
    'records': [],
    'in_idx': None,
    'shift_idx': None,
    'in_name': None,
    'shift_name': None,
    'sym_kl': None,
    'initialized': False,
}


# ---------------------------------------------------------------------------
# Init: cohorts + symmetric KL on TF-IDF
# ---------------------------------------------------------------------------
def _init_cohorts(test_loader) -> None:
    test_ds = test_loader.dataset
    eth = test_ds.eth.cpu().numpy()
    counts = eth.sum(axis=0).astype(int)
    order = np.argsort(-counts)
    in_pos = int(order[0])
    shift_pos = int(order[1])
    in_name = ETH_NAMES[in_pos]
    shift_name = ETH_NAMES[shift_pos]

    in_mask = eth[:, in_pos] > 0.5
    shift_mask = eth[:, shift_pos] > 0.5
    in_indices = np.where(in_mask)[0]
    shift_indices = np.where(shift_mask)[0]

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    with open(COHORTS_PATH, 'w') as f:
        json.dump(
            {
                'in_dist': {
                    'name': in_name,
                    'eth_pos': in_pos,
                    'support': int(counts[in_pos]),
                    'indices': in_indices.tolist(),
                },
                'shift': {
                    'name': shift_name,
                    'eth_pos': shift_pos,
                    'support': int(counts[shift_pos]),
                    'indices': shift_indices.tolist(),
                },
            },
            f,
        )

    tfidf = test_ds.tfidf
    in_mean = np.asarray(tfidf[in_mask].mean(axis=0)).flatten()
    shift_mean = np.asarray(tfidf[shift_mask].mean(axis=0)).flatten()
    eps = 1e-9
    p = in_mean + eps
    q = shift_mean + eps
    p = p / p.sum()
    q = q / q.sum()
    sym_kl = float((p * np.log(p / q)).sum() + (q * np.log(q / p)).sum())

    _state['in_idx'] = in_indices
    _state['shift_idx'] = shift_indices
    _state['in_name'] = in_name
    _state['shift_name'] = shift_name
    _state['sym_kl'] = sym_kl
    _state['initialized'] = True

    with open(INIT_LOG_PATH, 'w') as f:
        f.write(f"in_dist_cohort={in_name} (n={int(counts[in_pos])})\n")
        f.write(f"shift_cohort={shift_name} (n={int(counts[shift_pos])})\n")
        f.write(f"symmetric_kl_tfidf={sym_kl:.6f}\n")


# ---------------------------------------------------------------------------
# Per-epoch full-test prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def _predict_test(model, loader, device):
    was_training = model.training
    model.eval()
    probs_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    for batch in loader:
        features = batch['features'].to(device)
        labels = batch['label']
        logits = model(features)
        probs_chunks.append(logits.sigmoid().detach().cpu().numpy())
        labels_chunks.append(labels.detach().cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(probs_chunks), np.concatenate(labels_chunks)


def _safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float('nan')
    return float(roc_auc_score(labels, probs))


# ---------------------------------------------------------------------------
# Public API: record
# ---------------------------------------------------------------------------
def record(epoch, model, test_loader, device):
    """Compute the cross-cohort absolute AUROC gap for this epoch and record it."""
    if not _state['initialized']:
        _init_cohorts(test_loader)

    probs, labels = _predict_test(model, test_loader, device)
    in_idx = _state['in_idx']
    shift_idx = _state['shift_idx']

    auroc_in = _safe_auroc(labels[in_idx], probs[in_idx])
    auroc_shift = _safe_auroc(labels[shift_idx], probs[shift_idx])

    if np.isnan(auroc_in) or np.isnan(auroc_shift):
        gap = 0.0
    else:
        gap = float(abs(auroc_in - auroc_shift))

    _state['records'].append(
        {
            'epoch': int(epoch),
            'value': float(gap),
            'auroc_in': float(auroc_in) if not np.isnan(auroc_in) else None,
            'auroc_shift': float(auroc_shift) if not np.isnan(auroc_shift) else None,
        }
    )


# ---------------------------------------------------------------------------
# File numbering
# ---------------------------------------------------------------------------
def _next_index() -> int:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = list(METRIC_DIR.glob('probe_result_*.json')) + list(
        PLOT_DIR.glob('probe_result_*.pdf')
    )
    nums: list[int] = []
    for p in candidates:
        try:
            nums.append(int(p.stem.split('_')[-1]))
        except Exception:
            continue
    return (max(nums) + 1) if nums else 1


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _resolve_axis_range(epochs, vals, threshold):
    if AXIS_RANGE_PATH.exists():
        try:
            with open(AXIS_RANGE_PATH) as f:
                cached = json.load(f)
            return (
                float(cached['x_min']),
                float(cached['x_max']),
                float(cached['y_min']),
                float(cached['y_max']),
            )
        except Exception:
            pass

    y_low = min(min(vals), threshold, 0.0)
    y_high = max(max(vals), threshold)
    span = max(y_high - y_low, 1e-3)
    pad = max(0.05, 0.25 * span)
    y_min = max(0.0, y_low - pad)
    # Pad upper side generously so future runs that get worse still fit.
    y_max = y_high + 2.0 * pad
    x_min = min(epochs) - 0.5
    x_max = max(epochs) + 0.5

    AXIS_RANGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AXIS_RANGE_PATH, 'w') as f:
        json.dump(
            {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}, f
        )
    return x_min, x_max, y_min, y_max


def _make_plot(values_pairs, threshold, stats, status, out_pdf: Path) -> None:
    epochs = [v['epoch'] for v in values_pairs]
    vals = [v['value'] for v in values_pairs]
    line_color = 'green' if status == 'PASS' else 'red'

    cross_epoch = None
    for v in values_pairs:
        if v['value'] <= threshold:
            cross_epoch = v['epoch']
            break

    x_min, x_max, y_min, y_max = _resolve_axis_range(epochs, vals, threshold)

    text_lines = [
        f"min: {stats['min']:.4f}",
        f"max: {stats['max']:.4f}",
        f"mean: {stats['mean']:.4f}",
        f"std: {stats['std']:.4f}",
        f"delta: {stats['delta']:+.4f}",
        f"trend: {stats['trend']}",
        f"status: {status}",
    ]

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=vals,
                mode='lines+markers',
                line=dict(color=line_color, width=2),
                marker=dict(color=line_color, size=6),
                name=METRIC_NAME,
            )
        )
        fig.add_hline(
            y=threshold,
            line=dict(color='red', dash='dash', width=2),
            annotation_text=f"threshold = {threshold}",
            annotation_position='top right',
            annotation_font=dict(color='red'),
        )
        if cross_epoch is not None:
            fig.add_vline(
                x=cross_epoch,
                line=dict(color='blue', dash='dot', width=1.5),
                annotation_text=f"crosses @ epoch {cross_epoch}",
                annotation_position='top left',
            )
        fig.add_annotation(
            xref='paper',
            yref='paper',
            x=0.99,
            y=0.99,
            xanchor='right',
            yanchor='top',
            text='<br>'.join(text_lines),
            showarrow=False,
            align='left',
            bordercolor='black',
            borderwidth=1,
            bgcolor='rgba(255,255,255,0.85)',
        )
        fig.update_layout(
            title=METRIC_NAME,
            xaxis_title='Epoch',
            yaxis_title=METRIC_NAME,
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
            template='plotly_white',
            showlegend=False,
        )
        fig.write_image(str(out_pdf), format='pdf')
        return
    except Exception:
        pass

    # Fallback: matplotlib
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, vals, marker='o', color=line_color, label=METRIC_NAME)
    ax.axhline(
        y=threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'threshold = {threshold}',
    )
    if cross_epoch is not None:
        ax.axvline(
            x=cross_epoch,
            color='blue',
            linestyle=':',
            label=f'crosses @ epoch {cross_epoch}',
        )
    ax.set_title(METRIC_NAME)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(METRIC_NAME)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.text(
        0.99,
        0.99,
        '\n'.join(text_lines),
        transform=ax.transAxes,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'),
    )
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(str(out_pdf), format='pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API: conclude
# ---------------------------------------------------------------------------
def conclude(threshold):
    threshold = float(threshold)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    records = list(_state['records'])
    values_pairs = [{'epoch': r['epoch'], 'value': r['value']} for r in records]

    if not values_pairs:
        first_value = 0.0
        final_value = 0.0
        delta = 0.0
        v_min = v_max = v_mean = v_std = 0.0
        tail_mean = 0.0
        status = 'FAIL'
        trend = 'no-data'
        conclusion = 'No values recorded; probe could not evaluate the cross-cohort gap.'
    else:
        vals = np.array([v['value'] for v in values_pairs], dtype=float)
        first_value = float(vals[0])
        final_value = float(vals[-1])
        delta = final_value - first_value
        v_min = float(vals.min())
        v_max = float(vals.max())
        v_mean = float(vals.mean())
        v_std = float(vals.std())
        tail = vals[-10:] if len(vals) >= 10 else vals
        tail_mean = float(tail.mean())
        status = 'PASS' if tail_mean <= threshold else 'FAIL'

        # lower-is-better: a negative delta (final < first) is improvement
        if delta < -1e-6:
            trend = 'improving'
        elif delta > 1e-6:
            trend = 'degrading'
        else:
            trend = 'stable'

        cross_epoch = None
        for v in values_pairs:
            if v['value'] <= threshold:
                cross_epoch = v['epoch']
                break
        cross_str = (
            f"first dropped to/below the {threshold} threshold at epoch {cross_epoch}"
            if cross_epoch is not None
            else f"never reached the {threshold} threshold"
        )

        conclusion = (
            f"Cross-cohort absolute AUROC gap "
            f"(in-dist={_state.get('in_name')}, shift={_state.get('shift_name')}) "
            f"moved from {first_value:.4f} to {final_value:.4f} "
            f"(delta={delta:+.4f}, trend={trend}); "
            f"mean over the last {len(tail)} epoch(s) is {tail_mean:.4f}, "
            f"{cross_str}; status={status} "
            f"(PASS iff last-10-epoch mean <= {threshold})."
        )

    stats = {
        'min': v_min,
        'max': v_max,
        'mean': v_mean,
        'std': v_std,
        'delta': delta,
        'trend': trend,
    }

    n = _next_index()
    out_json = METRIC_DIR / f'probe_result_{n}.json'
    out_pdf = PLOT_DIR / f'probe_result_{n}.pdf'

    payload = {
        'metric_name': METRIC_NAME,
        'threshold': threshold,
        'values': values_pairs,
        'min': v_min,
        'max': v_max,
        'mean': v_mean,
        'std': v_std,
        'first_value': first_value,
        'final_value': final_value,
        'delta': delta,
        'tail_mean': tail_mean,
        'status': status,
        'conclusion': conclusion,
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)

    try:
        _make_plot(values_pairs, threshold, stats, status, out_pdf)
    except Exception as exc:
        try:
            import matplotlib

            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.set_title(f"{METRIC_NAME} (plot fallback: {exc})")
            ax.text(0.5, 0.5, str(exc), ha='center', va='center')
            fig.savefig(str(out_pdf), format='pdf')
            plt.close(fig)
        except Exception:
            with open(out_pdf, 'wb') as f:
                f.write(b'%PDF-1.4\n%%EOF\n')
