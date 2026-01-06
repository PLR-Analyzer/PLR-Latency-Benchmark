# SPDX-FileCopyrightText: 2026 Marcel Schepelmann <schepelmann@chi.uni-hannover.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dataset metrics helpers for PLR visualizer.

Provides functions to compute per-file PLR metrics and summarize them across
multiple files. These are extracted from visualizer logic so the GUI file can
remain smaller.
"""

from pathlib import Path

import numpy as np


def compute_metrics_from_file(filepath):
    """Load an .npz file and compute PLR metrics.

    Returns a dict with keys: D_max, D_min, amplitude, t75, avg_constr_vel,
    max_constr_vel, dil_vel. Returns None on failure.
    """
    try:
        npz = np.load(str(filepath), allow_pickle=True)
        D_obs = np.asarray(npz["diameter_observed"])
        fps = float(npz.get("fps", 30.0))
        stim_time = float(npz.get("stim_time", 500.0))  # in ms
        led_duration = float(npz.get("led_duration", 167.0))  # in ms
        true_latency_abs = npz.get("true_latency", None)
        if true_latency_abs is None:
            true_latency = np.nan
        else:
            true_latency = float(true_latency_abs - stim_time)
    except Exception:
        return None

    dt = 1000.0 / fps  # dt in ms (since fps is samples per second)
    n = len(D_obs)
    t = np.linspace(0, dt * (n - 1), n)  # t in ms

    # Compute gradient in mm/ms, then convert to mm/s
    deriv = (
        np.gradient(D_obs, dt) * 1000.0
    )  # gradient gives mm/ms, multiply by 1000 to get mm/s

    # Baseline (before stimulus)
    baseline_mask = t < stim_time
    if baseline_mask.sum() == 0:
        D_baseline_mean = np.nan
    else:
        D_baseline_mean = np.mean(D_obs[baseline_mask])
    D_max = D_baseline_mean

    # Response region (after stimulus + small offset)
    response_start_idx = int((stim_time + 200.0) * fps / 1000.0)  # +200 ms offset
    response_start_idx = min(response_start_idx, len(D_obs) - 1)
    response_region = D_obs[response_start_idx:]
    D_min = np.min(response_region) if len(response_region) > 0 else np.nan

    # Amplitude (percent constriction)
    amplitude = ((D_max - D_min) / D_max * 100) if D_max > 0 else np.nan

    # 75% recovery time
    if np.isfinite(D_min) and D_max > D_min:
        D75 = D_min + 0.75 * (D_max - D_min)
        min_idx = response_start_idx + int(np.argmin(response_region))
        idx_recovery = np.where(D_obs[min_idx:] >= D75)[0]
        t75 = (idx_recovery[0] * dt) if len(idx_recovery) > 0 else np.nan
    else:
        t75 = np.nan

    # Constriction velocities (during response window)
    constr_mask = (t >= stim_time) & (
        t <= stim_time + led_duration + 1000.0
    )  # +1000 ms window
    constr_deriv = deriv[constr_mask]
    constr_neg = constr_deriv[constr_deriv < 0]
    avg_constr_vel = np.mean(constr_neg) if len(constr_neg) > 0 else np.nan
    max_constr_vel = np.min(constr_neg) if len(constr_neg) > 0 else np.nan

    # Dilation velocity (after minimum)
    if np.isfinite(D_min) and response_start_idx + 1 < len(D_obs):
        min_idx = response_start_idx + int(np.argmin(response_region))
        dil_window_end = min(
            min_idx + int(0.5 * fps), len(deriv)
        )  # 0.5 s = 500 ms window
        dil_deriv = deriv[min_idx:dil_window_end]
        dil_deriv_pos = dil_deriv[dil_deriv > 0]
        dil_vel = np.mean(dil_deriv_pos) if len(dil_deriv_pos) > 0 else np.nan
    else:
        dil_vel = np.nan

    return {
        "D_max": float(D_max) if np.isfinite(D_max) else np.nan,
        "D_min": float(D_min) if np.isfinite(D_min) else np.nan,
        "amplitude": float(amplitude) if np.isfinite(amplitude) else np.nan,
        "latency": float(true_latency) if np.isfinite(true_latency) else np.nan,
        "t75": float(t75) if np.isfinite(t75) else np.nan,
        "avg_constr_vel": (
            float(avg_constr_vel) if np.isfinite(avg_constr_vel) else np.nan
        ),
        "max_constr_vel": (
            float(max_constr_vel) if np.isfinite(max_constr_vel) else np.nan
        ),
        "dil_vel": float(dil_vel) if np.isfinite(dil_vel) else np.nan,
    }


def summarize_metrics(filepaths):
    """Summarize metrics across multiple files.

    Returns a dict mapping short metric keys to (mean, std, count).
    Keys: max_diameter, min_diameter, amplitude, t75,
    avg_constr_vel, max_constr_vel, dil_vel
    """
    keys = [
        "D_max",
        "D_min",
        "amplitude",
        "latency",
        "t75",
        "avg_constr_vel",
        "max_constr_vel",
        "dil_vel",
    ]
    accum = {k: [] for k in keys}

    for p in filepaths:
        metrics = compute_metrics_from_file(p)
        if metrics is None:
            continue
        for k in keys:
            v = metrics.get(k, np.nan)
            if np.isfinite(v):
                accum[k].append(v)

    # convert to summary with label keys used in GUI
    def stats(lst):
        if len(lst) == 0:
            return (np.nan, np.nan, 0)
        a = np.array(lst)
        return (float(np.mean(a)), float(np.std(a)), int(len(a)))

    summary = {}
    mapping = {
        "max_diameter": "D_max",
        "min_diameter": "D_min",
        "amplitude": "amplitude",
        "constr_latency": "latency",
        "t75": "t75",
        "avg_constr_vel": "avg_constr_vel",
        "max_constr_vel": "max_constr_vel",
        "dil_vel": "dil_vel",
    }
    for out_key, in_key in mapping.items():
        summary[out_key] = stats(accum[in_key])

    return summary
