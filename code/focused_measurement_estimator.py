"""Peak/half-height estimators used in the focused measurement diagnosis."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def estimate(
    tau: np.ndarray,
    raw: np.ndarray,
    *,
    smoothing_width_tau: float,
    selection: str,
    reference_peak_tau: float | np.ndarray | None = None,
    peak_window: tuple[float, float] = (0.2, 1.8),
) -> dict:
    raw = np.atleast_2d(raw)
    step = float(tau[1] - tau[0])
    window = max(5, int(round(smoothing_width_tau / step)))
    window += 1 - window % 2
    smooth = savgol_filter(raw, window, 3, axis=1, mode="interp")
    rows = np.arange(len(smooth))
    eligible = (tau >= peak_window[0]) & (tau <= peak_window[1])
    maxima = np.zeros_like(smooth, dtype=bool)
    maxima[:, 1:-1] = (smooth[:, 1:-1] > smooth[:, :-2]) & (smooth[:, 1:-1] > smooth[:, 2:])
    maxima &= eligible

    if selection == "largest":
        index = np.argmax(np.where(maxima, smooth, -np.inf), axis=1)
    elif selection == "nearest_to_reference_peak":
        if reference_peak_tau is None:
            raise ValueError("nearest_to_reference_peak requires reference_peak_tau")
        target = np.broadcast_to(np.asarray(reference_peak_tau, dtype=float), (len(smooth),))
        index = np.argmin(np.where(maxima, np.abs(tau[None, :] - target[:, None]), np.inf), axis=1)
    else:
        raise ValueError(f"Unknown peak selection rule: {selection}")

    index = np.clip(index, 1, len(tau) - 2)
    left = smooth[rows, index - 1]
    center = smooth[rows, index]
    right = smooth[rows, index + 1]
    denominator = left - 2 * center + right
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = (left - right) / (2 * denominator)
        peak = center - (left - right) ** 2 / (8 * denominator)
    valid = maxima.any(axis=1) & (denominator < 0) & (np.abs(vertex) < 1) & (peak > 0)
    threshold = 0.5 * peak
    rises = (
        (smooth[:, :-1] < threshold[:, None])
        & (smooth[:, 1:] >= threshold[:, None])
        & (np.arange(len(tau) - 1)[None, :] < index[:, None])
    )
    crossing = np.max(np.where(rises, np.arange(len(tau) - 1)[None, :], -1), axis=1)
    valid &= crossing >= 0
    crossing = np.maximum(crossing, 0)
    slope = (smooth[rows, crossing + 1] - smooth[rows, crossing]) / step
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = (threshold - smooth[rows, crossing]) / (slope * step)
    half = tau[crossing] + fraction * step
    valid &= np.isfinite(half) & (slope > 0) & (fraction >= 0) & (fraction <= 1)
    half[~valid] = np.nan
    peak_tau = tau[index] + vertex * step
    peak_tau[~valid] = np.nan
    selected_index = index.astype(float)
    selected_index[~valid] = np.nan
    return {
        "half_tau": half,
        "peak_tau": peak_tau,
        "selected_index": selected_index,
        "n_local_maxima": maxima.sum(axis=1),
        "valid": valid,
        "window_samples": window,
    }
