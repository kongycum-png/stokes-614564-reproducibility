"""Draw the five main-text figures from retained Manuscript 614564 evidence.

No wave propagation or coefficient fitting is performed here.  The script
reads the cached complex fields, event tables, analytic coefficients, and
frozen-protocol detector samples copied into ``../figure_data``.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.special import jnp_zeros


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figure_data"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73", 4: "#CC79A7",
          5: "#E69F00", 6: "#56B4E9"}
BOUNDARY_COLORS = {"0.9b0": "#0072B2", "b0": "#222222",
                   "1.1b0": "#D55E00", "actual": "#009E73"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "lines.linewidth": 1.15,
    "lines.markersize": 4.2,
    "axes.linewidth": 0.7,
})


def boxed(ax: plt.Axes) -> None:
    """Use boxed axes with inward ticks."""
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color("#333333")
    ax.tick_params(direction="in", top=True, right=True, length=3, width=0.7)


def label_panels(axes) -> None:
    for i, ax in enumerate(np.ravel(axes)):
        ax.text(-0.19, 1.06, chr(97 + i), transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom")
        boxed(ax)


def save(fig: plt.Figure, name: str,
         formats: tuple[str, ...] = ("pdf", "svg", "png")) -> None:
    for ext in formats:
        fig.savefig(FIG / f"{name}.{ext}", dpi=400, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def break_jumps(values: np.ndarray, limit: float = 0.3) -> np.ndarray:
    out = np.asarray(values, float).copy()
    jump = np.abs(np.diff(out)) > limit
    out[1:][jump] = np.nan
    return out


def figure1(*, formats: tuple[str, ...] = ("pdf", "svg", "png")) -> None:
    fields = np.load(DATA / "R24_eta0.7_fields.npz")
    curves = np.load(DATA / "R24_eta0.7_curves.npz")
    meta = json.loads((DATA / "R24_eta0.7.json").read_text())
    tau = fields["tau"]
    use = (tau >= -1.25) & (tau <= 1.50)
    ids = np.flatnonzero(use)
    x = np.linspace(0, 8.5, 401)
    u = np.array([CubicSpline(fields["x"][i], fields["U"][i, 2])(x) for i in ids])
    v = np.array([CubicSpline(fields["x"][i], fields["U"][i, 4])(x) for i in ids])
    s0 = 0.5 * (np.abs(u) ** 2 + np.abs(v) ** 2)
    s3 = 0.5 * (np.abs(u) ** 2 - np.abs(v) ** 2)
    norm = float(np.max(s0))

    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.15), layout="constrained")
    ax = axs.ravel()
    im0 = ax[0].pcolormesh(tau[use], x, (s0 / norm).T, cmap="viridis",
                           shading="auto", rasterized=True)
    root3 = break_jumps(curves["m3__actual_root__boundary_x"])
    ax[0].plot(tau[use], root3[use], color="white", lw=1.2, label="actual zero")
    ax[0].axhline(jnp_zeros(3, 1)[0], color="white", ls="--", lw=1.0,
                  label=r"proxy $j'_{3,1}$")
    ax[0].set(xlabel=r"$\tau$", ylabel=r"$x=z\rho/(2kw^3)$",
              title=r"Intensity caustic, $S_0/S_{0,\max}$")
    ax[0].legend(loc="upper left", frameon=False, labelcolor="white")
    cb0 = fig.colorbar(im0, ax=ax[0], pad=0.02, shrink=0.92)

    im1 = ax[1].pcolormesh(tau[use], x, (s3 / norm).T, cmap="RdBu_r",
                           vmin=-1, vmax=1, shading="auto", rasterized=True)
    ax[1].plot(tau[use], root3[use], color="#111111", lw=1.2, label="actual zero")
    ax[1].axhline(jnp_zeros(3, 1)[0], color="#111111", ls="--", lw=1.0,
                  label=r"proxy $j'_{3,1}$")
    ax[1].axhline(0, color="#777777", lw=0.5)
    ax[1].set(xlabel=r"$\tau$", ylabel=r"$x=z\rho/(2kw^3)$",
              title=r"Signed domain, $S_3/S_{0,\max}$")
    ax[1].legend(loc="upper left", frameon=False)
    cb1 = fig.colorbar(im1, ax=ax[1], pad=0.02, shrink=0.92)

    tp = float(meta["events"]["3"]["scaled_1"]["tau_peak"])
    it = int(np.argmin(np.abs(tau - tp)))
    xp = fields["x"][it]
    plus = 0.5 * np.abs(fields["U"][it, 2]) ** 2
    minus = 0.5 * np.abs(fields["U"][it, 4]) ** 2
    pnorm = float(np.max(plus + minus))
    show = xp <= 8.5
    ax[2].plot(xp[show], plus[show] / pnorm, color=COLORS[1], label=r"$I_+$ ($q_+=2$)")
    ax[2].plot(xp[show], minus[show] / pnorm, "--", color=COLORS[2], label=r"$I_-$ ($q_-=4$)")
    ax[2].plot(xp[show], (plus[show] - minus[show]) / pnorm, color=COLORS[3],
               label=r"$S_3=I_+-I_-$")
    proxy = float(jnp_zeros(3, 1)[0])
    actual = float(curves["m3__actual_root__boundary_x"][it])
    ax[2].axvline(proxy, color="#222222", ls="--", lw=1.0, label="proxy")
    ax[2].axvline(actual, color="#777777", ls=":", lw=1.2, label="actual zero")
    ax[2].axhline(0, color="#777777", lw=0.55)
    ax[2].set(xlim=(0, 8.5), xlabel=r"$x$",
              ylabel=r"$I_\pm,\ S_3$ (norm. to $S_{0,\max}$)",
              title=fr"Radial profiles, $m=3$, $\tau_p={tp:.3f}$")
    ax[2].set_ylim(ax[2].get_ylim()[0], 1.30)
    ax[2].legend(loc="upper right", ncol=1, frameon=False, fontsize=7)

    for m in (1, 2, 3, 4):
        actual_curve = break_jumps(curves[f"m{m}__actual_root__boundary_x"])
        ax[3].plot(tau[use], actual_curve[use], color=COLORS[m], label=f"$m={m}$")
        ax[3].axhline(jnp_zeros(m, 1)[0], color=COLORS[m], ls="--", lw=0.85)
    ax[3].set(xlabel=r"$\tau$", ylabel=r"Radial boundary $x$",
              title="Actual zero (solid) and proxy (dashed)")
    ax[3].set_ylim(ax[3].get_ylim()[0], 7.20)
    ax[3].legend(loc="upper center", ncol=2, frameon=False)
    label_panels(axs)
    for panel in ax:
        panel.tick_params(axis="both", labelsize=8)
    for colorbar in (cb0, cb1):
        colorbar.ax.tick_params(labelsize=8)
    save(fig, "Figure1_local_domain", formats=formats)


def figure2(*, formats: tuple[str, ...] = ("pdf", "svg", "png")) -> None:
    curves = np.load(DATA / "R24_eta0.7_curves.npz")
    meta = json.loads((DATA / "R24_eta0.7.json").read_text())
    signal = pd.read_csv(DATA / "qt_signal_events.csv")
    pairs = pd.read_csv(DATA / "qt_paired_events.csv")
    tau = curves["tau"]
    use = (tau >= -1.20) & (tau <= 1.55)
    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.0), layout="constrained")
    ax = axs.ravel()

    for m in (1, 2, 3, 4):
        q = curves[f"m{m}__scaled_1__Q"]
        ev = meta["events"][str(m)]["scaled_1"]
        p, qp = float(ev["tau_peak"]), float(ev["q_peak"])
        h = float(ev["thresholds"]["0.5"]["tau"])
        ax[0].plot(tau[use], q[use], color=COLORS[m], label=f"$m={m}$")
        ax[0].plot(p, qp, "o", mfc="white", mec=COLORS[m], mew=1.0)
        ax[1].plot(tau[use], q[use] / qp, color=COLORS[m], label=f"$m={m}$")
        ax[1].plot(h, 0.5, "o", mfc="white", mec=COLORS[m], mew=1.0)
    ax[0].set(xlabel=r"$\tau$", ylabel=r"$Q_m$",
              title="Input-normalized first-domain signal")
    ax[0].legend(loc="upper left", ncol=2, frameon=False)
    ax[1].axhline(0.5, color="#666666", ls="--", lw=0.8)
    ax[1].set(xlabel=r"$\tau$", ylabel=r"$Q_m/Q_m(\tau_p)$",
              title="Rising half-height positions")
    ax[1].legend(loc="lower right", ncol=2, frameon=False)

    m = 4
    q = curves[f"m{m}__scaled_1__Q"]
    t = curves[f"m{m}__scaled_1__T"]
    pb = curves[f"m{m}__scaled_1__pbar"]
    qr = signal[(signal.R0 == 24) & (signal.eta == 0.7) & (signal.m == m) &
                (signal.boundary == "scaled_1") & (signal.signal == "Q")].iloc[0]
    tr = signal[(signal.R0 == 24) & (signal.eta == 0.7) & (signal.m == m) &
                (signal.boundary == "scaled_1") & (signal.signal == "T")].iloc[0]
    ax[2].plot(tau[use], q[use] / qr.peak_value_input_normalized,
               color=COLORS[1], label=r"$Q/Q(\tau_{p,Q})$")
    ax[2].plot(tau[use], t[use] / tr.peak_value_input_normalized, "--",
               color=COLORS[2], label=r"$T/T(\tau_{p,T})$")
    ax[2].plot(qr["h_0.5"], 0.5, "o", mfc="white", mec=COLORS[1])
    ax[2].plot(tr["h_0.5"], 0.5, "s", mfc="white", mec=COLORS[2])
    ax2r = ax[2].twinx()
    ax2r.plot(tau[use], pb[use], color=COLORS[3], lw=1.0, alpha=0.9,
              label=r"$\bar p_3$")
    ax2r.set_ylabel(r"$\bar p_3$", color=COLORS[3])
    ax2r.tick_params(axis="y", colors=COLORS[3], direction="in", left=False, right=True)
    ax2r.spines["right"].set_color(COLORS[3])
    handles, labels = ax[2].get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax[2].legend(handles + h2, labels + l2, loc="upper left", frameon=False)
    ax[2].set(xlabel=r"$\tau$", ylabel="Own-peak-normalized signal",
              title=r"$Q$, $T$ and $\bar p_3$, $m=4$")
    # Align the independent y scales at identical fractional tick positions.
    ax[2].set_ylim(0.0, 1.25)
    ax[2].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax2r.set_ylim(0.68, 0.805)
    ax2r.set_yticks([0.70, 0.72, 0.74, 0.76, 0.78, 0.80])

    grid = pairs[(pairs.boundary == "scaled_1") & (pairs.gamma == 0.5) &
                 (pairs.event_status == "defined") &
                 np.isfinite(pairs.Delta_Q) & np.isfinite(pairs.Delta_T)]
    for m in (2, 3, 4, 5, 6):
        part = grid[grid.m == m]
        ax[3].scatter(part.Delta_T, part.Delta_Q, s=13, alpha=0.42,
                      color=COLORS[m], edgecolors="none", label=f"$m={m}$")
    lim = 0.89
    ax[3].plot([0, lim], [0, lim], color="#333333", ls="--", lw=0.85,
               label=r"$\Delta_Q=\Delta_T$")
    for R, marker, label_offset in ((24, "*", (5, -10)), (96, "P", (12, -5))):
        r = grid[(grid.R0 == R) & (grid.eta == 0.7) & (grid.m == 4)].iloc[0]
        ax[3].plot(r.Delta_T, r.Delta_Q, marker=marker, ms=8, color="#111111",
                   mfc="white", mew=1.0, ls="none")
        ax[3].annotate(fr"$R_0={R}$", (r.Delta_T, r.Delta_Q), xytext=label_offset,
                       textcoords="offset points", fontsize=6.5)
    ax[3].set(xlim=(0, lim), ylim=(0, lim), aspect="auto",
              xlabel=r"Power advance $\Delta_T$", ylabel=r"Stokes advance $\Delta_Q$",
              title="Stokes versus power advance")
    ax[3].legend(loc="upper left", ncol=1, frameon=False, fontsize=6.2)
    label_panels(axs)
    # Use the same rectangular plotting frame and panel-label anchor as b.
    ax[2].tick_params(axis="y", right=False)
    save(fig, "Figure2_event_definition", formats=formats)


def figure3(*, formats: tuple[str, ...] = ("pdf", "svg", "png")) -> None:
    pred = pd.read_csv(DATA / "analytic_predictions.csv")
    radii = [24, 32, 40, 56, 72, 96]
    fig, axs = plt.subplots(2, 2, figsize=(7.25, 5.0), layout="constrained")
    ax = axs.ravel()
    for ia, eta in enumerate((0.35, 0.70)):
        for m in (2, 3, 4):
            p = pred[(pred.boundary == "scaled_1") & (pred.eta == eta) &
                     (pred.m == m) & (pred.R0.isin(radii))].sort_values("R0")
            inv = 1.0 / p.R0.to_numpy(float)
            ax[ia].plot(inv, p.advance, "o", color=COLORS[m], label=f"$m={m}$")
            xx = np.linspace(0, max(inv) * 1.06, 160)
            ax[ia].plot(xx, float(p.A.iloc[0]) * xx, "--", color=COLORS[m], lw=1.0)
        ax[ia].set(xlabel=r"$1/R_0$", ylabel=r"$\Delta\tau_{50}$",
                   title=fr"Axial advance, $\eta={eta:.2f}$")
        ax[ia].legend(loc="upper left", frameon=False)

    for eta, marker in ((0.35, "o"), (0.70, "s")):
        for m in (2, 3, 4):
            p = pred[(pred.boundary == "scaled_1") & (pred.eta == eta) &
                     (pred.m == m) & (pred.R0.isin(radii))].sort_values("R0")
            ratio = p.R0 * p.advance / p.A
            ax[2].plot(p.R0, ratio, marker=marker, ls="-", color=COLORS[m])
    ax[2].axhline(1, color="#555555", ls=":", lw=0.9)
    ax[2].set(xlabel=r"$R_0$", ylabel=r"$R_0\Delta\tau_{50}/A_m$",
              title="Normalized axial advance")
    order_handles = [Line2D([0], [0], color=COLORS[m], ls="-", label=f"$m={m}$")
                     for m in (2, 3, 4)]
    eta_handles = [Line2D([0], [0], color="#333333", ls="-", marker=marker,
                          label=fr"$\eta={eta:.2f}$")
                   for eta, marker in ((0.35, "o"), (0.70, "s"))]
    leg1 = ax[2].legend(handles=order_handles, loc="center left",
                       bbox_to_anchor=(0.01, 0.50), frameon=False, fontsize=6.3)
    ax[2].add_artist(leg1)
    ax[2].legend(handles=eta_handles, loc="lower right", frameon=False,
                 fontsize=6.3)
    ax[2].annotate("Asymptotic limit", xy=(0.97, 1.0),
                   xycoords=("axes fraction", "data"), xytext=(0, -6),
                   textcoords="offset points", ha="right", va="top",
                   fontsize=6.3, color="#555555")

    for eta, marker, color in ((0.35, "o", COLORS[1]), (0.70, "s", COLORS[2])):
        p = pred[(pred.boundary == "scaled_1") & (pred.eta == eta) &
                 (pred.m == 4)].sort_values("R0")
        residual = p.R0 ** 2 * (p.advance - p.A / p.R0 - p.B / p.R0 ** 1.5)
        ax[3].plot(p.R0, residual, marker=marker, color=color,
                   label=fr"$\eta={eta:.2f}$")
        ax[3].axhline(float(p.C.iloc[0]), color=color, ls="--", lw=1.0)
    ax[3].set(xscale="log", xlabel=r"$R_0$",
              ylabel=r"$R_0^2(\Delta\tau_{50}-A/R_0-B/R_0^{3/2})$",
              title=r"Fourth-order residual, $m=4$")
    ax[3].set_xticks([16, 24, 40, 72, 128, 256], [16, 24, 40, 72, 128, 256])
    ax[3].legend(loc="best", frameon=False)
    label_panels(axs)
    save(fig, "Figure3_topological_advance", formats=formats)


def figure4(*, formats: tuple[str, ...] = ("pdf", "svg", "png")) -> None:
    boundary = pd.read_csv(DATA / "four_boundary_representative.csv")
    asym = pd.read_csv(DATA / "actual_proxy_asymptotic_comparison.csv")
    root = np.load(DATA / "R24_eta0.9_desingularized.npz")
    fig, axs = plt.subplots(2, 3, figsize=(7.25, 5.35), layout="constrained")
    ax = axs.ravel()
    defs = [
        ("Delta_b0_0p9", "0.9b0", "o", "-"),
        ("Delta_b0", "b0", "s", "-"),
        ("Delta_b0_1p1", "1.1b0", "^", "-"),
    ]
    for a, (R, eta) in zip(ax[:4], ((24, .35), (24, .70), (96, .35), (96, .70))):
        p = boundary[(boundary.R0 == R) & (boundary.eta == eta)].sort_values("m")
        for key, label, marker, ls in defs:
            a.plot(p.m, p[key], marker=marker, ls=ls,
                   color=BOUNDARY_COLORS[label], label=label)
        a.plot(p.m, p.Delta_actual, ls=":", color=BOUNDARY_COLORS["actual"])
        continuous = p.Q_root_branch_status_actual == "continuous"
        a.plot(p.loc[continuous, "m"], p.loc[continuous, "Delta_actual"], "D",
               color=BOUNDARY_COLORS["actual"], ls="none")
        a.plot(p.loc[~continuous, "m"], p.loc[~continuous, "Delta_actual"], "D",
               mfc="white", mec=BOUNDARY_COLORS["actual"], mew=1.0, ls="none")
        a.axhline(0, color="#777777", lw=0.6)
        a.set_xticks([2, 3, 4])
        a.set(xlabel="$m$", ylabel=r"Signed advance $\Delta\tau_{50}$",
              title=fr"$R_0={R},\ \eta={eta:.2f}$")
    legend_defs = defs + [("Delta_actual", "actual", "D", ":")]
    handles = [Line2D([0], [0], color=BOUNDARY_COLORS[label], marker=marker,
                      ls=ls, mfc=BOUNDARY_COLORS[label],
                      label={"0.9b0": r"$0.9b_0$", "b0": r"$b_0$",
                             "1.1b0": r"$1.1b_0$", "actual": "actual zero"}[label])
               for _, label, marker, ls in legend_defs]
    handles.append(Line2D([0], [0], color=BOUNDARY_COLORS["actual"],
                          marker="D", ls=":", mfc="white",
                          mec=BOUNDARY_COLORS["actual"], mew=1.0,
                          label="actual zero\n(root break)"))
    ax[0].legend(handles=handles, loc="upper left", frameon=False,
                  fontsize=6.5, handlelength=1.4, handletextpad=0.4,
                  labelspacing=0.25, borderpad=0.25)

    proxy = asym[asym.boundary == "scaled_1"].sort_values("R0")
    actual = asym[asym.boundary == "actual_root"].sort_values("R0")
    ax[4].plot(proxy.R0, proxy.effective_K, "o-", color=COLORS[1], label="proxy")
    cont = actual.asymptotic_sequence_status == "eligible_continuous_Q_branch"
    ax[4].plot(actual.loc[cont, "R0"], actual.loc[cont, "effective_K"], "s-",
               color=COLORS[3], label="actual, continuous")
    broken = ~cont
    ax[4].plot(actual.loc[broken, "R0"], actual.loc[broken, "effective_K"], "s",
               mfc="white", mec=COLORS[3], mew=1.0, ls="none", label="actual, root break")
    ax[4].axhline(float(proxy.K_theory.iloc[0]), color=COLORS[1], ls="--", lw=1.0)
    ax[4].axhline(float(actual.K_theory.iloc[0]), color=COLORS[3], ls="--", lw=1.0)
    ax[4].set(xscale="log", xlabel=r"$R_0$",
              ylabel=r"$R_0\Delta\tau_{50}/(D_1-D_4)$",
              title="Asymptotic coefficient by boundary")
    ax[4].set_xticks([16, 24, 40, 72, 128, 256], [16, 24, 40, 72, 128, 256])
    ax[4].legend(loc="best", frameon=False, fontsize=6.2)

    tr = root["tau"]
    use = (tr >= -3.0) & (tr <= 5.0)
    for m in (1, 4, 6):
        values = root[f"m{m}_roots"]
        ax[5].plot(tr[use], values[use], ".", ms=2.0, color=COLORS[m], alpha=0.28)
        ax[5].plot(tr[use], break_jumps(values)[use], color=COLORS[m], label=f"$m={m}$")
        ax[5].axhline(jnp_zeros(m, 1)[0], color=COLORS[m], ls="--", lw=0.75)
    ax[5].set(xlabel=r"$\tau$", ylabel="Selected first zero in $x$",
              title="Selected actual-zero trajectories")
    for m, target_tau, text_position in ((4, -1.25, (-3.0, 10.0)),
                                         (6, 2.375, (0.55, 13.20))):
        i = int(np.argmin(np.abs(tr - target_tau)))
        annotation = fr"$m={m}$: root jump"
        if m == 4:
            annotation += "\n" + r"$\tau\simeq-1.26$"
        ax[5].annotate(annotation,
                       xy=(tr[i], root[f"m{m}_roots"][i]),
                       xytext=text_position, fontsize=6.3, color="#222222",
                       linespacing=1.0,
                       arrowprops={"arrowstyle": "-|>", "mutation_scale": 9,
                                   "shrinkB": 0, "lw": 0.85,
                                   "color": "#333333"})
    ax[5].legend(loc="center left", bbox_to_anchor=(0.02, 0.19),
                 frameon=False, labelspacing=0.2, borderpad=0.2)
    label_panels(axs)
    save(fig, "Figure4_boundary_dependence", formats=formats)


def estimate_current_blind(tau: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, float, float]:
    step = float(tau[1] - tau[0])
    window = max(5, int(round(0.25 / step)))
    window += 1 - window % 2
    smooth = savgol_filter(raw, window, 3, mode="interp")
    eligible = (tau >= 0.2) & (tau <= 1.8)
    maxima = np.zeros_like(smooth, dtype=bool)
    maxima[1:-1] = (smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] > smooth[2:])
    maxima &= eligible
    idx = int(np.argmax(np.where(maxima, smooth, -np.inf)))
    a, b, c = smooth[idx - 1], smooth[idx], smooth[idx + 1]
    den = a - 2 * b + c
    vertex = (a - c) / (2 * den)
    peak = b - (a - c) ** 2 / (8 * den)
    threshold = 0.5 * peak
    rising = np.flatnonzero((smooth[:-1] < threshold) & (smooth[1:] >= threshold) &
                            (np.arange(len(tau) - 1) < idx))
    cross = int(rising[-1])
    fraction = (threshold - smooth[cross]) / (smooth[cross + 1] - smooth[cross])
    half = float(tau[cross] + fraction * step)
    peak_tau = float(tau[idx] + vertex * step)
    return smooth, half, peak_tau


def figure5(*, formats: tuple[str, ...] = ("pdf", "svg", "png")) -> None:
    stages = pd.read_csv(DATA / "measurement_stage_pairs.csv")
    ablation = pd.read_csv(DATA / "measurement_failure_ablations.csv")
    split = pd.read_csv(DATA / "measurement_failure_noise_split.csv")
    s24 = np.load(DATA / "R24_eta0.7_N1e+06_focused_samples.npz")
    s96 = np.load(DATA / "R96_eta0.7_N1e+06_focused_samples.npz")
    cfg24 = json.loads((DATA / "R24_eta0.7_N1e+06.json").read_text())
    fig, axs = plt.subplots(2, 3, figsize=(7.25, 5.35), layout="constrained")
    ax = axs.ravel()

    tau = s24["tau"]
    z = tau * float(cfg24["Lc_mm"])
    order_index = 2  # m=3
    raw = s24["calibration_raw_aggregate_Iplus_Iminus"][0, order_index]
    expected = s24["calibration_expected_raw_aggregate"][0, order_index]
    monitor = s24["calibration_reference_monitor"][0]
    ax[0].plot(z, raw[0], color=COLORS[1], lw=0.8, alpha=0.70, label=r"measured $C_+$")
    ax[0].plot(z, raw[1], color=COLORS[2], lw=0.8, alpha=0.70, label=r"measured $C_-$")
    ax[0].plot(z, expected[0], "--", color=COLORS[1], lw=1.0, label=r"mean $C_+$")
    ax[0].plot(z, expected[1], "--", color=COLORS[2], lw=1.0, label=r"mean $C_-$")
    ax[0].set(xlabel=r"$z-z_c$ (mm)", ylabel="ROI electrons",
              title=r"Raw $C_+$ and $C_-$ channel counts")
    ax[0].set_ylim(top=22000)
    ax[0].legend(loc="upper left", ncol=2, frameon=False, fontsize=6.2)

    qraw = (raw[0] - raw[1]) / monitor
    qmean = (expected[0] - expected[1]) / 1e6
    qsmooth, half, peak = estimate_current_blind(tau, qraw)
    ax[1].plot(z, 1e3 * qraw, color="#AAAAAA", lw=0.65, label="count difference")
    ax[1].plot(z, 1e3 * qsmooth, color=COLORS[1], label="fixed smoothing")
    ax[1].plot(z, 1e3 * qmean, "--", color=COLORS[2], label="detector mean")
    peak_y = float(np.interp(peak, tau, qsmooth))
    ax[1].plot(peak * cfg24["Lc_mm"], 1e3 * peak_y, "o", mfc="white", mec=COLORS[1])
    ax[1].plot(half * cfg24["Lc_mm"], 0.5e3 * peak_y, "s", mfc="white", mec=COLORS[1])
    ax[1].set(xlabel=r"$z-z_c$ (mm)", ylabel=r"$10^3\widehat Q$",
              title="Estimated Stokes signal")
    ax[1].set_ylim(top=18)
    ax[1].legend(loc="upper left", frameon=False, fontsize=6.2,
                 handlelength=1.4, handletextpad=0.4, labelspacing=0.35)

    for a, tag, title in ((ax[2], "R24_eta0.7_N1e+06", r"Axial advance, $R_0=24$"),
                          (ax[3], "R96_eta0.7_N1e+06", r"Axial advance, $R_0=96$")):
        p = stages[(stages.tag == tag) & (stages.estimator == "current_blind")].sort_values("m")
        keys = ["Delta_ideal_mm", "Delta_detector_mm", "Delta_estimator0_mm", "Delta_random_mean_mm"]
        xpos = np.arange(4)
        for m in (2, 3, 4):
            r = p[p.m == m].iloc[0]
            a.plot(xpos, [r[k] for k in keys], "o-", color=COLORS[m], label=f"$m={m}$")
        a.axhline(0, color="#777777", lw=0.6)
        a.set_xticks(xpos, ["A\nField", "B\nDetector", "C\nEstimator", "D\nNoise"])
        a.set(ylabel="Advance (mm)", title=title)
        a.legend(loc="center right" if a is ax[2] else "best",
                 bbox_to_anchor=(1.0, 0.45) if a is ax[2] else None,
                 frameon=False, fontsize=6.3)

    h24 = s24["current_blind_val_half_mm"]
    h96 = s96["current_blind_val_half_mm"]
    d24 = h24[:, 0] - h24[:, 3]
    d96 = h96[:, 0] - h96[:, 3]
    bp = ax[4].boxplot([d24[np.isfinite(d24)], d96[np.isfinite(d96)]], positions=[1, 2], widths=0.5,
                       showfliers=False, patch_artist=True,
                       medianprops={"color": "#111111", "lw": 1.2})
    for patch, color in zip(bp["boxes"], (COLORS[1], COLORS[2])):
        patch.set_facecolor(color); patch.set_alpha(0.45)
    ideal24 = float(stages[(stages.tag == "R24_eta0.7_N1e+06") &
                           (stages.estimator == "current_blind") & (stages.m == 4)].Delta_ideal_mm.iloc[0])
    ideal96 = float(stages[(stages.tag == "R96_eta0.7_N1e+06") &
                           (stages.estimator == "current_blind") & (stages.m == 4)].Delta_ideal_mm.iloc[0])
    ax[4].plot([1, 2], [ideal24, ideal96], "D", color="#111111", mfc="white", label="ideal")
    ax[4].axhline(0, color="#777777", lw=0.6)
    ax[4].set_xticks([1, 2], [r"$R_0=24$", r"$R_0=96$"])
    ax[4].set(ylabel=r"Estimated $\Delta z_{50}$ (mm)",
              title=r"Advance distributions, $m=4$")
    ax[4].legend(loc="lower left", frameon=False)

    fail = stages[(stages.tag == "R96_eta0.7_N1e+06") &
                  (stages.estimator == "current_blind")].set_index("m")
    sources = {
        "counts only": ablation[(ablation.scenario == "counting_only") &
                                  (ablation.estimator == "current_blind")].set_index("m").sign_recovery_all,
        "+ background": split[(split.scenario == "counting_plus_background") &
                                (split.estimator == "current_blind")].set_index("m").sign_recovery_all,
        "+ read": split[(split.scenario == "counting_plus_read") &
                          (split.estimator == "current_blind")].set_index("m").sign_recovery_all,
        "full model": fail.sign_recovery_all,
    }
    for (label, vals), marker, color in zip(sources.items(), ("o", "s", "^", "D"),
                                            (COLORS[1], COLORS[3], COLORS[4], COLORS[2])):
        ax[5].plot([2, 3, 4], 100 * np.array([vals.loc[m] for m in (2, 3, 4)]),
                   marker=marker, color=color, label=label)
    ax[5].axhline(50, color="#777777", ls="--", lw=0.7)
    ax[5].set(xlabel="$m$", ylabel="Correct-sign trials (%)", ylim=(0, 103),
              title="Correct-sign rate by noise source")
    ax[5].set_xticks([2, 3, 4])
    ax[5].legend(loc="best", frameon=False, fontsize=6.1)
    label_panels(axs)
    save(fig, "Figure5_measurement_diagnosis", formats=formats)


def main() -> None:
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    print("Wrote Figures 1--5 as PDF, SVG, and PNG.")


if __name__ == "__main__":
    main()
