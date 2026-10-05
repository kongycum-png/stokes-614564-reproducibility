"""Draw the proposed channel-resolved experiment for Manuscript 614564.

This is a vector optical-layout schematic for a proposed measurement.  It does
not imply that the apparatus was built or that experimental data were acquired.
"""
from pathlib import Path
from math import cos, radians, sin

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)


HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#6F4C9B"
CYAN = "#78C6D0"
RED = "#A83232"
DARK = "#30343B"
GRAY = "#62666D"
LIGHT = "#F5F7F8"
PALE_BLUE = "#E7F1F8"
PALE_ORANGE = "#F9ECE6"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
    }
)


def panel(ax, label, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#A5A5A5")
        spine.set_linewidth(0.85)
    ax.text(0.012, 0.982, label, va="top", ha="left", fontsize=11.2, fontweight="bold")
    ax.text(0.075, 0.980, title, va="top", ha="left", fontsize=9.8, fontweight="bold")


def arrow(ax, start, end, color=DARK, lw=1.0, style="-", ms=7.5, z=8):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=style,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def beam(ax, points, color, width=2.0, alpha=1.0, arrows=True, z=1):
    """Continuous beam with a faint ribbon and optional direction arrows."""
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=width * 4.0, alpha=0.10 * alpha,
            solid_capstyle="round", solid_joinstyle="round", zorder=z)
    ax.plot(xs, ys, color=color, lw=width, alpha=alpha,
            solid_capstyle="round", solid_joinstyle="round", zorder=z + 0.2)
    if arrows:
        lengths = []
        total = 0.0
        for p0, p1 in zip(points[:-1], points[1:]):
            seg = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
            lengths.append(seg)
            total += seg
        target = total * 0.60
        acc = 0.0
        for i, seg in enumerate(lengths):
            if acc + seg >= target and seg > 0:
                frac = (target - acc) / seg
                p0, p1 = points[i], points[i + 1]
                x = p0[0] + frac * (p1[0] - p0[0])
                y = p0[1] + frac * (p1[1] - p0[1])
                dx = (p1[0] - p0[0]) / seg * 0.035
                dy = (p1[1] - p0[1]) / seg * 0.035
                arrow(ax, (x - dx, y - dy), (x + dx, y + dy),
                      color=color, lw=1.25, ms=7.2, z=z + 1)
                break
            acc += seg


def label(ax, x, y, txt, where="below", size=7.0, color=DARK, weight="normal"):
    dy = -0.053 if where == "below" else 0.055
    va = "top" if where == "below" else "bottom"
    ax.text(x, y + dy, txt, ha="center", va=va, fontsize=size,
            color=color, fontweight=weight, linespacing=1.0, zorder=12)


def laser(ax, x, y, w=0.090, h=0.105):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            facecolor="#525A62", edgecolor=DARK, linewidth=0.9, zorder=9,
        )
    )
    ax.add_patch(Rectangle((x + w / 2, y - h * 0.22), w * 0.16, h * 0.44,
                           facecolor="#9AA0A6", edgecolor=DARK, lw=0.7, zorder=9))
    ax.text(x, y + 0.004, "CW", ha="center", va="center",
            fontsize=7.4, color="white", fontweight="bold", zorder=11)
    label(ax, x, y - h / 2, "632.8 nm laser", "below", 6.7)


def lens(ax, x, y, name="L", h=0.125, w=0.022, where="below"):
    ax.add_patch(Ellipse((x, y), w, h, facecolor="#BDE8EF",
                         edgecolor="#317A8B", lw=0.9, zorder=9))
    label(ax, x, y - h / 2 if where == "below" else y + h / 2,
          name, where, 6.7)


def thin_plate(ax, x, y, name, h=0.118, where="below", fc="#E8E5F2"):
    ax.add_patch(Rectangle((x - 0.006, y - h / 2), 0.012, h,
                           facecolor=fc, edgecolor=DARK, lw=0.75, zorder=9))
    label(ax, x, y - h / 2 if where == "below" else y + h / 2,
          name, where, 6.6)


def iris(ax, x, y, name="SF", r=0.026, where="below"):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=DARK, lw=0.9, zorder=9))
    ax.add_patch(Circle((x, y), r * 0.23, facecolor=DARK, edgecolor=DARK, lw=0.5, zorder=10))
    label(ax, x, y - r, name, where, 6.7)


def mirror(ax, x, y, name, angle=45, length=0.080, where="below"):
    theta = radians(angle)
    dx = cos(theta) * length / 2
    dy = sin(theta) * length / 2
    ax.plot([x - dx, x + dx], [y - dy, y + dy],
            color=RED, lw=3.0, solid_capstyle="round", zorder=10)
    ax.plot([x - dx, x + dx], [y - dy, y + dy],
            color="#5C1515", lw=0.65, zorder=11)
    label(ax, x, y - 0.040 if where == "below" else y + 0.040,
          name, where, 6.6)


def pbs(ax, x, y, name, size=0.073, where="below"):
    ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size,
                           facecolor="#D9EEF4", edgecolor="#3B7890",
                           alpha=0.82, lw=0.9, zorder=9))
    ax.plot([x - size / 2, x + size / 2], [y + size / 2, y - size / 2],
            color="#68A4B5", lw=1.0, zorder=10)
    label(ax, x, y - size / 2 if where == "below" else y + size / 2,
          name, where, 6.7, weight="bold")


def slm(ax, x, y, name, h=0.125, where="below", tint="#E7E2F2"):
    w = 0.026
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h,
                           facecolor=tint, edgecolor=DARK, lw=0.85, zorder=9))
    for k in range(-2, 3):
        yy = y + k * h / 6
        ax.plot([x - w / 2 + 0.003, x + w / 2 - 0.003],
                [yy - 0.009, yy + 0.009], color=PURPLE, lw=0.55, zorder=10)
    label(ax, x, y - h / 2 if where == "below" else y + h / 2,
          name, where, 6.7)


def camera(ax, x, y, name, tint, edge, w=0.115, h=0.105):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.005,rounding_size=0.008",
            facecolor="#474D54", edgecolor=DARK, linewidth=0.9, zorder=9,
        )
    )
    ax.add_patch(Rectangle((x - w * 0.10, y - h * 0.30), w * 0.26, h * 0.60,
                           facecolor=tint, edgecolor=edge, lw=0.8, zorder=10))
    ax.add_patch(Circle((x + w * 0.34, y), h * 0.13,
                        facecolor="#11151A", edgecolor="#A8ADB3", lw=0.6, zorder=10))
    label(ax, x, y - h / 2, name, "below", 6.6, color=edge, weight="bold")


def data_box(ax, xy, wh, txt, fc="white", ec=GRAY, size=7.7, bold=False):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.009,rounding_size=0.012",
        facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=4,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
            fontsize=size, fontweight="bold" if bold else "normal",
            linespacing=1.08, zorder=5)
    return patch


fig = plt.figure(figsize=(7.25, 5.80))
gs = fig.add_gridspec(2, 2, height_ratios=[1.16, 0.92], hspace=0.14, wspace=0.10)
ax_a = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[1, 1])


# (a) Polarization interferometer and field preparation.
panel(ax_a, "a", "Preparation of the two circular Airy channels")

y_in = 0.59
laser(ax_a, 0.055, y_in)
beam(ax_a, [(0.108, y_in), (0.322, y_in)], GREEN, 1.8)
iris(ax_a, 0.135, y_in, "SF")
lens(ax_a, 0.177, y_in, "L$_1$")
lens(ax_a, 0.218, y_in, "L$_2$")
thin_plate(ax_a, 0.264, y_in, "HWP")
pbs(ax_a, 0.322, y_in, "PBS$_1$")

y_top, y_bot = 0.785, 0.365
x_left, x_right = 0.375, 0.785
beam(ax_a, [(0.322, y_in), (x_left, y_top), (x_right, y_top), (0.825, y_in)],
     BLUE, 1.8)
beam(ax_a, [(0.322, y_in), (x_left, y_bot), (x_right, y_bot), (0.825, y_in)],
     ORANGE, 1.8)
mirror(ax_a, x_left, y_top, "M$_1$", -45, where="above")
mirror(ax_a, x_left, y_bot, "M$_2$", 45, where="below")
mirror(ax_a, x_right, y_top, "M$_3$", 45, where="above")
mirror(ax_a, x_right, y_bot, "M$_4$", -45, where="below")

slm(ax_a, 0.455, y_top, "SLM$_+$", where="above", tint=PALE_BLUE)
lens(ax_a, 0.535, y_top, "L$_3$", where="above")
iris(ax_a, 0.605, y_top, "F$_+$", where="above")
lens(ax_a, 0.675, y_top, "L$_4$", where="above")
slm(ax_a, 0.455, y_bot, "SLM$_-$", where="above", tint=PALE_ORANGE)
lens(ax_a, 0.535, y_bot, "L$_5$", where="above")
iris(ax_a, 0.605, y_bot, "F$_-$", where="above")
lens(ax_a, 0.675, y_bot, "L$_6$", where="above")

ax_a.text(0.575, 0.872, r"$f_{\rm A}(r)\exp[i(m-1)\phi]$",
          ha="center", va="center", fontsize=7.7, color=BLUE)
ax_a.text(0.575, 0.275, r"$f_{\rm A}(r)\exp[i(m+1)\phi]$",
          ha="center", va="center", fontsize=7.7, color=ORANGE)

pbs(ax_a, 0.825, y_in, "PBS$_2$")
beam(ax_a, [(0.862, y_in), (0.980, y_in)], PURPLE, 2.1)
thin_plate(ax_a, 0.892, y_in, "QWP\n45$^\\circ$", h=0.130)

# Small caustic envelope to indicate free propagation, without presenting data.
ax_a.add_patch(
    Polygon(
        [[0.918, y_in - 0.025], [0.965, y_in - 0.060],
         [0.988, y_in - 0.016], [0.988, y_in + 0.016],
         [0.965, y_in + 0.060], [0.918, y_in + 0.025]],
        closed=True, facecolor=PURPLE, edgecolor="none", alpha=0.10, zorder=0,
    )
)
ax_a.text(0.948, 0.675, "orthogonal circular\nchannels",
          ha="center", va="bottom", fontsize=7.0, color=PURPLE, linespacing=1.0)
ax_a.text(0.960, 0.430, "free propagation", ha="center", va="top",
          fontsize=6.8, color=GRAY)

# Diagnostic pick-off inset.
diag = FancyBboxPatch(
    (0.050, 0.055), 0.275, 0.145,
    boxstyle="round,pad=0.010,rounding_size=0.012",
    facecolor="white", edgecolor="#777777", linewidth=0.8,
    linestyle="--", zorder=2,
)
ax_a.add_patch(diag)
ax_a.text(0.062, 0.181, "input certification", ha="left", va="top",
          fontsize=7.0, fontweight="bold", color=GRAY)
ax_a.text(0.062, 0.121, "QWP + LP: full Stokes",
          ha="left", va="center", fontsize=6.8)
ax_a.text(0.062, 0.079, r"circular selector + reference: $q_\pm$",
          ha="left", va="center", fontsize=6.8)
ax_a.plot([0.343, 0.343, 0.325], [0.415, 0.225, 0.225],
          color=GRAY, lw=0.8, ls="--", zorder=2)
arrow(ax_a, (0.325, 0.225), (0.298, 0.200),
      color=GRAY, lw=0.8, style="--", ms=6)

# Coordinate glyph.
arrow(ax_a, (0.942, 0.115), (0.942, 0.180), color=DARK, lw=0.8, ms=6)
arrow(ax_a, (0.942, 0.115), (0.985, 0.115), color=DARK, lw=0.8, ms=6)
ax_a.add_patch(Circle((0.942, 0.115), 0.006, facecolor=DARK, edgecolor=DARK, zorder=9))
ax_a.text(0.942, 0.190, "$x$", ha="center", va="bottom", fontsize=7)
ax_a.text(0.991, 0.115, "$z$", ha="left", va="center", fontsize=7)
ax_a.text(0.929, 0.100, "$y$", ha="right", va="top", fontsize=7)


# (b) Simultaneous channel-resolved acquisition.
panel(ax_b, "b", "Simultaneous axial acquisition")

y_b = 0.635
beam(ax_b, [(0.045, y_b), (0.745, y_b)], PURPLE, 2.0)
ax_b.plot([0.075, 0.075], [0.535, 0.735], color=DARK, lw=1.0, zorder=7)
ax_b.text(0.075, 0.760, "plane $z$", ha="center", va="bottom", fontsize=7.0)
lens(ax_b, 0.178, y_b, "relay L", h=0.145)
pbs(ax_b, 0.315, y_b, "BS", size=0.080)

# Power-monitor branch.
beam(ax_b, [(0.315, y_b), (0.315, 0.325)], GRAY, 1.1, alpha=0.75)
ax_b.add_patch(Rectangle((0.277, 0.236), 0.076, 0.062,
                         facecolor="#E3E6E8", edgecolor=DARK, lw=0.8, zorder=9))
ax_b.add_patch(Rectangle((0.291, 0.251), 0.048, 0.032,
                         facecolor="#A8C6A0", edgecolor="#4D7150", lw=0.6, zorder=10))
ax_b.text(0.315, 0.210, "$P_{\\rm in}$ monitor", ha="center", va="top", fontsize=6.8)

thin_plate(ax_b, 0.455, y_b, "QWP$_A$", h=0.135)
pbs(ax_b, 0.590, y_b, "PBS$_3$", size=0.082)

# The two channel paths are geometrically distinct and additionally labelled.
beam(ax_b, [(0.631, y_b), (0.805, y_b)], BLUE, 1.8)
camera(ax_b, 0.885, y_b, "camera $+$\n$C_+(x,y,z)$", PALE_BLUE, BLUE)
ax_b.text(0.705, y_b + 0.052, "$+$", ha="center", va="bottom",
          fontsize=8.2, color=BLUE, fontweight="bold")

beam(ax_b, [(0.590, y_b), (0.665, 0.385), (0.805, 0.385)], ORANGE, 1.8)
mirror(ax_b, 0.665, 0.385, "M$_5$", -43, where="below")
camera(ax_b, 0.885, 0.385, "camera $-$\n$C_-(x,y,z)$", PALE_ORANGE, ORANGE)
ax_b.text(0.725, 0.430, "$-$", ha="center", va="bottom",
          fontsize=8.2, color=ORANGE, fontweight="bold")

arrow(ax_b, (0.100, 0.425), (0.215, 0.425), color=GRAY, lw=0.9, ms=7)
arrow(ax_b, (0.215, 0.425), (0.100, 0.425), color=GRAY, lw=0.9, ms=7)
ax_b.text(0.158, 0.390, "translated imaging head", ha="center", va="top",
          fontsize=6.8, color=GRAY)
ax_b.text(0.505, 0.090,
          "Simultaneous full-frame counts; the moving ROI is applied digitally.",
          ha="center", va="center", fontsize=7.3, fontweight="bold")


# (c) Calibrated observable and event extraction.
panel(ax_c, "c", "From two count images to the axial event")

data_box(ax_c, (0.045, 0.705), (0.205, 0.145), "raw counts\n$C_+,\\ C_-$",
         fc="white", ec=GRAY, size=7.5)
data_box(ax_c, (0.300, 0.690), (0.245, 0.175),
         "dark / flat / gain\nleakage\nregistration",
         fc=LIGHT, ec=GRAY, size=6.8)
data_box(ax_c, (0.600, 0.705), (0.350, 0.145),
         "same moving ROI\n$\\rho\\leq\\rho_s(z)$",
         fc="white", ec=GRAY, size=7.5)
arrow(ax_c, (0.250, 0.777), (0.300, 0.777), lw=0.9)
arrow(ax_c, (0.545, 0.777), (0.600, 0.777), lw=0.9)

data_box(ax_c, (0.600, 0.420), (0.350, 0.155),
         "$Q=I_+-I_-$     $T=I_++I_-$\n"
         "$\\bar p_3=Q/T$\n$Q=T\\bar p_3$",
         fc="white", ec=GREEN, size=6.9)
data_box(ax_c, (0.300, 0.405), (0.245, 0.185),
         "first stationary\npeak + rising\nhalf-height",
         fc=LIGHT, ec=GRAY, size=7.0)
data_box(ax_c, (0.045, 0.390), (0.205, 0.215),
         "$z_{50}(m)$\n$\\Delta z_m=$\n$z_{50}(1)-z_{50}(m)$",
         fc="white", ec=PURPLE, size=7.5)
arrow(ax_c, (0.775, 0.705), (0.775, 0.575), lw=0.9)
arrow(ax_c, (0.600, 0.498), (0.545, 0.498), lw=0.9)
arrow(ax_c, (0.300, 0.498), (0.250, 0.498), lw=0.9)

# Two circular-count thumbnails make the link to the cameras explicit.
for x, col, sign in [(0.105, BLUE, "+"), (0.190, ORANGE, "-")]:
    ax_c.add_patch(Circle((x, 0.670), 0.026, facecolor=col,
                          edgecolor=DARK, alpha=0.72, lw=0.6, zorder=7))
    ax_c.add_patch(Circle((x, 0.670), 0.010, facecolor="white",
                          edgecolor="none", alpha=0.72, zorder=8))
    ax_c.text(x, 0.670, sign, ha="center", va="center",
              fontsize=6.5, fontweight="bold", color=DARK, zorder=9)

ax_c.text(0.500, 0.285,
          r"Proxy: $x_s=j'_{m,1}$; actual boundary: calibrated $S_3=0$ contour.",
          ha="center", va="center", fontsize=7.0)
ax_c.text(0.500, 0.165,
          "The scan step sets axial sampling; uncertainty is obtained from\n"
          "repeated calibrated count traces.",
          ha="center", va="center", fontsize=7.0, color=GRAY, linespacing=1.15)
ax_c.text(0.500, 0.070,
          "Apply the same event definition to every topological order.",
          ha="center", va="center", fontsize=7.1, color=DARK, fontweight="bold")


fig.subplots_adjust(left=0.025, right=0.987, bottom=0.025, top=0.987)
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"proposed_experimental_layout.{ext}",
                dpi=450, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUT / "proposed_experimental_layout.pdf")
