"""Replot SI S2/S3 from retained arrays; no field propagation or event extraction."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from publication_style_si import apply_style, finalize_figure, save_figure_set

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/copyedit_inputs'
OUT = ROOT / 'latex_source/figures'
QA = ROOT / 'qa/copyedit_20261004'
QA.mkdir(parents=True, exist_ok=True)
audit = {'propagation_runs': 0, 'event_reextraction': False, 'data': {}, 'lines': {},
         'format_changes': ['S2 scientific notation', 'S3 subscript spacing', 'S3 legend clearance']}

def digest(a):
    a = np.ascontiguousarray(a, dtype='<f8')
    return hashlib.sha256(a.tobytes()).hexdigest()

def retain_lines(fig, name):
    audit['lines'][name] = [
        [{'x': digest(line.get_xdata()), 'y': digest(line.get_ydata())}
         for line in ax.lines] for ax in fig.axes
    ]

def read(path):
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        for k, v in row.items():
            try:
                row[k] = float(v) if v else np.nan
            except ValueError:
                pass
    return rows

def series(rows, select, x, y):
    rr = sorted((r for r in rows if select(r)), key=lambda r: r[x])
    return np.array([r[x] for r in rr]), np.array([r[y] for r in rr])

apply_style()
terms = np.load(DATA / 'moving_domain_terms.npz')
tau = terms['tau']
fig, axs = plt.subplots(1, 3, figsize=(7.0, 3.5))
for ax, m in zip(axs, [1, 2, 4]):
    total = terms[f'm{m}_scaled_1_total']
    current = terms[f'm{m}_scaled_1_current']
    moving = terms[f'm{m}_scaled_1_moving']
    for values, label, color, ls in [
        (total, r'$dQ/d\tau$', 'k', '-'),
        (current, 'current', '#2563eb', '-'),
        (moving, 'moving', '#ea580c', '-'),
        (current + moving, 'sum', '#15803d', '--'),
    ]:
        ax.plot(tau, values, color=color, ls=ls, label=label)
    ax.set(xlim=(-.5, 1.25), xlabel=r'$\tau$', ylabel=r'$dQ/d\tau$', title=f'm = {m}')
    # The scale remains unchanged; only the scientific-notation typography changes.
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
axs[0].legend(ncol=1, loc='lower left')
fig.tight_layout(pad=1.2, w_pad=1.0)
finalize_figure(fig, axs, height_in=3.8, annotate_titles=True)
for ax in axs:
    ax.yaxis.get_offset_text().set_x(.08)
retain_lines(fig, 'Figure S2')
save_figure_set(fig, OUT / 'moving_domain_decomposition')
plt.close(fig)

apply_style()
colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
inputs = read(DATA / 'input_tolerances.csv')
translations = read(DATA / 'translation_events.csv')
fig, axs = plt.subplots(2, 3, figsize=(7.3, 4.8), layout='constrained')
axes = axs.ravel()
for i, (radius, eta) in enumerate([(24, .35), (24, .7), (96, .35), (96, .7)]):
    for j, kind in enumerate(['power', 'scale', 'ring', 'phase']):
        x, y = series(inputs, lambda r: r['R0'] == radius and r['eta'] == eta
                      and r['m'] == 4 and r['kind'] == kind, 'value', 'advance')
        baseline = next(r['advance'] for r in inputs if r['R0'] == radius and r['eta'] == eta
                        and r['m'] == 4 and r['kind'] == 'balanced')
        axes[j].plot(x * 100 if j < 3 else x, y / baseline, 'o-', color=colors[i],
                     label=fr'${radius},{eta:g}$')
    rr = sorted((r for r in inputs if r['R0'] == radius and r['eta'] == eta
                 and r['m'] == 4 and r['kind'] == 'finite_aperture'), key=lambda r: r['radius_r0'])
    axes[4].plot([r['radius_r0'] for r in rr],
                 [r['advance'] / r['balanced_advance'] for r in rr], 'o-', color=colors[i])
    for row in rr:
        if not np.isfinite(row['advance']):
            axes[4].plot(row['radius_r0'], .05 + i * .12, 'x', color=colors[i], ms=6)
    rr = sorted((r for r in translations if r['tag'].startswith(
        f'R{radius}_eta{eta:g}_plus_channel_only_') and r['m'] == 4),
        key=lambda r: float(r['tag'].split('_d')[-1]))
    baseline = rr[0]['advance']
    axes[5].plot([float(r['tag'].split('_d')[-1]) for r in rr],
                 [r['advance'] / baseline for r in rr], 'o-', color=colors[i])
for ax in axes:
    ax.axhline(1, color='.5', ls=':', lw=.7)
axes[1].set_yscale('symlog', linthresh=2)
axes[0].legend(loc='lower right', ncol=2, frameon=False, title=r'$R_0,\eta$',
               handlelength=1.0, columnspacing=.5, handletextpad=.5, borderaxespad=.4)
labels = ['Power mismatch (%)', 'Scale mismatch (%)', 'Ring mismatch (%)',
          'Phase mismatch (rad)', r'Aperture / $r_0$', r'Shift / $w$']
for ax, label in zip(axes, labels):
    ax.set(xlabel=label, ylabel=r'$\Delta_4/\Delta_{4,\,\mathrm{balanced}}$')
finalize_figure(fig, axs, height_in=5.6)
# Keep this two-column legend within its panel after the shared style is applied.
for label in axes[0].get_legend().get_texts():
    label.set_fontsize(7.5)
axes[0].get_legend().get_title().set_fontsize(8)
retain_lines(fig, 'Figure S3')
save_figure_set(fig, OUT / 'input_sensitivity')
plt.close(fig)

for p in sorted(DATA.iterdir()):
    audit['data'][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
(QA / 'si_figure_data_audit.json').write_text(json.dumps(audit, indent=2))
print('SI S2/S3 regenerated from retained data; scientific labels and legend clearance updated.')
