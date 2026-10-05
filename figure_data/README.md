# Main-figure data provenance

These files are unchanged copies of the retained Codex evidence for Manuscript
614564. They were copied from
`<original-local-root>/Documents/光学模拟/poincare_rs_aligned/revision_614564/data/` on
2026-09-21. `SHA256SUMS.txt` records the copied-byte hashes.

- `principal_refined/R24_eta0.7_*` supplies the stored complex channels,
  observable curves, event identities, and actual-zero trajectories used in
  Figures 1 and 2.
- `focused_refinement/qt_*` supplies the Q/T event comparison derived from the
  600-field matrix.
- `final_analysis/analytic_predictions.csv` supplies the independently
  generated, non-fitted A/B/C predictions used in Figure 3.
- `focused_refinement/four_boundary_representative.csv`,
  `actual_proxy_asymptotic_comparison.csv`, and
  `root_audit/R24_eta0.9_desingularized.npz` supply Figure 4.
- `focused_measurement/*` supplies the frozen-protocol count traces, A--D
  event stages, validation samples, and noise ablations used in Figure 5.

The plotting script reads only the copied files in this directory. It does not
read fitted propagation results, alter event identities, or replace branch
failures by interpolation.
