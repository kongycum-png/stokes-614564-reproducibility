# Evidence index

| Compact SI | Extended Technical Appendix | Primary retained records |
|---|---|---|
| 1. Polarization, Q/T, transport | Secs. 1–2; Figs. TA1–TA2; Table TA1 | data/focused_refinement/qt_paired_events.csv, qt_double_ratio_curves.csv, qt_double_ratio_summary.csv; data/physics/; code/observables.py |
| 2. Full amplitude and event recursion | Sec. 3; Table TA2 | code/formal_asymptotics.py, formal_event_coefficients.py, generate_event_coefficients.py; data/focused_refinement/qt_formal_coefficients.csv |
| 3. Boundaries and general P | Secs. 4–7, 9–10; Table TA4 | data/focused_refinement/four_boundary_original_radii.csv, boundary_20pct_stress_representative.csv, actual_proxy_asymptotic_comparison.csv; data/final_analysis/general_P_predictions.csv; data/adaptive_roots/, data/root_audit/ |
| 4. Numerical/model/input limits | Secs. 8–12; Table TA3; Fig. TA3 | data/final_analysis/final_convergence.csv, original36_tail_corrections.csv, Maxwell_summary.csv, input_tolerances.csv; data/analysis_refined/model_comparison.csv; data/convergence_final/, data/maxwell/ |
| 5. Counts and failure diagnosis | Sec. 13; Tables TA5–TA7; Figs. TA4–TA5 | data/focused_measurement/measurement_stage_pairs.csv, measurement_failure_ablations.csv, measurement_failure_noise_split.csv, per-scenario JSON and samples; data/measurement/; config/MEASUREMENT_FOCUSED_PROTOCOL.json |
| 6. Optical proposal | Sec. 13; Fig. TA6 | figures/Figure_S1_source.pptx; figures/Figure_S1_optical_layout.pdf (current SI); Fig. TA6 retains the earlier schematic |
| 7. Finite-radius residual | Sec. 14; Fig. TA7; Tables TA8–TA12 | residual_model/data/analytic_coefficients_N14.json, radius_predictions.csv, event_continuation.csv, high_precision.json, direct_check.json, direct_events.json, coefficient_decomposition.json, minima.json, minimum_derivatives.json; residual_model/theory/ |

All six original radii, both apodizations and three comparison orders remain in compact SI Table S2. Full single-order events and root status fields are retained in CSV/JSON.

The full 600-beam inventory is data/analysis_refined/principal_all_events.csv and the principal_refined job records. Fields retain complex channels; the measurement starts from channel intensities and raw counts.

The optical arrangement is proposed, not an experimental result. Exact-kz versus matched unexpanded scalar R-S agreement checks implementation; Fresnel versus exact-kz is a model comparison.
