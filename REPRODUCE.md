# Reproduction

Use a working copy, Python 3.9 or later and requirements-science.txt. The retained environment is residual_model/environment.json. Use one process and one BLAS/OpenMP thread; the main-run memory target is 8 GiB.

From the repository root:

    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    PYTHONPATH=code python3 code/generate_event_coefficients.py --m 4 --eta 0.7 --observable Q --gamma 0.5 --points 25 --nx 1201 --output /tmp/Q_m4_eta07.json

This command generates the analytic event coefficients without reading propagation-fit data. The coefficients are approximately (9.21e-12, 7.9177839090, -8.3269932807, 15.7103148247) for epsilon through epsilon^4; RELEASE_CHECKS.json records the coefficient check.

## Reuse corrected fields and curves

After extracting the code/table archive and all raw-field archive parts, these existing commands regenerate postprocessing and figures in a working copy:

    PYTHONPATH=code python3 code/focused_qt_boundary_analysis.py
    python3 figure_scripts/draw_narrative_result_figures.py

The first command reads principal_refined curves, finds each observable's own first peak and rising crossing, and writes data/focused_refinement/. The second writes figures/ using the retained plotting layouts. The final publication figures are supplied separately in Publication_Figures_614564_20261006.zip. Its publication_figures/supporting/S1_panels/ directory contains the three author-supplied PNG panels used in the current SI optical diagram. figures/Figure_S1_source.pptx retains an earlier editable scheme.

## Fresh main-matrix propagation

The historical workflow first generates data/principal/ with code/run_principal.py, then restores the omitted source tail coherently with code/refine_principal_tail.py. The latter reads baseline/tail_correction_check.json; its validation protocol is in code/check_tail_correction.py. Delivered authoritative results are in data/principal_refined/; the corrected cutoff is in config/PRINCIPAL_REFINEMENT.json.

Run fresh propagation in an empty working copy, following the archived configurations and convergence checks. The initial truncated-source results are historical; use the corrected principal_refined records for comparison.

## Residual module

The residual module keeps its own imports and outputs:

    cd residual_model
    RESEARCH_PYTHON=python3 sh RUN_REPRODUCTION.sh

This deterministic workflow includes direct quadrature and high-precision checks and rewrites its local outputs. Its retained execution records are in residual_model/logs/.

## Documents and versions

The October 2026 journal manuscript, compact Supporting Information, and response letter are supplied through the journal. This research package contains the extended evidence rather than draft copies of those documents. PUBLICATION_FIGURES.md identifies the dated archive of final figure layouts.

The extended appendix is included here. Compile
technical_appendix/latex_source/supplement/supplement.tex from its source folder;
its dependencies are in sibling figures/ and manuscript/references.bib. A
compiled appendix PDF is also supplied. Its extended numerical
tables retain their own TA numbering and are not compact SI tables.

The release reuses the retained propagation and detector results with their
run logs. RELEASE_CHECKS.json contains the recorded coefficient smoke check.
The original absolute user-home prefix is redacted in public log copies.
Numerical records and complex arrays are preserved.
