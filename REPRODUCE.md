# Reproduction

Use a working copy, Python 3.9 or later and requirements-science.txt. The retained environment is residual_model/environment.json. Use one process and one BLAS/OpenMP thread; the main-run memory target is 8 GiB.

From the repository root:

    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    PYTHONPATH=code python3 code/generate_event_coefficients.py --m 4 --eta 0.7 --observable Q --gamma 0.5 --points 25 --nx 1201 --output /tmp/Q_m4_eta07.json

This command was actually run from the assembled repository. It reads no propagation-fit data. The resulting coefficients are approximately (9.21e-12, 7.9177839090, -8.3269932807, 15.7103148247) for epsilon through epsilon^4. Its machine-readable output is included in the delivery audit.

## Reuse corrected fields and curves

After extracting the code/table archive and all raw-field archive parts, these existing commands regenerate postprocessing and figures in a working copy:

    PYTHONPATH=code python3 code/focused_qt_boundary_analysis.py
    python3 figure_scripts/draw_narrative_result_figures.py

The first reads principal_refined curves and finds each observable's own first peak and rising crossing. It writes data/focused_refinement/ without propagating. The figure command writes figures/. The current optical proposal is edited in figures/Figure_S1_source.pptx and exported through a native presentation application; the legacy Python optical-layout script is not its source. These full regeneration commands were not rerun solely for this packaging round; retained inputs and outputs have checksums.

## Fresh main-matrix propagation

The historical workflow first generates data/principal/ with code/run_principal.py, then restores the omitted source tail coherently with code/refine_principal_tail.py. The latter reads baseline/tail_correction_check.json; its validation protocol is in code/check_tail_correction.py. Delivered authoritative results are in data/principal_refined/; the corrected cutoff is in config/PRINCIPAL_REFINEMENT.json.

A fresh 600-beam run was not repeated for this delivery. Run it in an empty working copy, following the archived configurations and convergence checks. Do not promote the initial truncated-source results.

## Residual module

The residual module keeps its own imports and outputs:

    cd residual_model
    RESEARCH_PYTHON=python3 sh RUN_REPRODUCTION.sh

This complete deterministic workflow includes direct quadrature and high-precision checks and rewrites its local outputs. Earlier execution records are in logs/; the whole workflow was not rerun during SI compaction.

## Documents and versions

The current journal files are manuscript.pdf (16 pages),
Supplementary_Material.pdf (12 pages), and response_to_reviewers.pdf (14 pages),
all compiled from the corresponding October 2026 LaTeX sources without line
numbers. They are supplied through the journal, not duplicated as draft
manuscripts in this public research package. Response locations use that current
manuscript/SI pagination. This supersedes the old 20-page Word-export pagination.

The extended appendix is included here. Compile
technical_appendix/latex_source/supplement/supplement.tex from its source folder;
its dependencies are in sibling figures/ and manuscript/references.bib. The
supplied appendix PDF is compiled with bundled Tectonic. Its extended numerical
tables retain their own TA numbering and are not compact SI tables.

No full propagation rerun was performed solely to prepare the public release.
The release audit distinguishes the new coefficient smoke check and archive
validation from the retained original calculation logs. In public log copies,
the original absolute user-home prefix is redacted; numerical records and
complex arrays are otherwise preserved.
