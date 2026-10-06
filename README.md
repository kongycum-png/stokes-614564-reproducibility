# Manuscript 614564: reproducibility package

**Topological-order-dependent axial advance of a local Stokes domain in Poincaré circular Airy beams**

Youchao Kong, Weiwei Liu, Ze Chen, and Xiyuan Feng.

This version accompanies the October 2026 revision. It provides the scalar
propagation code, independently generated event coefficients, retained complex
channels, detector simulations, figure data, and extended Technical Appendix.
The authoritative main-matrix results are the source-tail-corrected
`data/principal_refined/` records. All retained event states, including missing
crossings and root-branch changes, are included.

## Start here

- [Reproduction instructions](REPRODUCE.md)
- [Evidence index](EVIDENCE_INDEX.md)
- [Extended Technical Appendix](technical_appendix/Technical_Appendix_614564.pdf)
- [Release downloads](../../releases/tag/v1.0.0)

The Git tree contains code, configuration, CSV/JSON results, figure inputs, and
the Technical Appendix. The release additionally contains a complete code/table
ZIP and independently extractable raw-field ZIPs. Extract the code/table
ZIP into an empty folder, then extract each raw-field ZIP **into that same
folder**. No binary concatenation is needed. See DOWNLOAD.md for verification.

`DISTRIBUTION_MANIFEST.json` records every file's relative path, byte size,
SHA-256 digest, and download part. `RELEASE_ASSETS.json` and `SHA256SUMS_ASSETS.txt`
identify the downloadable archives. The v1.0.0 tag fixes the original numerical code and records; the current
code/table archive includes the 6 October 2026 documentation and source-comment
updates. A DOI
has not been assigned.

## Evidence families

`code/` contains propagation, event extraction, analytic coefficients, and
detector models. `config/` records parameters, estimator rules, and random seeds.
`data/focused_refinement/` contains Q/T and boundary comparisons;
`data/focused_measurement/` contains stage events, count samples, covariance,
and noise ablations. `figure_data/` contains the inputs for the five main figures.
`residual_model/` retains the separate finite-radius analysis. The appendix uses
the independent equation, figure, and table prefix **TA**.

The `figures/` folder retains the layouts used during the numerical revision.
Final publication PDFs, corresponding PNG renderings, and the author-supplied
source panels for the current SI optical diagram are in the separate
`Publication_Figures_614564_20261006.zip` release asset. See
[Publication figures](PUBLICATION_FIGURES.md) for its checksum and contents.
The earlier editable `Figure_S1_source.pptx`, its vector export, and the
`proposed_experimental_layout.*` files remain historical scheme versions.
The proposed apparatus has not been built in this study.

This package contains simulations, not measured experimental data. Exact-kz
and matched unexpanded scalar R–S are implementation checks of the same boundary
problem; Fresnel versus exact-kz measures an approximation difference.

## Rights

Source code is released under the MIT License. Data and figures are released
under Creative Commons Attribution 4.0 International (CC BY 4.0). See COPYING.md
and licenses/ for the scope and full terms. Technical Appendix prose and theory
notes retain author copyright; no additional license is granted for those texts.
