# NanoporeOS code and numerical data

This repository contains the Python source, tests, numerical protocols, and machine-readable outputs for a steady axisymmetric Poisson–Nernst–Planck simulation workflow.

## Contents

- `src/`: simulation package and command-line interface.
- `scripts/`, `benchmarks/`: numerical workflows, analysis, and verification code.
- `config/`: model parameters and frozen protocols.
- `results/`: saved fields, solver outputs, and analysis summaries in machine-readable formats.
- `results/derived_tables/`: numeric CSV outputs.
- `results/figure_source_data/`: numeric inputs for visualizations; rendered figure files are not stored here.

## Reproduction

The tested environment is Python 3.12 on Windows. Install the pinned environment from `requirements-lock-windows-py312.txt`, then install the package in editable mode with its verification extra. Run software tests with `python -m unittest discover -s tests -v`. Analysis scripts use the saved numerical outputs; simulation scripts can generate new outputs locally.

No manuscript document or rendered figure files are included in this repository. API credentials and author contact configuration are also excluded.


## License and archive

This repository is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0), consistent with the archived release at Zenodo: https://doi.org/10.5281/zenodo.24491826.
