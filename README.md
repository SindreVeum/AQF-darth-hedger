# AQF-darth-hedger

Replication of McGhee (2018), *An ANN Representation of the SABR Stochastic Volatility Model*, with β = 1 and normalized forward F₀ = 1.

## Layout

- `src/` — pricing, dataset generation, ANN training and calibration.
- `AQF_S1_Hagan.ipynb` — Hagan dataset checks.
- `AQF_S2_ANN_Hagan.ipynb` — ANN results and calibration demonstration; training cells can launch training.
- `AQF_SABR_AllInOne.ipynb` — integration background and published-example checks.
- `figures/report_figures.py` — regenerate report figures and result tables without training.
- `data/datasets/` — Hagan and integration NPZ datasets.
- `data/models/` — trained weights and resumable checkpoints.
- `data/logs/` — per-model training logs and job output.
- `data/chunks/` — resumable integration-generation chunks.

All of `data/` is local and ignored by Git.
- `wiki/` — [project findings](wiki/index.md), [report](wiki/report/ann-section.md), [LaTeX](wiki/report/ann-section.tex), and [supporting research](wiki/research/README.md).

Research scripts, reference PDFs, historical audits and experiment outputs are consolidated in `wiki/research/`. Diagnostic CSVs and caches live in its ignored `results/` directory, including legacy CSV dataset exports. Report images remain in `wiki/report/figures/`.

## Setup and checks

```sh
uv sync
uv run python -m pytest -q
uv run python figures/report_figures.py
```

Figure regeneration requires the existing datasets and checkpoints in `data/`. It writes report images and Markdown tables to `wiki/report/figures/`, and diagnostic CSVs to `wiki/research/results/report/`. The LaTeX file is a section fragment, using figure filenames relative to the Overleaf document.

## Training

Both maturity windows, **[1D, 1Y] and [1Y, 2Y]**, have completed Hagan pretraining and integration fine-tuning at widths 250, 500, 750 and 1000, with raw and transformed inputs.

| Stage/window | Raw checkpoint | Transformed checkpoint |
| --- | --- | --- |
| Hagan, 1D–1Y | `ann_{N}.pt` | `ann_log_{N}.pt` |
| Integration, 1D–1Y | `ann_{N}_ln.pt` | `ann_log_{N}_ln.pt` |
| Hagan, 1Y–2Y | `ann_{N}_1y2y.pt` | `ann_log_{N}_1y2y.pt` |
| Integration, 1Y–2Y | `ann_{N}_ln_1y2y.pt` | `ann_log_{N}_ln_1y2y.pt` |

Checkpoints in the table above live in `data/models/`. Training uses `data/datasets/hagan_{window}.npz` and `data/datasets/ln_parity_{window}.npz`. Each has 250,000 training and 50,000 test smiles. Integration labels are saved in resumable chunks in `data/chunks/ln_parity_parts/`.

Generation and training are explicit, potentially lengthy operations:

```sh
uv run python -m src.hagan                         # Hagan datasets, both windows
uv run python -m src.integration --workers 4      # integration datasets, both windows
uv run python -m src.train_ann hagan_1d_1y --log-inputs
uv run python -m src.train_ann ln_1d_1y --log-inputs
```

`src.train_ann` takes one of the runs `hagan_1d_1y`, `hagan_1y_2y`, `ln_1d_1y`, `ln_1y_2y` (defined in `RUNS`), optionally followed by widths (default 250 500 750 1000). The `ln_*` runs fine-tune the matching Hagan network. Omit `--log-inputs` for raw inputs. Completed checkpoints are skipped; interrupted training resumes.

## Calibration

`src.calibration` fits `(alpha, nu_1M, rho)` using bounded SLSQP and IV least squares. The current interface supports [1D, 1Y] and defaults to transformed N=750. It interpolates ANN outputs using cubic IV against strike; alternative interpolation rules remain research experiments.

```python
from src.calibration import Quotes, calibrate, load_ann

quotes = Quotes(maturity=0.5, forward=100.0,
                strikes=[95, 100, 105], ivs=[0.22, 0.20, 0.19])
result = calibrate(quotes, "ann", load_ann())
```

Maturity is in years, IVs are decimals, and strikes are absolute. The opt-in synthetic benchmark tests the pipeline; it is not market-data evidence:

```sh
uv run python -m src.calibration
```

Outputs go to `wiki/research/results/calibration.csv`.
