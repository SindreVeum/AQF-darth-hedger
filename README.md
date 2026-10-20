# AQF-darth-hedger

Replication of McGhee (2018), *An ANN Representation of the SABR Stochastic Volatility Model* (β = 1, F₀ = 1).

```
src/
  hagan.py        # Hagan β=1 implied vol, strike grid (Tables 3-4), 250k grid+random training set (§4)
  model.py        # 14 → Dense(N, softplus) → 10, Adam + MSE training, cubic-spline smile (§4.3)
  train_ann.py    # train several hidden sizes in parallel, resumable checkpoints
AQF_S1_Hagan.ipynb      # dataset check against the paper
AQF_S2_ANN_Hagan.ipynb  # ANN results vs. Figures 2-6
data/                   # generated datasets and checkpoints (git-ignored)
```

```bash
uv sync
uv run python -m src.hagan                 # -> data/hagan_1d_1y.npz (+ CSVs)
uv run python -m src.train_ann 250 500 750 1000
```

Only the T ∈ [1D, 1Y] network trained on the Hagan approximation is done (Figures 2-6).
TODO: SABR integration scheme labels and the second T ∈ [1Y, 2Y] network (§5.2-5.3, Table 5, Figure 7).
