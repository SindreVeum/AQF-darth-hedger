# AQF-darth-hedger

Replication of McGhee (2018), *An ANN Representation of the SABR Stochastic Volatility Model* (β = 1, F₀ = 1).

```
sabr_ann/
  hagan.py        # Hagan β=1 approximation (pre-training labels)
  integration.py  # McGhee integration scheme (target labels)  TODO
  sampling.py     # strike grid (eq. 46) + input sampling (§4.1)
  model.py        # 14 → Dense(N, softplus) → 10               TODO
experiments/      # one script per paper figure/table
tests/            # paper numbers as the answer key
data/             # generated datasets (git-ignored)
```

```bash
uv sync
uv run pytest
```
