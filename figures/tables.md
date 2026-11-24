# Generated tables (figures/report_figures.py)

## Input squeeze after min–max scaling

| input | raw: share in lowest 5% of range | transformed: share in lowest 5% of range |
| --- | --- | --- |
| Maturity | 5.0% | 0.9% |
| Vol of vol | 63.6% | 7.0% |
| Strikes (scaled per column, then pooled) | 15.1% | 3.4% |

## Hagan stage: ANN vs Hagan formula, 1D–1Y test set (IV bp)

| inputs | N | MAE | RMSE | p99 | max | epochs |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 250 | 4.72 | 7.04 | 25.06 | 153 | 12,738 |
| raw | 500 | 3.60 | 5.39 | 18.99 | 133 | 15,463 |
| raw | 750 | 3.78 | 5.63 | 20.07 | 130 | 11,788 |
| raw | 1000 | 3.55 | 5.27 | 18.48 | 131 | 13,543 |
| transformed | 250 | 1.32 | 2.14 | 8.29 | 40.41 | 20,000 |
| transformed | 500 | 1.16 | 1.90 | 7.45 | 41.90 | 19,817 |
| transformed | 750 | 1.20 | 1.94 | 7.61 | 38.71 | 20,000 |
| transformed | 1000 | 1.29 | 2.18 | 8.70 | 53.16 | 15,265 |

## Fine-tuned: ANN vs integration, 1D–1Y test set (IV bp)

| inputs | N | MAE | RMSE | p99 | max | epochs |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 250 | 4.07 | 6.03 | 21.25 | 140 | 7,063 |
| raw | 500 | 3.21 | 4.76 | 16.57 | 141 | 6,089 |
| raw | 750 | 3.23 | 4.83 | 17.07 | 106 | 5,618 |
| raw | 1000 | 3.17 | 4.68 | 16.41 | 121 | 6,229 |
| transformed | 250 | 1.13 | 1.86 | 7.23 | 53.11 | 9,594 |
| transformed | 500 | 1.06 | 1.78 | 7.03 | 46.09 | 7,733 |
| transformed | 750 | 1.04 | 1.74 | 7.04 | 44.69 | 8,361 |
| transformed | 1000 | 1.16 | 1.99 | 7.94 | 42.25 | 6,935 |

Reference row, Hagan formula vs integration on the same labels: MAE 83.2, RMSE 185.9, p99 835, max 2375.

## Hagan stage: 1Y–2Y (IV bp)

| inputs | N | MAE | RMSE | p99 | max | epochs |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 250 | 1.34 | 2.05 | 7.76 | 24.38 | 18,784 |
| raw | 500 | 1.28 | 1.97 | 7.13 | 47.12 | 20,000 |
| raw | 750 | 1.26 | 1.94 | 7.28 | 21.72 | 20,000 |
| raw | 1000 | 1.11 | 1.74 | 6.64 | 41.58 | 20,000 |
| transformed | 250 | 1.17 | 1.85 | 7.24 | 27.45 | 20,000 |
| transformed | 500 | 1.22 | 1.95 | 7.52 | 33.10 | 20,000 |
| transformed | 750 | 1.23 | 1.95 | 7.59 | 41.43 | 20,000 |
| transformed | 1000 | 1.17 | 1.93 | 7.80 | 25.08 | 17,959 |

## Fine-tuned: 1Y–2Y (IV bp), 50,000 complete test smiles

| inputs | N | MAE | RMSE | p99 | max | epochs |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 250 | 1.17 | 1.77 | 6.53 | 31.74 | 10,800 |
| raw | 500 | 1.09 | 1.64 | 5.83 | 48.58 | 8,908 |
| raw | 750 | 1.04 | 1.59 | 5.93 | 27.63 | 10,947 |
| raw | 1000 | 1.00 | 1.54 | 5.71 | 39.00 | 8,010 |
| transformed | 250 | 1.03 | 1.68 | 6.53 | 48.08 | 9,691 |
| transformed | 500 | 1.09 | 1.77 | 6.90 | 39.29 | 9,636 |
| transformed | 750 | 1.09 | 1.76 | 6.94 | 41.54 | 11,053 |
| transformed | 1000 | 1.02 | 1.69 | 6.66 | 48.63 | 16,665 |

Hagan vs integration, same test inputs: {'MAE': np.float64(104.39580206262225), 'RMSE': np.float64(236.60266518452661), 'p99': np.float64(1074.5811145862024), 'max': np.float64(2760.578460398855)}.
