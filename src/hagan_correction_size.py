"""Size of the correction factor 1 + [...]T in Hagan's formula for beta = 1.

For beta = 1 the factor is 1 + A*T with A = rho*nu*alpha/4 + (2 - 3*rho**2)*nu**2/24.
We report |A*T| on the final Hagan data sets (inputs: columns T, alpha, nu, rho) and for the
parameter sets of Cases A, B and C of the report. Run from the repository root:
    python -m src.hagan_correction_size
Writes results/hagan_correction_size.json.
"""
import json

import numpy as np

from src.config import DATA_DIR, RESULTS_DIR, WINDOWS

CASES = {'A': (1.6316, 0.108266, 0.887903, -0.464053),
         'B': (540 / 365, 0.30, 0.60, -0.35),
         'C': (1.0, 0.12, 0.85, 0.45)}


def size(T, alpha, nu, rho):
    """Absolute size of the correction factor, |[rho*nu*alpha/4 + (2-3rho^2)nu^2/24] T|."""
    return np.abs((rho * nu * alpha / 4 + (2 - 3 * rho**2) * nu**2 / 24) * T)


def main():
    out = {}
    for window in WINDOWS:
        data = np.load(DATA_DIR / f'hagan_{window}.npz')
        for part in ('X_train', 'X_test'):
            X = data[part]
            c = size(X[:, 0], X[:, 1], X[:, 2], X[:, 3])
            out[f'{window}/{part}'] = dict(
                n=len(c), median=float(np.median(c)), p95=float(np.percentile(c, 95)),
                max=float(c.max()), share_below_1pct=float(np.mean(c < 0.01)),
                share_above_3pct=float(np.mean(c > 0.03)))
            r = out[f'{window}/{part}']
            print(f"{window} {part}: median {r['median']*100:.2f}%, p95 {r['p95']*100:.2f}%, "
                  f"max {r['max']*100:.2f}%, <1%: {r['share_below_1pct']*100:.1f}%, "
                  f">3%: {r['share_above_3pct']*100:.1f}%")
    for name, params in CASES.items():
        out[f'case_{name}'] = float(size(*params))
        print(f"Case {name}: {out[f'case_{name}']*100:.2f}%")
    (RESULTS_DIR / 'hagan_correction_size.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
