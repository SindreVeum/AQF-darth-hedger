"""Case C of the report: the positive-correlation smile on McGhee (2011) slide 25.

Run python -m src.fdm_case_c (about five minutes).
Parameters: F=1, T=1, alpha=12%, nu=85%, rho=45%, beta=1, ten strikes from 0.725 to 1.55. The slide
does not state the forward; F=1 matches the strike grid. The published column is the slide's Integration
row, rounded to 0.01 percent; the slide's 2F PDE row is not used.

Compares Hagan, our integration (put-call parity for K >= F, see src/integration.py) and the direct
call/put Craig-Sneyd FDM on the fine grid, and repeats the grid, time and domain checks of the report
(tolerances in src/fdm_experiment.py). The medium and fine grids are those of Case B in src/fdm_paper.py,
(640,320,400) and (1280,640,400); the default (400,200,400)/(800,400,400) pair changes the implied
volatilities by 0.5013 IV bp, just above the 0.5 IV bp tolerance. Writes fdm_case_c.json and fdm_case_c.csv.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.config import RESULTS_DIR
from src.fdm import sabr_cs_price
from src.fdm_experiment import TOLERANCES_BP, json_value
from src.fdm_refinement import expand_domain
from src.hagan import hagan_iv
from src.integration import invert_otm, quadrature_price

F, T, ALPHA, NU, RHO = 1., 1., .12, .85, .45
K = np.array([.725, .8167, .9083, 1., 1.0917, 1.1833, 1.275, 1.3667, 1.4583, 1.55])
PUBLISHED_IV = np.array([14.85, 12.82, 11.83, 12.67, 14.59, 16.65, 18.59, 20.38, 22.03, 23.55]) / 100
MEDIUM = dict(nx=640, ny=320, nt=400, x_width=8., y_width=6., x_focus=.5, y_focus=1.,
              fit_forward=False, positive_rho_parity=False)


def triplet(x):
    """MAE / RMSE / maximum absolute value, as in the report tables."""
    x = np.abs(np.asarray(x, float))
    return dict(mae=x.mean(), rmse=np.sqrt((x**2).mean()), max=x.max(), at_K=K[x.argmax()])


def fdm(settings):
    price = sabr_cs_price(F, K, T, ALPHA, RHO, NU, **settings)['price']
    return price, invert_otm(price, K, T)


def main(output=RESULTS_DIR):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    runs = dict(coarse=dict(MEDIUM, nx=MEDIUM['nx'] // 2, ny=MEDIUM['ny'] // 2),
                medium=MEDIUM,
                fine=dict(MEDIUM, nx=2 * MEDIUM['nx'], ny=2 * MEDIUM['ny']),
                time_fine=dict(MEDIUM, nt=800),
                x_wide=expand_domain(MEDIUM, 'x', 12.),
                y_wide=expand_domain(MEDIUM, 'y', 9.9))
    price, iv = {}, {}
    for name, settings in runs.items():
        price[name], iv[name] = fdm(settings)
        print(f'{name:9s} nx={settings["nx"]:5d} ny={settings["ny"]:4d} nt={settings["nt"]:4d} done', flush=True)

    hagan = hagan_iv(F, K, T, ALPHA, 1., RHO, NU)
    p_int = quadrature_price(F, K, T, ALPHA, RHO, NU)
    iv_int = invert_otm(p_int, K, T)
    bp = 1e4

    diff = (iv_int - iv['fine']) * bp
    parity = K >= F
    others = np.delete(np.abs(diff[parity]), np.abs(diff[parity]).argmax())
    stats = dict(
        integration_vs_published_iv_bp=triplet((iv_int - PUBLISHED_IV) * bp),
        integration_vs_fdm_iv_bp=triplet(diff),
        integration_vs_fdm_price_bp=triplet((p_int - price['fine']) * bp),
        integration_vs_fdm_parity_strikes=dict(
            n=int(parity.sum()), max=np.abs(diff[parity]).max(), at_K=K[parity][np.abs(diff[parity]).argmax()],
            others_max=others.max(), signed_iv_bp=diff[parity]),
        hagan_vs_integration_iv_bp=triplet((hagan - iv_int) * bp))

    def change(a, b):
        return float(np.max(np.abs(iv[a] - iv[b])) * bp)

    checks = dict(space=change('medium', 'fine'), time=change('medium', 'time_fine'),
                  x_domain=change('medium', 'x_wide'), y_domain=change('medium', 'y_wide'),
                  x_half_width=runs['x_wide']['x_width'], y_half_width=runs['y_wide']['y_width'],
                  space_coarse_to_medium=change('coarse', 'medium'))
    checks['space_change_ratio'] = checks['space_coarse_to_medium'] / checks['space']
    checks['within_tolerance'] = {k: checks[k] <= TOLERANCES_BP[k] for k in TOLERANCES_BP}

    result = dict(parameters=dict(F=F, T=T, alpha=ALPHA, nu=NU, rho=RHO), K=K,
                  iv=dict(hagan=hagan, integration=iv_int, fdm=iv['fine'], published_integration=PUBLISHED_IV),
                  stats=stats, checks=checks, tolerances_bp=TOLERANCES_BP, settings=runs,
                  note='Time and domain checks use the medium grid; the fdm column is the fine grid. '
                       'Differences are in IV bp or price bp (undiscounted, F=1).')
    (output / 'fdm_case_c.json').write_text(json.dumps(json_value(result), indent=1), encoding='utf8')
    with open(output / 'fdm_case_c.csv', 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['K', 'hagan_pct', 'integration_pct', 'fdm_pct', 'published_integration_pct'])
        for row in zip(K, hagan, iv_int, iv['fine'], PUBLISHED_IV):
            writer.writerow([row[0]] + [f'{100 * v:.4f}' for v in row[1:]])

    print(json.dumps(json_value(dict(stats=stats, checks=checks)), indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    main(args.output)
