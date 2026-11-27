"""Small deterministic FDM/LN/ANN comparison; no training or full sample generation.

Run from the repository root: python -m src.fdm_experiment
Writes results/fdm_pilot/{details.json,labels.csv,comparison.png}. Reruns recompute
every PDE; existing experiment outputs are replaced, datasets/models are untouched.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src import integration as ln
from src.config import DATA_DIR, MODEL_DIR, RESULTS_DIR
from src.fdm import sabr_cs_price
from src.hagan import hagan_iv, make_strikes

TOLERANCES_BP = dict(space=.5, time=.05, x_domain=.1, y_domain=.1)


def pilot_cases(random_per_sign=1, seed=20261126):
    """One published example, stratified existing test rows, and three domain corners."""
    if random_per_sign < 0 or int(random_per_sign) != random_per_sign:
        raise ValueError('random_per_sign must be a nonnegative integer.')
    k = np.array([.4419, .5152, .6007, .7004, .8166, .9520, 1.1100, 1.2941, 1.5088, 1.7591, 2.0509])
    cases = [dict(name='paper_2018', source='paper', window='1y_2y',
                  row=np.r_[1.6316, .108266, .887903, -.464053, k],
                  paper_fdm_iv=np.array([30.02, 26.78, 23.39, 19.81, 16.05, 12.23,
                                        10.53, 12.55, 15.23, 17.93, 20.56])/100)]
    rng = np.random.default_rng(seed)
    for window in ('1d_1y', '1y_2y'):
        with np.load(DATA_DIR / f'ln_parity_{window}.npz', allow_pickle=False) as d:
            X, y = d['X_test'], d['y_test']
            for sign in (-1, 1):
                pool = np.flatnonzero(np.sign(X[:, 3]) == sign)
                for i in rng.choice(pool, size=random_per_sign, replace=False):
                    cases.append(dict(name=f'{window}_test_{i}', source='test_dataset', window=window,
                                      row_index=int(i), row=X[i].copy(), saved_ln_iv=y[i].copy()))
    for name, T, alpha, rho in [('short_lowvol_negative', 1/365, .05, -.9),
                               ('short_lowvol_positive', 1/365, .05, .9),
                               ('long_highvol_positive', 2., .5, .9)]:
        nu = 4/np.sqrt(12*T)
        K = make_strikes(1., T, alpha, rho, nu)
        cases.append(dict(name=name, source='stress', window='1d_1y' if T <= 1 else '1y_2y',
                          row=np.r_[T, alpha, nu, rho, K]))
    return cases


def grid_runs(focus):
    """Independent changes: space, time and each domain; widths preserve internal nodes."""
    base = dict(nx=400, ny=200, nt=400, x_width=8., y_width=6., x_focus=focus, y_focus=1.)
    # Reach at least |x|=12 while keeping the old transformed mesh spacing.
    # A fixed +20% in transformed width gives unnecessarily huge domains at small focus.
    nx_wide = int(2*np.ceil(200*np.arcsinh(12/focus)/np.arcsinh(8/focus)))
    return {
        'coarse': dict(base, nx=200, ny=100), 'base': base,
        'fine': dict(base, nx=800, ny=400),
        'time_coarse': dict(base, nt=200), 'time_fine': dict(base, nt=800),
        'x_wide': dict(base, nx=nx_wide, x_width=focus*np.sinh(nx_wide/400*np.arcsinh(8/focus))),
        'y_wide': dict(base, ny=240, y_width=np.sinh(1.2*np.arcsinh(6))),
    }


def assess_convergence(runs):
    """Per-label numerical diagnostics, not a rigorous error bound against exact SABR."""
    base, fine = np.asarray(runs['base']['iv']), np.asarray(runs['fine']['iv'])
    changes = {
        'space': np.abs(fine-base)*1e4,
        'time': np.abs(np.asarray(runs['time_fine']['iv'])-base)*1e4,
        'x_domain': np.abs(np.asarray(runs['x_wide']['iv'])-base)*1e4,
        'y_domain': np.abs(np.asarray(runs['y_wide']['iv'])-base)*1e4,
    }
    status = np.full(len(fine), 'ok', dtype='<U120')
    finite = np.all([np.isfinite(r['iv']) for r in runs.values()], axis=0)
    for key, value in changes.items():
        for i in np.flatnonzero(value > TOLERANCES_BP[key]):
            status[i] = ('' if status[i] == 'ok' else status[i]+'|') + key+'_not_converged'
    status[~finite] = 'invalid_iv'
    return dict(iv=np.where(status == 'ok', fine, np.nan), iv_estimate=fine.copy(), status=status,
                changes_bp=changes,
                previous_space_change_bp=np.abs(base-np.asarray(runs['coarse']['iv']))*1e4,
                previous_time_change_bp=np.abs(base-np.asarray(runs['time_coarse']['iv']))*1e4)


def run_case(case, models):
    row = case['row']
    T, alpha, nu, rho = row[:4]
    K = row[4:]
    runs = {}
    # Adaptive focus resolves narrow short-expiry smiles; unchanged for each refinement.
    for name, settings in grid_runs(min(.5, alpha*np.sqrt(T))).items():
        start = perf_counter()
        try:
            values = sabr_cs_price(1., K, T, alpha, rho, nu,
                                   positive_rho_parity=True, **settings)
            prices = values['price']
            iv = ln.invert_otm(prices, K, T)
            error = None
        except (ArithmeticError, ValueError) as exc:
            prices, iv = np.full(len(K), np.nan), np.full(len(K), np.nan)
            error = f'{type(exc).__name__}: {exc}'
        runs[name] = dict(settings=settings, seconds=perf_counter()-start, price=prices, iv=iv, error=error)
    result = dict(case, runs=runs, **assess_convergence(runs))
    start = perf_counter()
    result['ln_iv'] = ln.label_row(row)
    result['ln_seconds'] = perf_counter()-start
    result['hagan_iv'] = hagan_iv(1., K, T, alpha, 1., rho, nu)
    result['ann'] = {}
    # Only canonical ten-strike inputs are given to the ANN. No spline is inserted into this comparison.
    if case['source'] != 'paper':
        for kind in ('raw', 'transformed'):
            start = perf_counter()
            pred = models[(case['window'], kind)].predict(row[None])[0]
            result['ann'][kind] = dict(iv=pred, seconds=perf_counter()-start)
    result['price_convention'] = 'put_call_parity' if rho > 0 else 'direct_otm'
    return result


def json_value(value):
    """Portable JSON: failed numerical results are null rather than non-standard NaN."""
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_outputs(results, output, metadata, complete=False):
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, completed_cases=len(results), complete=complete)
    (output/'details.json').write_text(json.dumps(json_value(dict(metadata=metadata, cases=results)),
                                                 indent=2, allow_nan=False), encoding='utf8')
    rows = []
    for c in results:
        T, alpha, nu, rho = c['row'][:4]
        for j, K in enumerate(c['row'][4:]):
            r = dict(case=c['name'], source=c['source'], window=c['window'],
                     T=T, alpha=alpha, nu=nu, rho=rho, strike_index=j+1, K=K,
                     price_convention=c['price_convention'], fdm_status=c['status'][j],
                     fdm_price=c['runs']['fine']['price'][j], fdm_iv=c['iv'][j],
                     fdm_iv_estimate=c['iv_estimate'][j], ln_iv=c['ln_iv'][j], hagan_iv=c['hagan_iv'][j],
                     ln_minus_fdm_bp=(c['ln_iv'][j]-c['iv'][j])*1e4)
            r.update({f'{key}_change_bp': values[j] for key, values in c['changes_bp'].items()})
            for kind in ('raw', 'transformed'):
                pred = c['ann'].get(kind, {}).get('iv', np.full(len(c['iv']), np.nan))[j]
                r[f'ann_{kind}_iv'] = pred
                r[f'ann_{kind}_minus_fdm_bp'] = (pred-c['iv'][j])*1e4
            rows.append(r)
    with (output/'labels.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_results(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots((len(results)+1)//2, 2, figsize=(12, 3*((len(results)+1)//2)), squeeze=False)
    for ax, c in zip(axes.flat, results):
        K = c['row'][4:]
        ax.plot(K, (c['ln_iv']-c['iv_estimate'])*1e4, 'o-', ms=3, label='LN - FDM estimate')
        for kind, p in c['ann'].items():
            ax.plot(K, (p['iv']-c['iv_estimate'])*1e4, '.--', label=f'ANN {kind} - FDM estimate')
        bad = c['status'] != 'ok'
        ax.scatter(K[bad], np.zeros(bad.sum()), marker='x', color='red', label='FDM check failed')
        ax.axhline(0, color='black', lw=.6)
        ax.set(title=c['name'], xlabel='K/F (log scale)', ylabel='IV difference (bp)', xscale='log')
        ax.legend(fontsize=6)
        ax.grid(alpha=.25)
    for ax in axes.flat[len(results):]:
        ax.set_visible(False)
    fig.suptitle('Same-strike pilot: estimates with failed convergence checks are explicitly marked')
    fig.tight_layout()
    fig.savefig(output/'comparison.png', dpi=150)
    plt.close(fig)


def main(random_per_sign=1, seed=20261126, output=RESULTS_DIR/'fdm_pilot'):
    import torch
    from src.model import SABRNet

    torch.set_num_threads(1)
    models, fingerprints = {}, {}
    for window, suffix in [('1d_1y', 'ln'), ('1y_2y', 'ln_1y2y')]:
        for kind, prefix in [('raw', 'ann'), ('transformed', 'ann_log')]:
            path = MODEL_DIR/f'{prefix}_750_{suffix}.pt'
            model = SABRNet(750, log_inputs=kind == 'transformed')
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state'])
            model.eval()
            model.predict(np.ones((1, 14)))  # untimed dispatch warm-up
            models[(window, kind)] = model
            fingerprints[str(path.relative_to(MODEL_DIR.parent.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('fdm.py', 'fdm_experiment.py', 'integration.py', 'hagan.py', 'model.py'):
        fingerprints[f'src/{name}'] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    cases = pilot_cases(random_per_sign, seed)
    metadata = dict(seed=seed, random_per_sign=random_per_sign, tolerances_bp=TOLERANCES_BP,
                    expected_cases=len(cases), case_names=[c['name'] for c in cases],
                    versions={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'torch')},
                    source_sha256=fingerprints,
                    limitation='Small pilot only; time/domain checks use the medium grid, not the finest grid.',
                    timing='Wall seconds; PDE timings include assembly/solve/IV inversion; ANN excludes checkpoint load.')
    output = Path(output)
    results = []
    for case in cases:
        print(f"Starting {case['name']}", flush=True)
        result = run_case(case, models)
        results.append(result)
        write_outputs(results, output, metadata)  # progress snapshot only; never reused as a numerical cache
        maxima = {key: float(np.nanmax(v)) for key, v in result['changes_bp'].items()}
        print(f"  accepted {np.sum(result['status']=='ok')}/{len(result['iv'])}, "
              f"max changes {maxima}, fine solve {result['runs']['fine']['seconds']:.2f}s", flush=True)
    plot_results(results, output)
    write_outputs(results, output, metadata, complete=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-per-sign', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261126)
    parser.add_argument('--output', type=Path, default=RESULTS_DIR/'fdm_pilot')
    args = parser.parse_args()
    main(args.random_per_sign, args.seed, args.output)
