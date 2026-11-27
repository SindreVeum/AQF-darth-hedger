"""Bounded diagnostic of fixed-log-forward payoff alignment, not a production solver.

This hypothesis is NOT a claim about McGhee's unpublished implementation. Reuse
the current Craig-Sneyd solver in memory with guarded replacements of its payoff
and readout only; no operator or time-step stage is changed. Evolve one direct
OTM payoff per strike on a common uniform log(F) grid, evaluating F=1 at a node.
"""

import hashlib
import inspect
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from src import fdm
from src.integration import black_price, invert_otm


def fixed_grid_solver(average):
    source = inspect.getsource(fdm.sabr_cs_price)
    start = '    # Only the explicit legacy convention reconstructs a call from a put.'
    end = '\n    drift = 1.'
    readout = '    values = CubicSpline(x, U[ny // 2], axis=0)(q) * K[:, None]'
    assert source.count(start) == source.count(end) == source.count(readout) == 1
    a, b = source.index(start), source.index(end)
    payoff = '''    if F != 1. or x_focus is not None or fit_forward or positive_rho_parity:
        raise ValueError('Diagnostic requires F=1, uniform X, and original direct stencil.')
    n_payoffs = K.size
    is_put = K < F
    payoff = np.where(is_put[None, :], np.maximum(K[None, :] - np.exp(x[:, None]), 0.),
                      np.maximum(np.exp(x[:, None]) - K[None, :], 0.))
    if AVERAGE:
        for j, k in enumerate(K):
            kink = np.log(k)
            i = int(np.argmin(np.abs(x - kink)))
            left, right = .5 * (x[i-1] + x[i]), .5 * (x[i] + x[i+1])
            if not left <= kink <= right:
                raise ArithmeticError('Kink is not in the selected cell.')
            # Exact integral of the payoff across the sole nonsmooth cell.
            if is_put[j]:
                payoff[i, j] = (k * (kink-left) - (k-np.exp(left))) / (right-left)
            else:
                payoff[i, j] = (np.exp(right)-k - k*(right-kink)) / (right-left)
    U = np.broadcast_to(payoff, (ny+1, nx+1, n_payoffs)).copy()
'''
    source = source[:a] + payoff + source[b:]
    source = source[:source.index(readout)] + '''    prices = U[ny // 2, nx // 2].copy()
    if not np.isfinite(prices).all():
        raise FloatingPointError('Nonfinite fixed-grid diagnostic price.')
    return {'price': prices, 'dx': dx, 'dy': dy, 'dt': dt}
'''
    namespace = dict(fdm.__dict__, AVERAGE=average)
    exec(compile(source, '<fixed-logF diagnostic>', 'exec'), namespace)
    return namespace['sabr_cs_price']


def main():
    root = Path(__file__).resolve().parents[2]
    baseline = json.loads((root / 'results/fdm_paper_direct/figure7.json').read_text())
    strikes = np.asarray(baseline['K'])
    own_iv = np.asarray(baseline['iv_estimate'])
    paper_iv = np.asarray(baseline['paper_fdm_iv'])
    p = baseline['parameters']
    out = root / 'results/fdm_diagnosis/fixed_grid_check.json'
    report = {
        'hypothesis_only': 'A common log F grid lets payoff kinks move relative to nodes. This is not a claim about the paper code.',
        'operator_reuse': 'Guarded in-memory payoff/readout replacements; same CS stages and spatial operators as src.fdm.',
        'fdm_source_sha256': hashlib.sha256(Path(fdm.__file__).read_bytes()).hexdigest(),
        'parameters': p,
        'settings': {'ny': 200, 'nt': 100, 'x_width': 8., 'y_width': 6., 'x_focus': None, 'y_focus': 1.},
        'limits': 'Fixed-grid plus uniform-X experiments vary several choices relative to the saved sinh-grid baseline; they do not isolate or reconstruct author settings.',
        'runs': [],
    }
    if '--resume' in sys.argv:
        prior = json.loads(out.read_text(encoding='utf-8'))
        if prior['fdm_source_sha256'] != report['fdm_source_sha256']:
            raise ValueError('Cannot resume against a changed production solver.')
        report = prior

    def save():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    def run(nx, average, indices, nu):
        if any(r['nx'] == nx and r['average_kink_cell'] == average
               and r['nu'] == nu and r['indices_zero_based'] == indices.tolist()
               for r in report['runs']):
            return
        k = strikes[indices]
        settings = dict(report['settings'], nx=nx)
        t0 = perf_counter()
        result = fixed_grid_solver(average)(1., k, p['T'], p['alpha'], p['rho'], nu, **settings)
        elapsed = perf_counter()-t0
        iv = invert_otm(result['price'], k, p['T'])
        if not np.isfinite(iv).all():
            raise ArithmeticError('Nonfinite diagnostic IV.')
        item = {'nx': nx, 'average_kink_cell': average, 'nu': nu,
                'indices_zero_based': indices.tolist(), 'K': k.tolist(),
                'seconds': elapsed, 'price': result['price'].tolist(), 'iv': iv.tolist()}
        if nu == 0.:
            exact = np.where(k<1., black_price(1., k, p['T'], p['alpha'], 'put'),
                             black_price(1., k, p['T'], p['alpha'], 'call'))
            item['black_price_error_bp'] = ((result['price']-exact)*1e4).tolist()
            item['black_iv_error_bp'] = ((iv-p['alpha'])*1e4).tolist()
        else:
            own_error = (iv-own_iv[indices])*1e4
            paper_error = (iv-paper_iv[indices])*1e4
            item.update(own_baseline_error_iv_bp=own_error.tolist(),
                        paper_error_iv_bp=paper_error.tolist(),
                        own_baseline_mae_iv_bp=float(np.mean(abs(own_error))),
                        own_baseline_max_iv_bp=float(np.max(abs(own_error))),
                        paper_mae_iv_bp=float(np.mean(abs(paper_error))),
                        paper_max_iv_bp=float(np.max(abs(paper_error))))
        report['runs'].append(item)
        save()
        summary = {key: item[key] for key in ('nx', 'average_kink_cell', 'nu', 'seconds')}
        for key in ('paper_mae_iv_bp', 'paper_max_iv_bp', 'own_baseline_max_iv_bp'):
            if key in item:
                summary[key] = item[key]
        if nu == 0.:
            summary['black_max_price_bp'] = float(np.max(abs(np.array(item['black_price_error_bp']))))
            summary['black_max_iv_bp'] = float(np.max(abs(np.array(item['black_iv_error_bp']))))
        print(json.dumps(summary), flush=True)

    key = np.array([0, 10, 12, 19])
    # Controls use the same key strikes, rather than carefully aligned strikes.
    for nx in (400, 800):
        for average in (False, True):
            run(nx, average, key, 0.)
    run(400, False, np.arange(len(strikes)), p['nu'])
    run(400, True, key, p['nu'])
    if '--skip-fine' not in sys.argv:
        for average in (False, True):
            run(800, average, key, p['nu'])


if __name__ == '__main__':
    main()
