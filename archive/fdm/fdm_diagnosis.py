"""Targeted Figure 7 grid/domain/interpolation diagnostics; no production changes.

Run python -m archive.fdm.fdm_diagnosis. The nominal 400x200x100 configuration interprets
the paper's unlabeled dimensions as X/Y/time; this is a hypothesis, not a claim
to know the author's grid. Every variant uses direct call and put evolution.
"""

import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.interpolate import CubicSpline

from src import fdm
from src.config import RESULTS_DIR
from src.fdm_experiment import json_value
from src.fdm_paper import metrics, otm_prices
from src.integration import invert_otm


OUTPUT = RESULTS_DIR / 'fdm_diagnosis'


def contracted_grid(base, axis, target):
    """Move endpoints inward while keeping the original interior grid nodes."""
    n, width, focus = (base[f'n{axis}'], base[f'{axis}_width'], base[f'{axis}_focus'])
    spacing = 2*np.arcsinh(width/focus)/n
    count = 2*int(np.round(np.arcsinh(target/focus)/spacing))
    return dict(base, **{f'n{axis}': count,
                        f'{axis}_width': focus*np.sinh(count*spacing/2)})


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = RESULTS_DIR/'fdm_paper_direct/figure7.json'
    c = json.loads(source.read_text(encoding='utf8'))
    K, p = np.asarray(c['K']), c['parameters']
    reference, paper = np.array(c['iv_estimate']), np.array(c['paper_fdm_iv'])
    base = dict(nx=400, ny=200, nt=100, x_width=8., y_width=6.,
                x_focus=.5, y_focus=1., fit_forward=False, positive_rho_parity=False)
    variants = dict(nominal_sinh=base,
                    uniform_log=dict(base, x_focus=None, y_focus=None),
                    contracted_x3=contracted_grid(base, 'x', 3.),
                    contracted_y3=contracted_grid(base, 'y', 3.),
                    contracted_y1p5=contracted_grid(base, 'y', 1.5))
    results = {}
    for name, settings in variants.items():
        captured = {}

        def capture(x, values, axis):
            # Observe the already-computed PDE slice without changing its values.
            captured.update(x=x.copy(), values=values.copy())
            return CubicSpline(x, values, axis=axis)

        print(f'{name}: {settings}', flush=True)
        start = perf_counter()
        with patch.object(fdm, 'CubicSpline', capture):
            result = fdm.sabr_cs_price(1., K, **p, **settings)
        cubic_iv = invert_otm(result['price'], K, p['T'])
        q = np.log(1/K)
        linear = np.column_stack([np.interp(q, captured['x'], captured['values'][:, j])
                                  for j in range(2)])*K[:, None]
        linear_price = np.where(K<1, linear[:, 1], linear[:, 0])
        linear_iv = invert_otm(linear_price, K, p['T'])
        results[name] = dict(settings=settings, seconds=perf_counter()-start,
                             cubic_price=result['price'], cubic_iv=cubic_iv,
                             linear_price=linear_price, linear_iv=linear_iv,
                             cubic_vs_fine_iv=metrics(cubic_iv, reference, K),
                             cubic_vs_paper_iv=metrics(cubic_iv, paper, K),
                             linear_vs_paper_iv=metrics(linear_iv, paper, K),
                             linear_minus_cubic_iv_bp=(linear_iv-cubic_iv)*1e4)
        baseline = np.asarray(results['nominal_sinh']['cubic_iv'])
        results[name]['change_from_nominal_iv'] = metrics(cubic_iv, baseline, K)
        (OUTPUT/'grid_variants.json').write_text(json.dumps(json_value(dict(
            complete=False, parameters=p, K=K, baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            variants=results)), indent=2, allow_nan=False), encoding='utf8')
        print(f'  {results[name]["seconds"]:.2f}s; max difference vs fine '
              f'{results[name]["cubic_vs_fine_iv"]["max_abs_bp"]:.4f} IV bp', flush=True)
    result = dict(complete=True, parameters=p, K=K, fine_iv=reference, paper_iv=paper,
                  variants=results, solver_sha256=hashlib.sha256(Path(fdm.__file__).read_bytes()).hexdigest(),
                  baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  limitation='Hypothesis tests of numerical settings, not recovered author settings. '
                             'Contracted domains preserve remaining interior nodes; interpolation '
                             'alternatives use the identical evolved surface. No production solver edits.')
    (OUTPUT/'grid_variants.json').write_text(json.dumps(json_value(result), indent=2, allow_nan=False), encoding='utf8')
    rows = []
    for name, r in results.items():
        for j, k in enumerate(K):
            rows.append(dict(variant=name, K=k, fine_iv=reference[j], paper_iv=paper[j],
                             cubic_iv=r['cubic_iv'][j], linear_iv=r['linear_iv'][j],
                             cubic_minus_fine_iv_bp=(r['cubic_iv'][j]-reference[j])*1e4,
                             cubic_minus_paper_iv_bp=(r['cubic_iv'][j]-paper[j])*1e4,
                             linear_minus_paper_iv_bp=(r['linear_iv'][j]-paper[j])*1e4))
    with (OUTPUT/'grid_variants.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    make_plots(c)
    return result


def make_plots(c=None):
    """Show the same paper strikes on linear/log axes, without resampling."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    if c is None:
        c = json.loads((RESULTS_DIR/'fdm_paper_direct/figure7.json').read_text(encoding='utf8'))
    K = np.asarray(c['K'])
    fdm_iv, paper_iv, ln_iv = (np.asarray(c[key]) for key in ('iv_estimate','paper_fdm_iv','ln_iv'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, scale in zip(axes, ('linear','log')):
        ax.plot(K, (ln_iv-fdm_iv)*1e4, '.-', label='Our LN - our FDM')
        ax.plot(K, (fdm_iv-paper_iv)*1e4, '.-', label='Our FDM - digitized paper FDM')
        ax.set(xscale=scale, xlabel=f'K/F ({scale} axis)', ylabel='IV difference (bp)',
               title='Same 50 paper strike positions')
        if scale=='log':
            ax.xaxis.set_major_locator(FixedLocator([.1,.2,.5,1.,2.,5.]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.axhline(0,color='black',lw=.5)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT/'strike_spacing.png',dpi=160)
    plt.close(fig)

    prices = np.asarray(c['runs']['fine']['price'])
    paper_prices = otm_prices(K,c['parameters']['T'],paper_iv)
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(K,(fdm_iv-paper_iv)*1e4,'.-')
    axes[1].plot(K,(prices-paper_prices)*1e4,'.-')
    for ax in axes:
        ax.axhline(0,color='black',lw=.5)
        ax.set(xlabel='K/F (linear axis)',title='Our FDM - digitized paper FDM')
        ax.grid(alpha=.25)
    axes[0].set_ylabel('IV difference (bp)')
    axes[1].set_ylabel('Price difference (bp at F=1)')
    fig.suptitle('Paper prices here are inferred from plotted IVs, not separately published prices')
    fig.tight_layout()
    fig.savefig(OUTPUT/'iv_and_price_difference.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    run()
