"""Direct call/put Craig-Sneyd comparisons at McGhee Tables 6/7 and Figure 7.

Run python -m src.fdm_paper --case tables or --case figure7, then --report-only.
No ANN training or random sample generation is performed. Figure 7 references
are digitized PDF coordinates, not an unpublished full-precision numerical table.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.config import RESULTS_DIR
from src.fdm import sabr_cs_price
from src.fdm_experiment import TOLERANCES_BP, assess_convergence, json_value
from src.fdm_refinement import expand_domain
from src.hagan import hagan_iv
from src.integration import black_price, invert_otm, label_row


OUTPUT = RESULTS_DIR / 'fdm_paper_direct'


def paper_case(name, output=OUTPUT):
    if name == 'tables':
        return dict(name=name, parameters=dict(T=1.6316, alpha=.108266, nu=.887903, rho=-.464053),
                    K=np.array([.4419, .5152, .6007, .7004, .8166, .9520, 1.1100,
                                1.2941, 1.5088, 1.7591, 2.0509]),
                    paper_fdm_iv=np.array([30.02, 26.78, 23.39, 19.81, 16.05, 12.23,
                                           10.53, 12.55, 15.23, 17.93, 20.56])/100,
                    paper_ln_iv=np.array([29.97, 26.72, 23.33, 19.76, 16.00, 12.20,
                                          10.51, 12.54, 15.24, 17.94, 20.56])/100,
                    paper_fdm_price=np.array([.15, .24, .41, .76, 1.57, 3.97,
                                              1.77, .41, .15, .07, .04])/100,
                    reference='Tables 6/7: printed IV and price columns, rounded to 0.01 percent.')
    reference = json.loads((Path(output) / 'figure7_reference.json').read_text(encoding='utf8'))
    return dict(name=name, parameters=reference['parameters'], K=np.array(reference['K']),
                paper_fdm_iv=np.array(reference['paper_fdm_iv']),
                paper_ln_iv=np.array(reference['paper_ln_iv']),
                reference='Figure 7: 50 PDF polyline vertices. Strike and IV coordinates are digitized; '
                          'the paper does not specify whether plotted vertices involve interpolation.')


def paper_grids(name='tables'):
    """Keep the existing negative-rho scheme; vary one numerical control at a time."""
    b = dict(nx=400, ny=200, nt=400, x_width=8., y_width=6., x_focus=.5,
             y_focus=1., fit_forward=False, positive_rho_parity=False)
    if name == 'figure7':
        # The initial 400x200 -> 800x400 comparison changes by 1.217 IV bp.
        b.update(nx=640, ny=320)
    return dict(coarse=dict(b, nx=b['nx']//2, ny=b['ny']//2), base=b,
                fine=dict(b, nx=2*b['nx'], ny=2*b['ny']),
                time_coarse=dict(b, nt=200), time_fine=dict(b, nt=800),
                x_wide=expand_domain(b, 'x', 12.), y_wide=expand_domain(b, 'y', 9.9))


def otm_prices(K, T, iv):
    return np.where(K < 1., black_price(1., K, T, iv, 'put'),
                    black_price(1., K, T, iv, 'call'))


def metrics(actual, reference, strikes, mask=None):
    actual, reference = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(reference)
    if mask is not None:
        valid &= mask
    indices = np.flatnonzero(valid)
    if not len(indices):
        return dict(points=0, mae_bp=None, rmse_bp=None, max_abs_bp=None, max_strike=None)
    diff = (actual[valid]-reference[valid])*1e4
    j = indices[np.argmax(abs(diff))]
    return dict(points=len(indices), mae_bp=float(np.mean(abs(diff))),
                rmse_bp=float(np.sqrt(np.mean(diff**2))), max_abs_bp=float(np.max(abs(diff))),
                max_strike=float(strikes[j]))


def run_case(name, output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    case = paper_case(name, output)
    K, p = case['K'], case['parameters']
    row = np.r_[p['T'], p['alpha'], p['nu'], p['rho'], K]
    runs = {}
    for grid_name, settings in paper_grids(name).items():
        print(f'{name}: {grid_name} {settings["nx"]}x{settings["ny"]}x{settings["nt"]}', flush=True)
        start = perf_counter()
        values = sabr_cs_price(1., K, **p, **settings)
        iv = invert_otm(values['price'], K, p['T'])
        runs[grid_name] = dict(settings=settings, price=values['price'], call=values['call'],
                               put=values['put'], iv=iv, price_convention=values['price_convention'],
                               seconds=perf_counter()-start)
        (output/f'{name}_progress.json').write_text(json.dumps(json_value(
            dict(complete=False, case=case, runs=runs)), indent=2, allow_nan=False), encoding='utf8')
        print(f'  {runs[grid_name]["seconds"]:.2f}s, {np.isfinite(iv).sum()}/{len(K)} finite IVs', flush=True)
    assessment = assess_convergence(runs)
    start = perf_counter()
    ln_iv = label_row(row)
    result = dict(case, runs=runs, **assessment, ln_iv=ln_iv,
                  ln_price=otm_prices(K, p['T'], ln_iv), ln_seconds=perf_counter()-start,
                  hagan_iv=hagan_iv(1., K, p['T'], p['alpha'], 1., p['rho'], p['nu']),
                  complete=True, tolerances_bp=TOLERANCES_BP,
                  price_convention='direct_otm',
                  limitation='Finite-domain direct call/put PDE. Both published cases have negative rho. '
                             'Time and domain checks use the medium spatial grid, not the finest grid.',
                  source_sha256={n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                 for n in ('fdm.py', 'fdm_paper.py', 'integration.py')})
    (output/f'{name}.json').write_text(json.dumps(json_value(result), indent=2, allow_nan=False), encoding='utf8')
    (output/f'{name}_progress.json').unlink()
    print(f'{name}: accepted {np.sum(result["status"] == "ok")}/{len(K)}', flush=True)
    return result


def report(output=OUTPUT):
    """Rebuild tables/plot from saved solves without executing another PDE."""
    output = Path(output)
    cases, summaries, rows, error_rows = [], {}, [], []
    for name in ('tables', 'figure7'):
        c = json.loads((output/f'{name}.json').read_text(encoding='utf8'))
        if not c['complete']:
            raise ValueError('Do not report an incomplete numerical experiment.')
        K = np.array(c['K'])
        # Older saved table solves predate the addition of the printed LN column.
        if name == 'tables' and 'paper_ln_iv' not in c:
            reference = paper_case('tables')
            if np.array_equal(K, reference['K']):
                c['paper_ln_iv'] = reference['paper_ln_iv'].tolist()
        fdm, ln, paper = (np.array(c[key], dtype=float) for key in ('iv_estimate', 'ln_iv', 'paper_fdm_iv'))
        paper_ln = np.array(c.get('paper_ln_iv', np.full(len(K), np.nan)), dtype=float)
        fp = np.array(c['runs']['fine']['price'], dtype=float)
        lp = np.array(c['ln_price'], dtype=float)
        ok = np.array(c['status']) == 'ok'
        pairs = dict(ln_vs_fdm_iv=(ln, fdm), ln_vs_fdm_price=(lp, fp), fdm_vs_paper_iv=(fdm, paper))
        paper_iv_implied_price = np.full(len(K), np.nan)
        if 'paper_fdm_price' in c:
            pairs['fdm_vs_paper_price'] = (fp, np.array(c['paper_fdm_price']))
            price_reference = 'Printed Table 7 prices, independently rounded.'
        else:
            paper_iv_implied_price = otm_prices(K, c['parameters']['T'], paper)
            pairs['fdm_vs_paper_iv_implied_price'] = (fp, paper_iv_implied_price)
            price_reference = 'Black OTM prices inferred from digitized Figure 7 IVs; not published prices.'
        if 'paper_ln_iv' in c:
            pairs['ln_vs_paper_ln_iv'] = (ln, np.array(c['paper_ln_iv']))
        changes = {k: np.array(v, dtype=float) for k, v in c['changes_bp'].items()}
        summaries[name] = dict(points=len(K), accepted=int(ok.sum()), reference=c['reference'],
                               paper_price_reference=price_reference,
                               max_grid_changes_bp={k: float(np.nanmax(v)) if np.isfinite(v).any() else None
                                                    for k, v in changes.items()},
                               fine_seconds=c['runs']['fine']['seconds'],
                               total_fdm_seconds=sum(r['seconds'] for r in c['runs'].values()),
                               all_estimates={key: metrics(a, b, K) for key, (a, b) in pairs.items()},
                               accepted_only={key: metrics(a, b, K, ok) for key, (a, b) in pairs.items()})
        for key, values in summaries[name]['all_estimates'].items():
            if key == 'ln_vs_paper_ln_iv':
                continue
            unit = 'IV bp' if key.endswith('_iv') else 'price bp'
            reference = ('Our LN and our direct FDM.' if key.startswith('ln_vs_fdm') else
                         c['reference'] if unit == 'IV bp' else price_reference)
            error_rows.append(dict(case=name, comparison=key, unit=unit, reference=reference, **values))
        for j, strike in enumerate(K):
            rows.append(dict(case=name, K=strike, status=c['status'][j], ln_iv=ln[j],
                             fdm_iv_estimate=fdm[j], fdm_iv=fdm[j] if ok[j] else np.nan,
                             paper_fdm_iv=paper[j], ln_minus_fdm_iv_bp=(ln[j]-fdm[j])*1e4,
                             paper_ln_iv=paper_ln[j], paper_ln_minus_paper_fdm_iv_bp=(paper_ln[j]-paper[j])*1e4,
                             fdm_minus_paper_iv_bp=(fdm[j]-paper[j])*1e4,
                             ln_price=lp[j], fdm_price=fp[j], ln_minus_fdm_price_bp=(lp[j]-fp[j])*1e4,
                             paper_fdm_price=c.get('paper_fdm_price', [np.nan]*len(K))[j],
                             fdm_minus_paper_price_bp=(fp[j]-c['paper_fdm_price'][j])*1e4
                                                     if 'paper_fdm_price' in c else np.nan,
                             paper_iv_implied_price=paper_iv_implied_price[j],
                             fdm_minus_paper_iv_implied_price_bp=(fp[j]-paper_iv_implied_price[j])*1e4,
                             **{f'{key}_change_bp': values[j] for key, values in c['changes_bp'].items()}))
        cases.append(c)
    (output/'summary.json').write_text(json.dumps(json_value(summaries), indent=2, allow_nan=False), encoding='utf8')
    with (output/'comparison.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output/'error_summary.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=error_rows[0].keys())
        writer.writeheader()
        writer.writerows(error_rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for axs, c in zip(axes, cases):
        K = np.array(c['K'])
        fdm, ln, paper = (np.array(c[key], dtype=float) for key in ('iv_estimate', 'ln_iv', 'paper_fdm_iv'))
        for iv, label, style in [(ln, 'Our LN', '-'), (fdm, 'Our direct FDM', '-'),
                                  (paper, 'Paper FDM (printed/digitized)', '--')]:
            axs[0].plot(K, iv*100, style, label=label)
        axs[1].plot(K, (ln-fdm)*1e4, '.-', label='Our LN - our FDM')
        axs[1].plot(K, (fdm-paper)*1e4, '.-', label='Our FDM - paper FDM')
        if 'paper_ln_iv' in c:
            paper_ln = np.array(c['paper_ln_iv'], dtype=float)
            axs[0].plot(K, paper_ln*100, ':o', color='tab:purple', markersize=3,
                        markerfacecolor='none', label='Paper LN (printed/digitized)')
            axs[1].plot(K, (paper_ln-paper)*1e4, '--s', color='tab:purple', markersize=3,
                        markerfacecolor='none', label='Paper LN - paper FDM')
        axs[1].axhline(0, color='black', linewidth=.5)
        bad = np.array(c['status']) != 'ok'
        if bad.any():
            axs[1].scatter(K[bad], (ln-fdm)[bad]*1e4, marker='x', color='red', label='FDM check failed')
        axs[0].set_ylabel('Black IV (%)')
        axs[1].set_ylabel('IV difference (bp)')
        for ax in axs:
            ax.set(xscale='linear', xlabel='K/F (linear scale)',
                   title='Tables 6/7' if c['name']=='tables' else 'Figure 7')
            padding = .03*max(K.max()-K.min(), .1*K.max())
            ax.set_xlim(max(0., K.min()-padding), K.max()+padding)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle('Published negative-correlation cases: direct call/put FDM, no parity conversion')
    fig.tight_layout()
    fig.savefig(output/'comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps(json_value(summaries), indent=2), flush=True)
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('tables', 'figure7'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        if args.case:
            parser.error('--report-only cannot be combined with --case.')
        report(args.output)
    elif args.case:
        run_case(args.case, args.output)
    else:
        for name in ('tables', 'figure7'):
            run_case(name, args.output)
        report(args.output)
