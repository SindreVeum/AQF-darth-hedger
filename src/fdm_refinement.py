"""Repeat the saved eight-smile pilot with targeted grid/domain refinements.

Run ``python -m src.fdm_refinement``; optionally select one ``--case NAME``.
These are pilot-specific settings, not a validated full-domain sampling rule.
The original results/fdm_pilot and all datasets/checkpoints are read-only inputs.
"""

import argparse
import csv
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np

from src.config import RESULTS_DIR
from src.fdm import sabr_cs_price
from src.fdm_experiment import (TOLERANCES_BP, assess_convergence, json_value,
                                plot_results, write_outputs)
from src.integration import invert_otm


def expand_domain(base, axis, target):
    """Extend a sinh grid while retaining every old interior node."""
    n, width, focus = (base[f'n{axis}'], base[f'{axis}_width'], base[f'{axis}_focus'])
    spacing = 2*np.arcsinh(width/focus)/n
    n_wide = 2*int(np.ceil(np.arcsinh(target/focus)/spacing))
    return dict(base, **{f'n{axis}': n_wide,
                        f'{axis}_width': focus*np.sinh(n_wide*spacing/2)})


def refinement_runs(case):
    """Explicit settings for the original pilot; no input/strike resampling."""
    T, alpha, nu, rho = case['row'][:4]
    std = alpha*np.sqrt(T)
    b = dict(nx=400, ny=200, nt=400, x_width=8., y_width=6.,
             x_focus=.5, y_focus=1., fit_forward=bool(rho > 0), positive_rho_parity=True)
    target_x = 12.
    name = case['name']
    if name == 'paper_2018':
        pass
    elif name == '1y_2y_test_10745':
        b.update(nx=440, ny=220)
    elif name == '1d_1y_test_3571':
        b.update(nx=480, ny=240, nt=800, x_focus=min(.5, std))
    elif name in ('1d_1y_test_44913', '1y_2y_test_1100'):
        nx, ny = (600, 160) if name == '1d_1y_test_44913' else (960, 256)
        b.update(nx=nx, ny=ny, x_width=1., x_focus=std, y_focus=nu*np.sqrt(T))
        target_x = 2.
    elif name in ('short_lowvol_negative', 'short_lowvol_positive'):
        b.update(nt=800, x_width=.5, x_focus=std)
        target_x = 1.
    elif name == 'long_highvol_positive':
        # Previously verified domain expansion, keeping the old transformed spacing.
        width = .5*np.sinh(528/400*np.arcsinh(8/.5))
        b.update(nx=1152, ny=480, x_width=width)
        target_x = 48.
    else:
        raise ValueError('Only the eight saved pilot cases have tuned settings.')
    return dict(coarse=dict(b, nx=b['nx']//2, ny=b['ny']//2), base=b,
                fine=dict(b, nx=2*b['nx'], ny=2*b['ny']),
                time_coarse=dict(b, nt=b['nt']//2), time_fine=dict(b, nt=2*b['nt']),
                x_wide=expand_domain(b, 'x', target_x), y_wide=expand_domain(b, 'y', 9.9))


def evaluate_grid(row, settings):
    T, alpha, nu, rho = row[:4]
    K = np.asarray(row[4:])
    start, cpu = perf_counter(), process_time()
    try:
        # Saved pilot settings predate the explicit convention flag.
        settings = {'positive_rho_parity': True, **settings}
        prices = sabr_cs_price(1., K, T, alpha, rho, nu, **settings)['price']
        iv, error = invert_otm(prices, K, T), None
    except (ArithmeticError, ValueError) as exc:
        prices, iv = np.full(len(K), np.nan), np.full(len(K), np.nan)
        error = f'{type(exc).__name__}: {exc}'
    return dict(settings=settings, price=prices, iv=iv, error=error,
                seconds=perf_counter()-start, cpu_seconds=process_time()-cpu)


def assemble_case(original, runs):
    """Reuse unchanged LN/ANN comparisons; only the FDM solves are replaced."""
    c = deepcopy(original)
    for key in ('row', 'ln_iv', 'hagan_iv'):
        c[key] = np.asarray(c[key], dtype=float)
    for value in c['ann'].values():
        value['iv'] = np.asarray(value['iv'], dtype=float)
    runs = {name: dict(r, price=np.asarray(r['price'], dtype=float),
                      iv=np.asarray(r['iv'], dtype=float)) for name, r in runs.items()}
    c.update(runs=runs, **assess_convergence(runs))
    return c


def comparison_summary(originals, results):
    """Compare exactly the same rows, including formerly failed strike indices."""
    old = {c['name']: c for c in originals}
    rows = []
    for c in results:
        p = old[c['name']]
        if not np.array_equal(p['row'], c['row']):
            raise ValueError('The input rows differ; this is not a paired comparison.')
        was_bad, now_ok = np.array(p['status']) != 'ok', np.array(c['status']) == 'ok'
        rows.append(dict(case=c['name'], candidates=len(now_ok),
                         before_passed=int((~was_bad).sum()), after_passed=int(now_ok.sum()),
                         recovered_strikes=(np.flatnonzero(was_bad & now_ok)+1).tolist(),
                         regressed_strikes=(np.flatnonzero(~was_bad & ~now_ok)+1).tolist(),
                         remaining_strikes=(np.flatnonzero(~now_ok)+1).tolist(),
                         max_changes_bp={k: float(np.nanmax(v)) for k, v in c['changes_bp'].items()},
                         old_fine_seconds=p['runs']['fine']['seconds'],
                         new_fine_seconds=c['runs']['fine']['seconds'],
                         old_validation_seconds=sum(r['seconds'] for r in p['runs'].values()),
                         new_validation_seconds=sum(r['seconds'] for r in c['runs'].values())))
    return rows


def write_comparison(originals, results, output):
    """Save paired results and the price sensitivity of remaining failed points."""
    output = Path(output)
    summary = comparison_summary(originals, results)
    (output/'comparison.json').write_text(json.dumps(json_value(summary), indent=2, allow_nan=False), encoding='utf8')
    rows = []
    for c in results:
        p = [np.asarray(c['runs'][key]['price']) for key in ('coarse', 'base', 'fine')]
        T = c['row'][0]
        for j in np.flatnonzero(c['status'] != 'ok'):
            K, iv = c['row'][4+j], c['iv_estimate'][j]
            std = iv*np.sqrt(T)
            d1 = np.log(1/K)/std+std/2
            vega = np.exp(-d1*d1/2)*np.sqrt(T/(2*np.pi))
            dp = abs(p[2][j]-p[1][j])
            rows.append(dict(case=c['name'], strike_index=int(j+1), K=K, status=c['status'][j],
                coarse_price=p[0][j], base_price=p[1][j], fine_price=p[2][j],
                space_price_change=dp, space_iv_change_bp=c['changes_bp']['space'][j],
                time_iv_change_bp=c['changes_bp']['time'][j], black_vega=vega,
                local_price_change_for_half_iv_bp=vega*.5e-4,
                successive_price_change_ratio=abs(p[1][j]-p[0][j])/dp if dp > 0 else np.nan))
    path = output/'remaining_points.csv'
    if rows:
        with path.open('w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    else:
        path.unlink(missing_ok=True)  # Remove a stale failure report from an earlier run.


def benchmark_coarse(output=RESULTS_DIR/'fdm_refined'):
    """Two paired repetitions on the same eight cases at each configuration's coarse level."""
    output = Path(output)
    source = RESULTS_DIR/'fdm_pilot/details.json'
    if output.resolve() == source.parent.resolve():
        raise ValueError('Choose an output directory separate from the original pilot.')
    cases = json.loads(source.read_text(encoding='utf8'))['cases']
    output.mkdir(parents=True, exist_ok=True)
    records = []
    # Untimed warm-up. All timed calls include assembly, pricing and IV inversion.
    warm = dict(cases[0]['runs']['coarse']['settings'], nx=20, ny=10, nt=4)
    evaluate_grid(cases[0]['row'], warm)
    for i, c in enumerate(cases):
        grids = dict(baseline=dict(c['runs']['coarse']['settings'], fit_forward=False),
                     refined=refinement_runs(c)['coarse'])
        for repeat in range(2):
            order = ('baseline', 'refined') if (i+repeat) % 2 == 0 else ('refined', 'baseline')
            for variant in order:
                run = evaluate_grid(c['row'], grids[variant])
                if variant == 'baseline':
                    np.testing.assert_allclose(run['price'], c['runs']['coarse']['price'], rtol=0, atol=5e-13)
                records.append(dict(case=c['name'], repeat=repeat+1, variant=variant, **run))
                print(f"{c['name']} {variant} repeat {repeat+1}: {run['seconds']:.3f}s", flush=True)
    metadata = dict(repeats=2, level='coarse', cases=8, execution='Serial calls, alternating variant order.',
                    scope='Measured coarse-grid costs only. No extrapolation to fine grids or complete datasets.',
                    baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    solver_sha256=hashlib.sha256(Path(__file__).with_name('fdm.py').read_bytes()).hexdigest())
    (output/'timing.json').write_text(json.dumps(json_value(dict(metadata=metadata, runs=records)),
                                                 indent=2, allow_nan=False), encoding='utf8')
    rows = []
    for c in cases:
        row = dict(case=c['name'])
        for variant in ('baseline', 'refined'):
            r = [x for x in records if x['case'] == c['name'] and x['variant'] == variant]
            row[f'{variant}_wall_seconds'] = float(np.mean([x['seconds'] for x in r]))
            row[f'{variant}_cpu_seconds'] = float(np.mean([x['cpu_seconds'] for x in r]))
        rows.append(row)
    with (output/'timing.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return records


def main(case_names=None, output=RESULTS_DIR/'fdm_refined'):
    source = RESULTS_DIR/'fdm_pilot/details.json'
    output = Path(output)
    if output.resolve() == source.parent.resolve():
        raise ValueError('Choose an output directory separate from the original pilot.')
    original = json.loads(source.read_text(encoding='utf8'))
    cases = original['cases']
    if case_names:
        known = {c['name'] for c in cases}
        if set(case_names)-known:
            raise ValueError('Unknown pilot case.')
        cases = [c for c in cases if c['name'] in case_names]
    metadata = dict(expected_cases=len(cases), case_names=[c['name'] for c in cases],
                    tolerances_bp=TOLERANCES_BP, baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    versions={p: importlib.metadata.version(p) for p in ('numpy', 'scipy')},
                    source_sha256={f'src/{n}': hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                   for n in ('fdm.py', 'fdm_refinement.py', 'fdm_experiment.py', 'integration.py')},
                    comparison='Same saved inputs and thresholds. LN/ANN predictions and timings reused unchanged.',
                    limitation='Pilot-specific grids. Time/domain checks are on the medium spatial grid.',
                    timing='Wall seconds plus process CPU where available. Historical wall times included contention.')
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for c in cases:
        runs = {}
        for name, settings in refinement_runs(c).items():
            print(f"{c['name']}: {name}", flush=True)
            runs[name] = evaluate_grid(c['row'], settings)
            # A separately named progress file never masquerades as completed labels.
            (output/'progress.json').write_text(json.dumps(json_value(dict(case=c['name'], runs=runs,
                                                                          complete=False)), indent=2), encoding='utf8')
        results.append(assemble_case(c, runs))
        write_outputs(results, output, metadata)
    plot_results(results, output)
    write_outputs(results, output, metadata, complete=True)
    write_comparison(cases, results, output)
    (output/'progress.json').unlink(missing_ok=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', dest='case_names')
    parser.add_argument('--output', type=Path, default=RESULTS_DIR/'fdm_refined')
    parser.add_argument('--timing-only', action='store_true', help='Only run serial paired coarse-grid timings.')
    args = parser.parse_args()
    if args.timing_only:
        if args.case_names:
            parser.error('--timing-only uses all eight cases; do not combine it with --case.')
        benchmark_coarse(args.output)
    else:
        main(args.case_names, args.output)
