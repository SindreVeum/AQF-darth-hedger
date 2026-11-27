import json


def test_report_keeps_all_failed_points_without_crashing_or_accepting_them(tmp_path):
    from src.fdm_paper import report
    from src.fdm_experiment import TOLERANCES_BP

    case = dict(complete=True, parameters={'T': 1.}, K=[1.], iv_estimate=[None], ln_iv=[.2],
                paper_fdm_iv=[.21], ln_price=[.08], status=['invalid_iv'],
                reference='Synthetic failure-handling check',
                changes_bp={key: [None] for key in TOLERANCES_BP},
                runs={'fine': {'price': [None], 'seconds': 0.}})
    for name in ('tables', 'figure7'):
        (tmp_path/f'{name}.json').write_text(json.dumps(dict(case, name=name)), encoding='utf8')

    result = report(tmp_path)
    assert all(v['points'] == 1 and v['accepted'] == 0 for v in result.values())
    assert all(v['all_estimates']['ln_vs_fdm_iv']['points'] == 0 for v in result.values())
    assert all(v['accepted_only']['ln_vs_fdm_iv']['mae_bp'] is None for v in result.values())
    assert all(all(x is None for x in v['max_grid_changes_bp'].values()) for v in result.values())
    assert (tmp_path/'comparison.png').is_file()
    assert (tmp_path/'comparison.csv').read_text().count('invalid_iv') == 2


def test_report_distinguishes_printed_prices_from_prices_inferred_from_iv(tmp_path):
    import csv
    import math
    import numpy as np
    from src.fdm_paper import otm_prices, report

    K = np.array([.8, 1.2])
    iv = np.array([.3, .3])
    price = otm_prices(K, 1., iv)
    case = dict(complete=True, parameters={'T': 1.}, K=K.tolist(),
                iv_estimate=iv.tolist(), ln_iv=iv.tolist(), paper_fdm_iv=iv.tolist(),
                ln_price=price.tolist(), status=['ok', 'ok'], reference='Synthetic reference',
                changes_bp={'space': [0., 0.]}, runs={'fine': {'price': price.tolist(), 'seconds': 0.}})
    for name in ('tables', 'figure7'):
        saved = dict(case, name=name)
        if name == 'tables':
            saved['paper_fdm_price'] = (price+.0001).tolist()
        (tmp_path/f'{name}.json').write_text(json.dumps(saved), encoding='utf8')

    summary = report(tmp_path)
    table = summary['tables']['all_estimates']
    figure = summary['figure7']['all_estimates']
    assert np.isclose(table['fdm_vs_paper_price']['mae_bp'], 1.)
    assert 'fdm_vs_paper_iv_implied_price' not in table
    assert figure['fdm_vs_paper_iv_implied_price']['max_abs_bp'] == 0.
    assert 'fdm_vs_paper_price' not in figure
    with (tmp_path/'comparison.csv').open() as f:
        rows = list(csv.DictReader(f))
    figure_rows = [r for r in rows if r['case'] == 'figure7']
    assert all(math.isnan(float(r['paper_fdm_price'])) for r in figure_rows)
    assert all(float(r['fdm_minus_paper_iv_implied_price_bp']) == 0. for r in figure_rows)
    with (tmp_path/'error_summary.csv').open() as f:
        errors = list(csv.DictReader(f))
    assert len(errors) == 8
    inferred = next(r for r in errors if r['comparison'] == 'fdm_vs_paper_iv_implied_price')
    assert inferred['unit'] == 'price bp'
    assert 'inferred' in inferred['reference']
