import numpy as np
import json


def test_domain_expansion_preserves_interior_sinh_nodes():
    from src.fdm_experiment import grid_runs

    runs = grid_runs(.05 / np.sqrt(365))
    base = runs['base']
    for axis in ('x', 'y'):
        wide = runs[f'{axis}_wide']
        focus = base[f'{axis}_focus']
        def nodes(g):
            return focus * np.sinh(np.linspace(-np.arcsinh(g[f'{axis}_width'] / focus),
                                               np.arcsinh(g[f'{axis}_width'] / focus), g[f'n{axis}'] + 1))
        extra = (wide[f'n{axis}'] - base[f'n{axis}']) // 2
        np.testing.assert_allclose(nodes(wide)[extra:-extra], nodes(base), atol=1e-13)
    assert runs['coarse']['nt'] == runs['base']['nt'] == runs['fine']['nt']
    assert runs['fine']['nx'] == 2 * base['nx']
    assert runs['time_fine']['nt'] == 2 * base['nt']


def test_acceptance_keeps_failed_labels_and_separates_diagnostics():
    from src.fdm_experiment import assess_convergence

    iv = np.full(6, .2)
    runs = {k: {'iv': iv.copy()} for k in
            ('coarse', 'base', 'fine', 'time_coarse', 'time_fine', 'x_wide', 'y_wide')}
    runs['fine']['iv'][1] += .6 / 1e4
    runs['time_fine']['iv'][2] += .06 / 1e4
    runs['x_wide']['iv'][3] += .11 / 1e4
    runs['y_wide']['iv'][4] += .11 / 1e4
    runs['fine']['iv'][5] = np.nan
    result = assess_convergence(runs)
    assert result['status'].tolist() == ['ok', 'space_not_converged', 'time_not_converged',
                                        'x_domain_not_converged', 'y_domain_not_converged', 'invalid_iv']
    assert result['iv'][0] == .2
    assert np.isnan(result['iv'][1:]).all()
    assert np.isfinite(result['iv_estimate'][1:5]).all()


def test_pilot_cases_keep_existing_inputs_and_both_correlation_signs():
    from src.fdm_experiment import pilot_cases

    cases = pilot_cases(random_per_sign=1, seed=20261126)
    assert len(cases) == 8  # one published case, four sampled rows, three stress cases
    sampled = [c for c in cases if c['source'] == 'test_dataset']
    assert {(c['window'], np.sign(c['row'][3])) for c in sampled} == {
        ('1d_1y', -1), ('1d_1y', 1), ('1y_2y', -1), ('1y_2y', 1)}
    for c in sampled:
        assert len(c['row']) == 14 and len(c['saved_ln_iv']) == 10
    for c in cases:
        T, alpha, nu, rho = c['row'][:4]
        assert 1/365 <= T <= 2 and .05 <= alpha <= .5
        assert -.9 <= rho <= .9
        # The published case is not constrained by the training-domain nu heuristic.
        if c['source'] != 'paper':
            assert .05 - 1e-12 <= nu*np.sqrt(12*T) <= 4 + 1e-12


def test_progress_snapshot_cannot_claim_complete(tmp_path):
    from src.fdm_experiment import write_outputs

    c = dict(name='partial', source='paper', window='1y_2y', row=np.r_[1., .2, .5, -.3, 1.],
             price_convention='direct_otm', status=np.array(['ok']),
             runs={'fine': {'price': np.array([.08])}}, iv=np.array([.2]), iv_estimate=np.array([.2]),
             ln_iv=np.array([.2]), hagan_iv=np.array([.2]), changes_bp={}, ann={})
    write_outputs([c], tmp_path, {'expected_cases': 8})
    saved = json.loads((tmp_path/'details.json').read_text())
    assert saved['metadata']['completed_cases'] == 1
    assert saved['metadata']['complete'] is False
