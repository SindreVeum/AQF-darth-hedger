import numpy as np
import pytest


def test_refinement_keeps_inputs_and_checks_domains_without_changing_spacing():
    from src.fdm_experiment import pilot_cases, TOLERANCES_BP
    from src.fdm_refinement import refinement_runs

    assert TOLERANCES_BP == dict(space=.5, time=.05, x_domain=.1, y_domain=.1)
    for c in pilot_cases():
        row = c['row'].copy()
        runs = refinement_runs(c)
        np.testing.assert_array_equal(c['row'], row)
        b = runs['base']
        assert runs['fine']['nx'] == b['nx']*2
        assert runs['fine']['ny'] == b['ny']*2
        assert runs['fine']['nt'] == b['nt']
        assert runs['time_fine']['nt'] == b['nt']*2
        for axis in ('x', 'y'):
            w = runs[f'{axis}_wide']
            f = b[f'{axis}_focus']
            def nodes(g):
                return f*np.sinh(np.linspace(-np.arcsinh(g[f'{axis}_width']/f),
                                              np.arcsinh(g[f'{axis}_width']/f), g[f'n{axis}']+1))
            extra = (w[f'n{axis}']-b[f'n{axis}'])//2
            assert extra > 0
            np.testing.assert_allclose(nodes(w)[extra:-extra], nodes(b), atol=5e-13)
    with pytest.raises(ValueError, match='pilot'):
        refinement_runs(dict(name='unseen_case', row=np.ones(14)))


def test_refinement_cannot_overwrite_original_pilot():
    from src.config import RESULTS_DIR
    from src.fdm_refinement import main, benchmark_coarse

    for function in (main, benchmark_coarse):
        with pytest.raises(ValueError, match='separate'):
            function(output=RESULTS_DIR/'fdm_pilot')
