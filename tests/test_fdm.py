"""Numerical controls for the Craig-Sneyd implementation and its price convention."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from scipy.special import ndtr


def test_black_limit_has_second_order_spatial_convergence():
    from src.fdm import sabr_cs_price

    K = np.array([.7, .9, 1., 1.1, 1.4])
    std = .2
    d1 = np.log(1 / K) / std + std / 2
    call = ndtr(d1) - K * ndtr(d1 - std)
    put = K * ndtr(std - d1) - ndtr(-d1)
    target = np.where(K < 1, put, call)
    errors = []
    for nx in (200, 400):
        result = sabr_cs_price(1., K, 1., .2, 0., 0., nx=nx, ny=4, nt=400,
                               x_width=2., y_width=1., x_focus=None, y_focus=None)
        errors.append(np.max(np.abs(result['price'] - target)))
    assert errors[1] < 7e-6
    assert 3.8 < errors[0] / errors[1] < 4.2


def test_short_low_volatility_black_limit_with_focused_grid():
    from src.fdm import sabr_cs_price
    from src.integration import invert_otm

    T, alpha = 1 / 365, .05
    std = alpha * np.sqrt(T)
    K = np.exp(-std**2 / 2 + np.linspace(-3.5, 3.5, 10) * std)
    errors = []
    for nx in (200, 400, 800):
        prices = sabr_cs_price(1., K, T, alpha, 0., 0., nx=nx, ny=4, nt=400,
                               x_width=8., y_width=1., x_focus=std)['price']
        errors.append(np.max(np.abs(invert_otm(prices, K, T) - alpha)) * 1e4)
    assert np.all(np.array(errors[:-1]) / errors[1:] > 3.)
    assert errors[-1] < .5  # IV bp; coarse-grid apparent success is not enough.


def test_positive_correlation_uses_bounded_put_and_parity():
    from src.fdm import sabr_cs_price

    F = 1.7
    K = F * np.array([.8, 1., 1.2, 1.4])
    result = sabr_cs_price(F, K, 1., .12, .45, .85, nx=160, ny=80, nt=160,
                           positive_rho_parity=True)
    assert result['price_convention'] == 'positive_rho_put_call_parity'
    assert np.all((result['put'] >= 0) & (result['put'] <= K))
    assert np.all((result['call'] >= np.maximum(F - K, 0)) & (result['call'] <= F))
    assert_allclose(result['call'] - result['put'], F - K, atol=2e-16, rtol=0)
    assert_allclose(result['price'], np.where(K < F, result['put'], result['call']))


def test_forward_fitting_removes_parity_black_wing_bias():
    """A nu=0 put-parity solve must agree with the direct Black-limit solve."""
    from src.fdm import sabr_cs_price
    from src.integration import invert_otm

    T, alpha = .9472502872878165, .13809985952607953
    K = np.array([.6544809079824075, .9943516126752021, 1.677234290224781])
    errors = []
    for nx in (200, 400):
        grid = dict(nx=nx, ny=4, nt=400, x_focus=alpha*np.sqrt(T), fit_forward=True)
        parity = sabr_cs_price(1., K, T, alpha, .5, 0., positive_rho_parity=True, **grid)
        direct = sabr_cs_price(1., K, T, alpha, 0., 0., **grid)
        parity_iv = invert_otm(parity['price'], K, T)
        direct_iv = invert_otm(direct['price'], K, T)
        assert np.max(np.abs(parity_iv-direct_iv))*1e4 < .002
        errors.append(np.max(np.abs(parity_iv-alpha))*1e4)
        assert parity['fit_forward'] is True
    assert errors[1] < 3.
    assert errors[0]/errors[1] > 3.5


def test_negative_correlation_retains_verified_notebook_prices():
    from src.fdm import sabr_cs_price

    K = np.array([.4419, .5152, .6007, .7004, .8166, .9520, 1.1100, 1.2941, 1.5088, 1.7591, 2.0509])
    # Original notebook's verified 200 x 100 x 400 CS result, not the rounded paper values.
    reference = np.array([
        .0014763589594210607, .002389134494684375, .004078181300552688,
        .0075126707776349754, .015584577455871529, .03973835156407152,
        .017779486107681295, .004016849805744928, .0014322191211432356,
        .0006507535726373628, .00034028533197343275,
    ])
    result = sabr_cs_price(1., K, 1.6316, .108266, -.464053, .887903,
                           nx=200, ny=100, nt=400)
    assert result['price_convention'] == 'direct_otm'
    assert_allclose(result['price'], reference, atol=2e-12, rtol=0)


def _independent_semidiscrete_reference(rho, fit_forward=False, option='put'):
    """Dense generator with weights from polynomial exactness, independent of CS sweeps."""
    nx, ny = 12, 8
    x, y = np.linspace(-1.5, 1.5, nx + 1), np.linspace(-1., 1., ny + 1)
    dx, dy = x[1] - x[0], y[1] - y[0]
    alpha, nu, maturity = .3, .6, .5
    n = (nx + 1) * (ny + 1)
    A = np.zeros((n, n))

    def derivative(z, i, order):
        shifts = z[i-1:i+2] - z[i]
        rhs = np.zeros(3)
        rhs[order] = 1 if order == 1 else 2
        return np.linalg.solve(np.vstack([shifts ** k for k in range(3)]), rhs)

    def index(j, i):
        return j * (nx + 1) + i

    for j in range(ny + 1):
        sigma = alpha * np.exp(y[j])
        for i in range(1, nx):
            row = index(j, i)
            wx1, wx2 = derivative(x, i, 1), derivative(x, i, 2)
            drift = 1.
            if fit_forward:
                forward = np.exp(x[i-1:i+2])
                drift = np.dot(wx2, forward)/np.dot(wx1, forward)
            for shift, value in zip((-1, 0, 1), .5 * sigma**2 * (wx2 - drift*wx1)):
                A[row, index(j, i + shift)] += value
            if j in (0, ny):
                neighbor = 1 if j == 0 else ny - 1
                A[row, index(j, i)] -= nu**2 / dy**2
                A[row, index(neighbor, i)] += nu**2 / dy**2
            else:
                wy1, wy2 = derivative(y, j, 1), derivative(y, j, 2)
                for shift, value in zip((-1, 0, 1), .5 * nu**2 * (wy2 - wy1)):
                    A[row, index(j + shift, i)] += value
                for sx, ax in zip((-1, 0, 1), wx1):
                    for sy, ay in zip((-1, 0, 1), wy1):
                        A[row, index(j + sy, i + sx)] += rho * nu * sigma * ax * ay
    payoff = np.maximum(1 - np.exp(x), 0) if option == 'put' else np.maximum(np.exp(x) - 1, 0)
    h = dx / 2
    payoff[nx // 2] = ((np.expm1(-h) + h) / dx if option == 'put'
                       else (np.expm1(h) - h) / dx)
    return (expm(maturity * A) @ np.tile(payoff, ny + 1))[index(ny // 2, nx // 2)]


@pytest.mark.parametrize('rho', [-.4, .4])
@pytest.mark.parametrize('fit_forward', [False, True])
@pytest.mark.parametrize('option', ['put', 'call'])
def test_cs_time_convergence_against_independent_matrix_exponential(rho, fit_forward, option):
    from src.fdm import sabr_cs_price

    target = _independent_semidiscrete_reference(rho, fit_forward, option)
    errors = []
    for nt in (20, 40, 80):
        price = sabr_cs_price(1., [1.], .5, .3, rho, .6, nx=12, ny=8, nt=nt,
                              x_width=1.5, y_width=1., x_focus=None, y_focus=None,
                              fit_forward=fit_forward)[option][0]
        errors.append(abs(price - target))
    assert errors[-1] < 1e-6
    assert np.all(np.array(errors[:-1]) / errors[1:] > 3.8)


def test_positive_correlation_defaults_to_direct_call_and_put():
    from src.fdm import sabr_cs_price

    args = (1., [.8, 1., 1.2], .5, .3, .4, .6)
    grid = dict(nx=24, ny=16, nt=80, x_width=1.5, y_width=1.,
                x_focus=None, y_focus=None)
    direct = sabr_cs_price(*args, **grid)
    parity = sabr_cs_price(*args, positive_rho_parity=True, **grid)
    assert direct['price_convention'] == 'direct_otm'
    assert direct['positive_rho_parity'] is False
    assert parity['positive_rho_parity'] is True
    assert_allclose(direct['put'], parity['put'], rtol=0, atol=1e-14)
    # This deliberately coarse, unfitted generator does not preserve exp(x).
    # The direct call must therefore not be silently reconstructed by parity.
    assert np.max(np.abs(direct['call'] - parity['call'])) > 1e-5


def test_scalar_strike_and_label_row_shapes():
    from src.fdm import label_row, sabr_cs_price
    from src.hagan import make_strikes

    assert sabr_cs_price(1., 1., .5, .2, .2, .5, nx=40, ny=20, nt=40)['price'].shape == (1,)
    K = make_strikes(1., .5, .2, .2, .5)
    result = label_row(np.r_[.5, .2, .5, .2, K], nx=100, ny=40, nt=100, x_focus=.15)
    assert result.shape == (10,)
    assert np.isfinite(result).all()
    assert (result > 0).all()
    with pytest.raises(ValueError):
        label_row(np.ones(13))


@pytest.mark.parametrize('changes', [
    {'rho': 1.}, {'rho': -1.}, {'nu': -.1}, {'T': 0.}, {'F': np.nan},
    {'K': []}, {'K': [0.]}, {'K': [[1.]]}, {'K': [np.inf]}, {'K': [np.exp(9.)]},
    {'nx': 21}, {'ny': 21}, {'nt': 3}, {'nt': np.inf}, {'x_width': 0.}, {'y_focus': -.1},
])
def test_invalid_inputs_are_rejected(changes):
    from src.fdm import sabr_cs_price

    kwargs = dict(F=1., K=[1.], T=.5, alpha=.2, rho=.3, nu=.5, nx=20, ny=10, nt=20)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        sabr_cs_price(**kwargs)
