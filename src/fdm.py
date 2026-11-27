"""Craig-Sneyd finite differences for beta-one SABR, with direct call/put pricing.

With x=log(F/K), y=log(sigma/alpha), tau=T-t and V=K*u, the PDE is

    u_tau = .5*alpha**2*exp(2*y)*(u_xx-u_x)
          + .5*nu**2*(u_yy-u_y) + rho*nu*alpha*exp(y)*u_xy.

The original Craig-Sneyd method uses theta=1/2. X endpoints retain the payoff;
Y endpoints use reflecting (u_y=0) ghost values. These finite-domain boundaries,
sinh grids, payoff-cell averaging and cubic interpolation are explicit project
choices: McGhee does not disclose enough settings to recover his exact grid.

By default, evolve call and put directly for every rho. For legacy comparisons,
positive_rho_parity=True evolves only put when rho > 0 and defines call=P+F-K,
matching src.integration. Direct finite-domain call prices are not automatically
the raw positive-rho SABR payoff expectation: the boundary/asymptotic convention
must also be validated. This switch does not change the boundary conditions.
No SLN is used.
Prices are undiscounted. A grid refinement/domain check is required before using
outputs as accurate training labels; a successful solve alone is insufficient.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg.lapack import dgttrf, dgttrs

from src.integration import invert_otm


def sabr_cs_price(F, K, T, alpha, rho, nu, *, nx=400, ny=200, nt=400,
                  x_width=8., y_width=6., x_focus=.5, y_focus=1., fit_forward=False,
                  positive_rho_parity=False):
    """Return OTM, call and put prices for a scalar parameter set and a strike vector.

    ``nx``/``ny`` count spatial intervals, ``nt`` counts time steps. A positive
    focus parameter concentrates a sinh grid near zero; None selects a uniform
    grid. The default X focus is for the published example, not all short tenors:
    small alpha*sqrt(T) requires a smaller focus and its own convergence checks.

    ``fit_forward=True`` uses Dxx-c*Dx with c=(Dxx exp(x))/(Dx exp(x)).
    This second-order drift correction makes the discrete X generator annihilate
    the forward exp(x), avoiding an artificial intrinsic-value bias when a call
    is recovered from an ITM put. False preserves the original log-grid stencil.

    ``price`` uses put for K<F and call otherwise. Both payoffs are evolved
    directly by default. ``positive_rho_parity=True`` retains the historical
    positive-rho put-only solve and derives call by P+F-K; it has no effect for
    rho<=0. Prices are never clipped to Black bounds, so downstream validation
    can expose numerical error.
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    if (not np.isfinite([F, T, alpha, rho, nu]).all()
            or min(F, T, alpha) <= 0 or nu < 0 or not -1 < rho < 1):
        raise ValueError('Require finite F,T,alpha>0, nu>=0 and -1<rho<1.')
    if K.ndim != 1 or not K.size or np.any(K <= 0) or not np.all(np.isfinite(K)):
        raise ValueError('K must be a nonempty finite positive vector or scalar.')
    if (not np.isfinite([nx, ny, nt]).all()
            or any(int(n) != n or n < 4 for n in (nx, ny, nt)) or nx % 2 or ny % 2):
        raise ValueError('Use even nx,ny>=4 and integer nt>=4 (interval counts).')
    nx, ny, nt = int(nx), int(ny), int(nt)
    if not np.isfinite([x_width, y_width]).all() or min(x_width, y_width) <= 0:
        raise ValueError('Grid half-widths must be finite and positive.')
    if any(v is not None and (not np.isfinite(v) or v <= 0) for v in (x_focus, y_focus)):
        raise ValueError('Grid focus must be positive, or None for a uniform grid.')

    def grid(n, width, focus):
        if focus is None:
            return np.linspace(-width, width, n + 1)
        return focus * np.sinh(np.linspace(-np.arcsinh(width / focus),
                                           np.arcsinh(width / focus), n + 1))

    def weights(z):
        hl, hr = np.diff(z)[:-1], np.diff(z)[1:]
        d1 = np.array([-hr / (hl * (hl + hr)), (hr - hl) / (hl * hr), hl / (hr * (hl + hr))])
        d2 = np.array([2 / (hl * (hl + hr)), -2 / (hl * hr), 2 / (hr * (hl + hr))])
        return d1, d2

    x, y = grid(nx, x_width, x_focus), grid(ny, y_width, y_focus)
    x1, x2 = weights(x)
    y1, y2 = weights(y)
    q = np.log(F / K)
    if np.any(np.abs(q) >= x_width):
        raise ValueError('Strikes fall outside the log-moneyness grid.')
    dx, dy, dt = x[nx // 2 + 1] - x[nx // 2], y[ny // 2 + 1] - y[ny // 2], T / nt
    sigma = alpha * np.exp(y)

    # Only the explicit legacy convention reconstructs a call from a put.
    parity = positive_rho_parity and rho > 0
    n_payoffs = 1 if parity else 2
    put_payoff = np.maximum(-np.expm1(x), 0)
    payoff = (put_payoff[:, None] if parity else
              np.stack((np.maximum(np.expm1(x), 0), put_payoff), axis=-1))
    h = dx / 2
    put_average = (np.expm1(-h) + h) / dx
    payoff[nx // 2] = ([put_average] if parity else [(np.expm1(h) - h) / dx, put_average])
    U = np.broadcast_to(payoff, (ny + 1, nx + 1, n_payoffs)).copy()

    drift = 1.
    if fit_forward:
        hl, hr = np.diff(x)[:-1], np.diff(x)[1:]
        # Local exponential increments avoid subtracting nearly equal exp(x) values.
        el, er = np.expm1(-hl), np.expm1(hr)
        drift = (x2[0]*el + x2[2]*er) / (x1[0]*el + x1[2]*er)
    low, mid, high = .5 * sigma[None, :, None]**2 * (x2 - drift*x1)[:, None, :]
    ym, yd, yp = np.zeros((3, ny + 1))
    ym[1:-1], yd[1:-1], yp[1:-1] = .5 * nu**2 * (y2 - y1)
    yp[0] = nu**2 / (y[1] - y[0])**2
    yd[0] = -yp[0]
    ym[-1] = nu**2 / (y[-1] - y[-2])**2
    yd[-1] = -ym[-1]
    cross = rho * nu * sigma[1:-1, None, None]

    def a1(V):
        out = np.zeros_like(V)
        out[:, 1:-1] = (low[:, :, None] * V[:, :-2] + mid[:, :, None] * V[:, 1:-1]
                        + high[:, :, None] * V[:, 2:])
        return out

    def a2(V):
        out = yd[:, None, None] * V
        out[1:] += ym[1:, None, None] * V[:-1]
        out[:-1] += yp[:-1, None, None] * V[1:]
        out[:, [0, -1]] = 0.
        return out

    def a0(V):
        out = np.zeros_like(V)
        dxv = x1[0, :, None] * V[:, :-2] + x1[1, :, None] * V[:, 1:-1] + x1[2, :, None] * V[:, 2:]
        out[1:-1, 1:-1] = cross * (y1[0, :, None, None] * dxv[:-2]
                                     + y1[1, :, None, None] * dxv[1:-1]
                                     + y1[2, :, None, None] * dxv[2:])
        return out

    factor = dt / 2
    # Independent X lines form a block-diagonal tridiagonal system.
    lo = np.zeros((ny + 1, nx + 1))
    di, up = np.ones_like(lo), np.zeros_like(lo)
    lo[:, 1:-1] = -factor * low
    di[:, 1:-1] -= factor * mid
    up[:, 1:-1] = -factor * high
    fx = dgttrf(lo.ravel()[1:], di.ravel(), up.ravel()[:-1])
    fy = dgttrf(-factor * ym[1:], 1 - factor * yd, -factor * yp[:-1])
    if fx[-1] != 0 or fy[-1] != 0:
        raise ArithmeticError('Tridiagonal factorization failed.')

    def sweep(V, old1, old2):
        rhs = V - factor * old1
        solved, info = dgttrs(*fx[:-1], rhs.reshape(-1, n_payoffs))
        if info != 0:
            raise ArithmeticError('X solve failed.')
        rhs = solved.reshape(U.shape) - factor * old2
        interior, info = dgttrs(*fy[:-1], rhs[:, 1:-1].reshape(ny + 1, -1))
        if info != 0:
            raise ArithmeticError('Y solve failed.')
        rhs[:, 1:-1] = interior.reshape(ny + 1, nx - 1, n_payoffs)
        return rhs

    for _ in range(nt):
        old1, old2 = a1(U), a2(U)
        Y0 = U + dt * (a0(U) + old1 + old2)
        Y2 = sweep(Y0, old1, old2)
        # Original CS correction: both sweeps subtract A_j U_old, not A_j Y2.
        U = sweep(Y0 + .5 * dt * a0(Y2 - U), old1, old2)
    values = CubicSpline(x, U[ny // 2], axis=0)(q) * K[:, None]
    if not np.isfinite(values).all():
        raise FloatingPointError('Non-finite PDE prices; inspect the grid and time step.')
    put = values[:, -1]
    call = put + F - K if parity else values[:, 0]
    prices = np.where(K < F, put, call)
    return {'price': prices, 'call': call, 'put': put,
            'price_convention': 'positive_rho_put_call_parity' if parity else 'direct_otm',
            'nx': nx, 'ny': ny, 'nt': nt, 'dx': dx, 'dy': dy, 'dt': dt,
            'x_width': x_width, 'y_width': y_width, 'x_focus': x_focus, 'y_focus': y_focus,
            'fit_forward': bool(fit_forward), 'positive_rho_parity': bool(positive_rho_parity)}


def label_row(row, **grid):
    """Return ten decimal Black IVs for [T,alpha,nu,rho,K1,...,K10], F=1.

    Failed price-to-IV inversions remain NaN. This single-grid convenience
    function does not certify convergence; use separate grid/domain comparisons.
    """
    row = np.asarray(row, dtype=float)
    if row.shape != (14,):
        raise ValueError('Expected one row with 14 entries: T,alpha,nu,rho,K1,...,K10.')
    T, alpha, nu, rho = row[:4]
    K = row[4:]
    result = sabr_cs_price(1., K, T, alpha, rho, nu, **grid)
    return invert_otm(result['price'], K, T)
