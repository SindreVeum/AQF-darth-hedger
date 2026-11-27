"""Independent diagnostic: unsplit SABR PDE in f=F/K, y=log(sigma/alpha).

This is not the production Craig-Sneyd solver. It uses sparse Crank-Nicolson
with two initial backward-Euler half steps and a normalized-forward grid.
Call and put are evolved independently; no parity substitution is performed.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

from src.integration import invert_otm


def forward_price(K, T, alpha, rho, nu, *, n_left=80, ny=100, nt=400,
                  f_max=32., y_width=4.5, focus=.2, y_boundary='reflecting'):
    """Undiscounted direct call/put prices at F=1; widths retain grid nodes."""
    if ny % 2 or y_boundary not in ('reflecting', 'frozen'):
        raise ValueError('Require even ny and a reflecting or frozen Y boundary.')
    dz = np.arcsinh(1/focus)/n_left
    nr = int(np.ceil(np.arcsinh((f_max-1)/focus)/dz))
    f = 1+focus*np.sinh(np.arange(-n_left, nr+1)*dz)
    f[0] = 0.
    y = np.linspace(-y_width, y_width, ny+1)
    nf, dy = len(f), y[1]-y[0]
    if np.max(1/np.asarray(K)) >= f[-1]:
        raise ValueError('Query is outside the forward domain.')
    rows, cols, w1, w2 = [], [], [], []
    for i in range(1, nf-1):
        h = f[i-1:i+2]-f[i]
        matrix = np.vstack([np.ones(3), h, h*h])
        a = np.linalg.solve(matrix, [0., 1., 0.])
        b = np.linalg.solve(matrix, [0., 0., 2.])
        rows.extend([i]*3); cols.extend([i-1, i, i+1])
        w1.extend(a); w2.extend(b)
    Df = csr_matrix((w1, (rows, cols)), shape=(nf, nf))
    Dff = csr_matrix((w2, (rows, cols)), shape=(nf, nf))
    Dy = diags([-np.ones(ny)/(2*dy), np.ones(ny)/(2*dy)], [-1, 1],
               shape=(ny+1, ny+1), format='lil')
    Dyy = diags([np.ones(ny)/dy**2, -2*np.ones(ny+1)/dy**2,
                 np.ones(ny)/dy**2], [-1, 0, 1], format='lil')
    Dy[0, :] = 0; Dy[-1, :] = 0
    Dyy[0, 1] = 2/dy**2; Dyy[-1, -2] = 2/dy**2
    sigma = alpha*np.exp(y)
    interior = np.ones(nf); interior[[0, -1]] = 0
    A = (kron(diags(.5*sigma*sigma), diags(f*f)@Dff)
         + kron(.5*nu*nu*(Dyy.tocsr()-Dy.tocsr()), diags(interior))
         + kron(diags(rho*nu*sigma)@Dy.tocsr(), diags(f)@Df)).tocsc()
    if y_boundary == 'frozen':
        evolve = np.ones(A.shape[0])
        evolve[:nf] = 0; evolve[-nf:] = 0
        A = (diags(evolve)@A).tocsc()
    dt = T/nt
    identity = eye(A.shape[0], format='csc')
    lu = splu(identity-.5*dt*A)
    rhs_operator = identity+.5*dt*A
    payoff = np.stack([np.maximum(f-1, 0), np.maximum(1-f, 0)], axis=-1)
    payoff[n_left] = (f[n_left+1]-f[n_left])/8
    values = np.tile(payoff, (ny+1, 1))
    values = lu.solve(lu.solve(values))
    for _ in range(nt-1):
        values = lu.solve(rhs_operator@values)
    unit = values.reshape(ny+1, nf, 2)[ny//2]
    values = CubicSpline(f, unit, axis=0)(1/np.asarray(K))*np.asarray(K)[:, None]
    call, put = values.T
    price = np.where(np.asarray(K)<1, put, call)
    return dict(price=price.tolist(), call=call.tolist(), put=put.tolist(),
                iv=invert_otm(price, np.asarray(K), T).tolist(),
                f_points=nf, f_max=float(f[-1]), y_points=ny+1,
                lu_nonzeros=lu.L.nnz+lu.U.nnz)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-left', type=int, default=80)
    parser.add_argument('--ny', type=int, default=100)
    parser.add_argument('--nt', type=int, default=400)
    parser.add_argument('--f-max', type=float, default=32.)
    parser.add_argument('--y-width', type=float, default=4.5)
    parser.add_argument('--y-boundary', choices=['reflecting', 'frozen'], default='reflecting')
    parser.add_argument('--name', default='coarse')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root/'results/fdm_paper_direct/figure7.json').read_text())
    K = np.asarray(source['K'])
    p = source['parameters']
    settings = dict(n_left=args.n_left, ny=args.ny, nt=args.nt, f_max=args.f_max,
                    y_width=args.y_width, y_boundary=args.y_boundary)
    start = perf_counter()
    result = forward_price(K, **p, **settings)
    result.update(seconds=perf_counter()-start, K=K.tolist(), parameters=p, settings=settings,
                  method='Unsplit forward-coordinate CN, two initial BE half steps; direct call/put.',
                  limitation='Independent numerical crosscheck, not an exact SABR oracle; finite forward/Y domains.')
    result['delta_current_iv_bp'] = ((np.asarray(result['iv'])-source['iv_estimate'])*1e4).tolist()
    result['delta_paper_iv_bp'] = ((np.asarray(result['iv'])-source['paper_fdm_iv'])*1e4).tolist()
    output=root/'results/fdm_diagnosis'
    output.mkdir(parents=True, exist_ok=True)
    (output/f'independent_{args.name}.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({k:result[k] for k in ['seconds','f_points','f_max','lu_nonzeros']}, indent=2))
    print('Max IV difference from current:',max(abs(np.asarray(result['delta_current_iv_bp']))))
    print('First node delta current/paper:',result['delta_current_iv_bp'][0],result['delta_paper_iv_bp'][0])


if __name__ == '__main__':
    main()
