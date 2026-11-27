"""Independent conditional randomized-QMC check of Figure 7 SABR prices.

Volatility GBM is sampled exactly at dyadic time nodes. Trapezoidal integrated
variance is refined on nested grids; the orthogonal asset Brownian motion is
integrated analytically. No conditional-LN approximation or parity is used.
"""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import log_ndtr, ndtri
from scipy.stats import qmc, t as student_t

from src.integration import black_implied_vol


def bridge(normals, maturity):
    """Brownian bridge ordered by scale on an equally spaced dyadic time grid."""
    paths, steps = normals.shape
    if steps < 4 or steps & (steps-1):
        raise ValueError('steps must be a power of two and at least four.')
    w = np.zeros((paths, steps+1))
    w[:, steps] = np.sqrt(maturity)*normals[:, 0]
    width, offset = steps, 1
    while width > 1:
        middle = np.arange(width//2, steps, width)
        left, right = middle-width//2, middle+width//2
        w[:, middle] = (w[:, left]+w[:, right])/2 + np.sqrt(maturity*width/(4*steps))*normals[:, offset:offset+len(middle)]
        offset += len(middle)
        width //= 2
    assert offset == steps
    return w


def conditional_prices(log_forward, variance, strikes):
    """Stable conditional Black OTM prices, with the initial forward equal to one."""
    std = np.sqrt(variance)[:, None]
    lf, lk = log_forward[:, None], np.log(strikes)[None, :]
    d1 = (lf-lk)/std + std/2
    d2 = d1-std
    is_call = strikes[None, :] >= 1.
    a = np.where(is_call, lf+log_ndtr(d1), lk+log_ndtr(-d2))
    b = np.where(is_call, lk+log_ndtr(d2), lf+log_ndtr(-d1))
    return np.exp(a)*(-np.expm1(np.minimum(b-a, 0.)))


def replicate(parameters, strikes, power, steps, seed):
    T, alpha, nu, rho = (parameters[k] for k in ('T', 'alpha', 'nu', 'rho'))
    sampler = qmc.Sobol(steps, scramble=True, seed=seed)
    uniforms = sampler.random_base2(power)
    normals = ndtri(np.clip(uniforms, np.finfo(float).eps, 1-np.finfo(float).eps))
    del uniforms
    w = bridge(normals, T)
    del normals
    sigma2 = alpha**2*np.exp(2*nu*w - nu**2*np.linspace(0., T, steps+1))
    del w
    terminal_sigma = np.sqrt(sigma2[:, -1])
    result = {}
    for count in (steps//4, steps//2, steps):
        values = sigma2[:, ::steps//count]
        integral = (T/count)*(values[:, 1:-1].sum(axis=1) + (values[:, 0]+values[:, -1])/2)
        lf = rho/nu*(terminal_sigma-alpha)-rho**2*integral/2
        prices = conditional_prices(lf, (1-rho**2)*integral, strikes)
        result[count] = dict(price=prices.mean(axis=0).tolist(), forward_mean=float(np.exp(lf).mean()))
    return result


def main(power=14, repeats=12, steps=512, seed=202611260,
         output=Path('results/fdm_diagnosis/mc_figure7.json')):
    if repeats < 4:
        raise ValueError('At least four independent scrambles are required.')
    source = Path('results/fdm_paper_direct')
    own = json.loads((source/'figure7.json').read_text(encoding='utf8'))
    ref = json.loads((source/'figure7_reference.json').read_text(encoding='utf8'))
    indices = np.array([0, 10, 12, 19])
    K = np.array(own['K'])[indices]
    T = own['parameters']['T']
    own_price = np.array(own['runs']['fine']['price'])[indices]
    own_iv = np.array(own['iv_estimate'])[indices]
    paper_iv = np.array(ref['paper_fdm_iv'])[indices]
    from src.integration import black_price
    paper_price = np.where(K < 1, black_price(1., K, T, paper_iv, 'put'),
                           black_price(1., K, T, paper_iv, 'call'))
    records = []
    started = perf_counter()
    for j in range(repeats):
        records.append(replicate(own['parameters'], K, power, steps, seed+j))
        print(f'Scramble {j+1}/{repeats}, elapsed {perf_counter()-started:.1f}s', flush=True)
    critical = float(student_t.ppf(.975, repeats-1))
    levels = {}
    def invert(price):
        return np.array([black_implied_vol(float(p), 1., float(k), T, 'put' if k < 1 else 'call')
                         for p, k in zip(price, K)])
    for count in (steps//4, steps//2, steps):
        r = np.array([item[count]['price'] for item in records])
        mean, se = r.mean(axis=0), r.std(axis=0, ddof=1)/np.sqrt(repeats)
        low, high = mean-critical*se, mean+critical*se
        levels[count] = dict(price=mean.tolist(), price_se=se.tolist(), price_ci95_low=low.tolist(),
                             price_ci95_high=high.tolist(), iv=invert(mean).tolist(),
                             iv_ci95_low=invert(low).tolist(), iv_ci95_high=invert(high).tolist(),
                             own_fdm_minus_qmc_in_se=((own_price-mean)/se).tolist(),
                             paper_fdm_minus_qmc_in_se=((paper_price-mean)/se).tolist(),
                             own_fdm_minus_qmc_iv_bp=((own_iv-invert(mean))*1e4).tolist(),
                             paper_fdm_minus_qmc_iv_bp=((paper_iv-invert(mean))*1e4).tolist())
    differences = {}
    for fine in (steps//2, steps):
        d = np.array([np.array(item[fine]['price'])-item[fine//2]['price'] for item in records])
        differences[f'{fine}-{fine//2}'] = dict(mean_price=d.mean(axis=0).tolist(),
                                              se_price=(d.std(axis=0, ddof=1)/np.sqrt(repeats)).tolist())
    result = dict(method='Independent randomized Sobol Brownian bridge conditional MC; nested trapezoidal variance',
                  parameters=own['parameters'], indices_zero_based=indices.tolist(), K=K.tolist(),
                  power=power, paths_per_scramble=2**power, independent_scrambles=repeats,
                  steps=steps, seed=seed, seconds=perf_counter()-started, student_t_critical=critical,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  own_fdm_price=own_price.tolist(), own_fdm_iv=own_iv.tolist(),
                  paper_fdm_iv=paper_iv.tolist(), paper_fdm_price_implied_by_plot=paper_price.tolist(),
                  levels=levels, paired_time_changes=differences, replicates=records,
                  limitations=['Student-t intervals estimate marginal randomization uncertainty from independent scrambles; they do not include time-discretization bias.',
                               'Nested time differences assess but do not rigorously bound trapezoidal variance bias.',
                               'Paper prices are Black prices implied by digitized Figure 7 IVs, not a published price table.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'K': K.tolist(), 'finest': levels[steps], 'paired_time_changes': differences}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--power', type=int, default=14)
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--seed', type=int, default=202611260)
    parser.add_argument('--output', type=Path, default=Path('results/fdm_diagnosis/mc_figure7.json'))
    args = parser.parse_args()
    main(args.power, args.repeats, args.steps, args.seed, output=args.output)
