"""McGhee §3.2 conditional-lognormal integration (F=1, beta=1).

    uv run python -m src.integration --workers 4

Labels both paper maturity windows; GH64/128 labels must agree within 0.1 IV bp (else GH128/256 is tried).
Positive-rho OTM calls come from the put integral via put-call parity, C = P + F - K: the fitted
LN distribution has a divergent call tail (E[F_T] = inf), while the put integrand is bounded.
Labels are saved in resumable chunks, so an interrupted run continues where it stopped.
"""

import argparse
import os
from functools import lru_cache
from multiprocessing import Pool
from time import perf_counter

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, logsumexp, ndtr, roots_hermitenorm, roots_legendre

from src.config import DATA_DIR, WINDOWS
from src.hagan import generate as hagan_dataset

N_LOW, N_HIGH, N_RETRY, IV_TOL_BP = 64, 128, 256, 0.1
CHUNK_SIZE = 100
CHUNK_DIR = DATA_DIR.parent / 'chunks' / 'ln_parity_parts'


def black_price(F, K, T, sigma, option='call'):
    F, K, T, sigma = np.broadcast_arrays(*[np.asarray(x, float) for x in (F, K, T, sigma)])
    std = sigma * np.sqrt(T)
    d1 = np.log(F / K) / std + std / 2
    d2 = d1 - std
    value = F * ndtr(d1) - K * ndtr(d2) if option == 'call' else K * ndtr(-d2) - F * ndtr(-d1)
    return np.maximum(value, 0)


def black_implied_vol(price, F, K, T, option='call'):
    if option not in ('call', 'put') or min(F, K, T) <= 0:
        raise ValueError('Require positive F,K,T and option call/put.')
    intrinsic = max(F - K, 0.0) if option == 'call' else max(K - F, 0.0)
    upper = F if option == 'call' else K
    if not np.isfinite(price) or not intrinsic <= price < upper:
        raise ValueError('Price outside finite Black IV bounds.')
    if price == intrinsic:
        return 0.0

    def objective(std):
        value = intrinsic if std == 0 else float(black_price(F, K, 1.0, std, option))
        return value - price

    hi = 1.0
    while objective(hi) < 0 and hi < 64:
        hi *= 2
    return brentq(objective, 0.0, hi, xtol=1e-13, rtol=1e-12) / np.sqrt(T)


@lru_cache(maxsize=8)
def legendre_rule(n=32):
    x, w = roots_legendre(n)
    return (x + 1) / 2, w / 2


@lru_cache(maxsize=8)
def normal_rule(n):
    z, w = roots_hermitenorm(n)
    return z, w / np.sqrt(2 * np.pi)


def conditional_mean_variance(alpha, nu, T, sigma_T):
    alpha, nu, T, sigma_T = np.broadcast_arrays(
        *[np.asarray(x, float) for x in (alpha, nu, T, sigma_T)])
    q = T + np.log(sigma_T / alpha) / nu**2
    a = -nu * q / np.sqrt(T)
    b = nu * (2 * T - q) / np.sqrt(T)
    mask = a > 0
    delta = np.empty_like(b)
    delta[mask] = ndtr(-a[mask]) - ndtr(-b[mask])
    delta[~mask] = ndtr(b[~mask]) - ndtr(a[~mask])
    return alpha**2 / T * np.sqrt(2 * np.pi * T) / (2 * nu) * np.exp(nu**2 * q**2 / (2 * T)) * delta


def conditional_second_moment(alpha, nu, T, sigma_T, n=32):
    alpha, nu, T, sigma_T = np.broadcast_arrays(
        *[np.asarray(x, dtype=float) for x in (alpha, nu, T, sigma_T)]
    )
    t, w = legendre_rule(n)
    u = (nu**2 * T)[..., None]
    ell = np.log(sigma_T / alpha)[..., None]
    g = np.exp(2 * u * t * (1 - t)) * np.expm1(2 * u * t) / (2 * u)
    return alpha**4 * np.sum(
        w * g * (np.exp(2 * ell * t) + np.exp(2 * ell * (2 - t))),
        axis=-1,
    )


def quadrature_price(F, K, T, alpha, rho, nu, n=128):
    """Undiscounted OTM prices under the paper's conditional LN approximation.

    For rho > 0 the call integral diverges, so calls are the put integral plus F - K
    (put-call parity, i.e. imposing E[F_T] = F). rho <= 0 calls use the call integral directly.
    """
    K = np.asarray(K, dtype=float)
    z, w = normal_rule(n)
    s = alpha * np.exp(-nu**2 * T / 2 + nu * np.sqrt(T) * z)
    m1 = conditional_mean_variance(alpha, nu, T, s)
    m2 = conditional_second_moment(alpha, nu, T, s)
    cv2 = (m2 - m1**2) / m1**2
    if np.any(~np.isfinite(cv2)) or np.any(m1 <= 0) or np.any(cv2 < -1e-10):
        raise ValueError('Invalid conditional moments.')
    tau2 = np.log1p(np.maximum(cv2, 0))
    v = m1[:, None] * np.exp(np.sqrt(tau2)[:, None] * z - tau2[:, None] / 2)
    log_f = np.log(F) + rho / nu * (s[:, None] - alpha) - rho**2 * v * T / 2
    std = np.sqrt((1 - rho**2) * v * T)[..., None]
    lf, lk = log_f[..., None], np.log(K)
    d1 = (lf - lk) / std + std / 2
    d2 = d1 - std
    call = (K >= F) & (rho <= 0)
    a = np.where(call, lf + log_ndtr(d1), lk + log_ndtr(-d2))
    b = np.where(call, lk + log_ndtr(d2), lf + log_ndtr(-d1))
    log_w = np.log(w[:, None]) + np.log(w[None, :])
    with np.errstate(divide='ignore', over='ignore'):
        price = np.exp(logsumexp(
            a + np.log(-np.expm1(np.minimum(b - a, 0))) + log_w[..., None], axis=(0, 1)))
    parity = (K >= F) & (rho > 0)
    return np.where(parity, price + F - K, price)


def invert_otm(prices, strikes, T):
    iv = np.full(len(strikes), np.nan)
    for j, (p, k) in enumerate(zip(prices, strikes)):
        if np.isfinite(p) and p > 0:
            try:
                iv[j] = black_implied_vol(p, 1., k, T, 'put' if k < 1 else 'call')
            except (ValueError, RuntimeError):
                pass
    return iv


def _ivs(row, n, cols=slice(None)):
    T, alpha, nu, rho = row[:4]
    K = row[4:][cols]
    return invert_otm(quadrature_price(1., K, T, alpha, rho, nu, n), K, T)


def label_row(row):
    """Ten IV labels from GH64/128 agreement, retrying failures with GH128/256; else NaN."""
    lo, hi = _ivs(row, N_LOW), _ivs(row, N_HIGH)
    ok = np.abs(hi - lo) * 1e4 <= IV_TOL_BP
    y = np.where(ok, hi, np.nan)
    retry = ~ok & np.isfinite(hi)
    if retry.any():
        top = _ivs(row, N_RETRY, retry)
        y[retry] = np.where(np.abs(top - hi[retry]) * 1e4 <= IV_TOL_BP, top, np.nan)
    return y


def check_against_paper():
    """Max IV error (bp) vs. rounded integration values in Table 6."""
    K = np.array([.4419, .5152, .6007, .7004, .8166, .9520, 1.1100, 1.2941, 1.5088, 1.7591, 2.0509])
    reference = np.array([29.97, 26.72, 23.33, 19.76, 16., 12.20, 10.51, 12.54, 15.24, 17.94, 20.56]) / 100
    prices = quadrature_price(1., K, 1.6316, .108266, -.464053, .887903)
    return np.max(np.abs(invert_otm(prices, K, 1.6316) - reference)) * 1e4


def check_against_2011():
    """Max IV error (bp) vs. McGhee (2011) slide 25, a rho > 0 smile (exercises the parity calls)."""
    K = np.array([.725, .8167, .9083, 1., 1.0917, 1.1833, 1.275, 1.3667, 1.4583, 1.55])
    reference = np.array([14.85, 12.82, 11.83, 12.67, 14.59, 16.65, 18.59, 20.38, 22.03, 23.55]) / 100
    prices = quadrature_price(1., K, 1., .12, .45, .85)
    return np.max(np.abs(invert_otm(prices, K, 1.) - reference)) * 1e4


def log10_mean_forward(T, alpha, rho, nu, n):
    """log10 of the n-node quadrature value of E[F_c]/F under the approximating law (F = 1).
    Proving divergence for rho>0 calls.
    """
    z, w = normal_rule(n)
    s = alpha * np.exp(-nu**2 * T / 2 + nu * np.sqrt(T) * z)
    m1 = conditional_mean_variance(alpha, nu, T, s)
    m2 = conditional_second_moment(alpha, nu, T, s)
    tau2 = np.log1p(np.maximum((m2 - m1**2) / m1**2, 0))
    v = m1[:, None] * np.exp(np.sqrt(tau2)[:, None] * z - tau2[:, None] / 2)
    log_f = rho / nu * (s[:, None] - alpha) - rho**2 * v * T / 2
    log_w = np.log(w[:, None]) + np.log(w[None, :])
    return logsumexp(log_f + log_w) / np.log(10)


def atomic_save(path, **arrays):
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def write_chunk(job):
    path, X = job
    y = np.array([label_row(row) for row in X])
    atomic_save(path, X=X, y=y)
    return len(X), int(np.isfinite(y).sum())


def label_split(pool, X, folder, label):
    """Label X chunk by chunk, skipping chunks already saved in folder."""
    folder.mkdir(parents=True, exist_ok=True)
    paths, jobs = [], []
    for first in range(0, len(X), CHUNK_SIZE):
        rows = X[first:first + CHUNK_SIZE]
        path = folder / f'{first:06d}_{first + len(rows):06d}.npz'
        paths.append(path)
        if not path.exists():
            jobs.append((path, rows))

    done = len(X) - sum(len(rows) for _, rows in jobs)
    print(f'{label}: resuming {done}/{len(X)} rows', flush=True)
    for count, valid in pool.imap_unordered(write_chunk, jobs):
        done += count
        print(f'{label}: {done}/{len(X)} rows; chunk {valid}/{count * 10} finite labels', flush=True)

    ys = []
    for first, path in zip(range(0, len(X), CHUNK_SIZE), paths):
        with np.load(path) as part:
            if not np.array_equal(part['X'], X[first:first + CHUNK_SIZE]):
                raise ValueError(f'Chunk does not match inputs: {path}')
            ys.append(part['y'])
    return np.vstack(ys)


def generate(workers=None):
    """Same inputs (sampling, seeds, row order) as the Hagan datasets -> ln_parity_<window>.npz."""
    started = perf_counter()
    with Pool(workers) as pool:
        for window, (lo, hi, seeds) in WINDOWS.items():
            data = hagan_dataset(lo, hi, seeds)
            for split in ('train', 'test'):
                y = data[f'y_{split}'] = label_split(pool, data[f'X_{split}'], CHUNK_DIR / window / split,
                                                     f'{window}/{split}')
                print(f'{window}/{split}: {np.isfinite(y).mean():.4%} finite labels, '
                      f'{np.isfinite(y).all(axis=1).mean():.4%} complete smiles', flush=True)
            data.update(method='ln_parity', n_low=N_LOW, n_high=N_HIGH, n_retry=N_RETRY,
                        iv_tol_bp=IV_TOL_BP, moment_n=32, positive_rho_calls='put_call_parity')
            target = DATA_DIR / f'ln_parity_{window}.npz'
            atomic_save(target, **data)
            print(f'Saved {target} ({(perf_counter() - started) / 3600:.2f} h elapsed)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    for label, check in (('Table 6 (2018)', check_against_paper), ('slide 25 (2011, rho > 0)', check_against_2011)):
        error = check()
        print(f'Max error vs. rounded {label} integration IVs: {error:.4f} bp', flush=True)
        if not np.isfinite(error) or error > 1:
            raise RuntimeError(f'{label} check failed (tolerance 1 IV bp).')
    generate(args.workers)
