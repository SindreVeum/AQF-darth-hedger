"""Compare the two routes to OTM calls (K >= F) for rho <= 0 in the integration scheme.

Route 1: call integral directly (what src/integration.py uses for rho <= 0).
Route 2: put integral + put-call parity, C = P + F - K (what src/integration.py uses for rho > 0).

Case A of the report (Table 6 of McGhee 2020). 
Writes results/call_vs_parity_rho_neg.json.
Also reports E[F_c] under the approximating law, which put-call parity assumes equals F.
"""
import json

import numpy as np
from scipy.special import log_ndtr, logsumexp

from src import integration as I
from src.config import RESULTS_DIR

F, T, ALPHA, RHO, NU = 1.0, 1.6316, 0.108266, -0.464053, 0.887903
STRIKES = np.array([0.4419, 0.5152, 0.6007, 0.7004, 0.8166, 0.9520,
                    1.1100, 1.2941, 1.5088, 1.7591, 2.0509])


def integral(K, n, use_call):
    """Undiscounted call (use_call) or put integral, same quadrature as I.quadrature_price."""
    z, w = I.normal_rule(n)
    s = ALPHA * np.exp(-NU**2 * T / 2 + NU * np.sqrt(T) * z)
    m1 = I.conditional_mean_variance(ALPHA, NU, T, s)
    m2 = I.conditional_second_moment(ALPHA, NU, T, s)
    tau2 = np.log1p(np.maximum((m2 - m1**2) / m1**2, 0))
    v = m1[:, None] * np.exp(np.sqrt(tau2)[:, None] * z - tau2[:, None] / 2)
    log_f = np.log(F) + RHO / NU * (s[:, None] - ALPHA) - RHO**2 * v * T / 2
    std = np.sqrt((1 - RHO**2) * v * T)[..., None]
    lf, lk = log_f[..., None], np.log(K)
    d1 = (lf - lk) / std + std / 2
    d2 = d1 - std
    a = lf + log_ndtr(d1) if use_call else lk + log_ndtr(-d2)
    b = lk + log_ndtr(d2) if use_call else lf + log_ndtr(-d1)
    log_w = np.log(w[:, None]) + np.log(w[None, :])
    price = np.exp(logsumexp(a + np.log(-np.expm1(np.minimum(b - a, 0))) + log_w[..., None],
                             axis=(0, 1)))
    return price, np.sum(np.exp(log_f + log_w))   # E[F_c] under the approximating law


def main():
    K = STRIKES[STRIKES >= F]
    out = {}
    for n in (64, 128, 256):
        call, ef = integral(K, n, True)
        put, _ = integral(K, n, False)
        parity = put + F - K
        iv_call, iv_par = I.invert_otm(call, K, T), I.invert_otm(parity, K, T)
        diff_bp = (iv_call - iv_par) * 1e4
        out[n] = dict(K=K.tolist(), iv_call_pct=(iv_call * 100).tolist(),
                      iv_parity_pct=(iv_par * 100).tolist(), diff_iv_bp=diff_bp.tolist(),
                      max_abs_diff_iv_bp=float(np.max(np.abs(diff_bp))),
                      max_abs_diff_price_bp=float(np.max(np.abs(call - parity)) * 1e4),
                      E_Fc=float(ef))
        print(f"n={n}: max |IV call - IV parity| = {out[n]['max_abs_diff_iv_bp']:.4f} bp, "
              f"max |C - C_parity| = {out[n]['max_abs_diff_price_bp']:.4f} price bp, E[F_c] = {ef:.7f}")
    print(f"\n{'K':>8} {'IV call %':>10} {'IV parity %':>12} {'diff bp':>9}   (n=128)")
    r = out[128]
    for k, a, b, d in zip(r['K'], r['iv_call_pct'], r['iv_parity_pct'], r['diff_iv_bp']):
        print(f"{k:8.4f} {a:10.4f} {b:12.4f} {d:9.4f}")
    (RESULTS_DIR / 'call_vs_parity_rho_neg.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
