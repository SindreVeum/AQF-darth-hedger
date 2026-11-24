import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from src import integration as I
from src.config import MODEL_DIR
from src.hagan import hagan_iv, make_strikes
from src.model import SABRNet


BOUNDS = np.array([[0.05, 0.5], [0.05, 4.0], [-0.9, 0.9]])
STARTS = np.array([[0.15, 0.5, -0.5], [0.25, 1.5, 0.0], [0.35, 2.5, 0.5]])


@dataclass
class Quotes:
    maturity: float
    forward: float
    strikes: np.ndarray
    ivs: np.ndarray

    def __post_init__(self):
        self.strikes = np.asarray(self.strikes, dtype=float).copy()
        self.ivs = np.asarray(self.ivs, dtype=float).copy()
        if not np.isfinite(self.maturity) or not 1 / 365 <= self.maturity <= 1:
            raise ValueError("Maturity must be within the completed ANN window [1D, 1Y].")
        if not np.isfinite(self.forward) or self.forward <= 0:
            raise ValueError("Forward must be positive and finite.")
        if (self.strikes.ndim != 1 or len(self.strikes) < 3
                or self.ivs.shape != self.strikes.shape
                or not np.isfinite(self.strikes).all() or not np.isfinite(self.ivs).all()
                or np.any(self.strikes <= 0) or np.any(self.ivs <= 0)
                or np.any(np.diff(self.strikes) <= 0)):
            raise ValueError("Require at least three sorted, distinct positive strikes and positive finite decimal IVs.")

    @property
    def moneyness(self):
        return self.strikes / self.forward


def load_ann(path=MODEL_DIR / "ann_log_750_ln.pt", hidden=750):
    """Load once, outside the timed fit; existing checkpoint format is unchanged."""
    net = SABRNet(hidden, log_inputs=True)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state"])
    return net.eval()


def strike_grid(maturity, parameters):
    alpha, nu1m, rho = np.asarray(parameters, dtype=float)
    nu = nu1m / np.sqrt(12 * maturity)
    return make_strikes(1.0, maturity, alpha, rho, nu)


def coverage(quotes, parameters):
    """Nonnegative margins mean that the ANN spline covers every fixed quote strike."""
    grid = strike_grid(quotes.maturity, parameters)
    return np.array([np.log(quotes.moneyness[0] / grid[0]),
                     np.log(grid[-1] / quotes.moneyness[-1])])


def predict(quotes, parameters, method, net=None):
    parameters = np.asarray(parameters, dtype=float)
    if (parameters.shape != (3,) or not np.isfinite(parameters).all()
            or np.any(parameters < BOUNDS[:, 0]) or np.any(parameters > BOUNDS[:, 1])):
        raise ValueError("Parameters must be (alpha, nu_1M, rho) inside the training bounds.")
    alpha, nu1m, rho = parameters
    T, K = quotes.maturity, quotes.moneyness
    nu = nu1m / np.sqrt(12 * T)
    if method == "hagan":
        ivs = hagan_iv(1.0, K, T, alpha, 1.0, rho, nu)
    elif method == "integration":
        # Fixed GH128 keeps the optimizer's objective smooth; the synthetic labels also pass GH64/128.
        ivs = I.invert_otm(I.quadrature_price(1.0, K, T, alpha, rho, nu, I.N_HIGH), K, T)
    elif method == "ann":
        if net is None:
            raise ValueError("Load the integration ANN before fitting.")
        if np.any(coverage(quotes, parameters) < -1e-10):
            raise ValueError("Quote strikes fall outside the ANN grid; extrapolation is disabled.")
        grid = strike_grid(T, parameters)
        values = net.predict(np.r_[T, alpha, nu, rho, grid][None])[0]
        ivs = CubicSpline(grid, values)(K, extrapolate=False)
    else:
        raise ValueError(f"Unknown method: {method}")
    if not np.isfinite(ivs).all() or np.any(ivs <= 0):
        raise ValueError("Model returned invalid IVs.")
    return ivs


@dataclass
class FitResult:
    method: str
    parameters: np.ndarray
    fitted_ivs: np.ndarray
    mae_bp: float
    rmse_bp: float
    success: bool
    message: str
    evaluations: int
    seconds: float


def calibrate(quotes, method, net=None, maxiter=150):
    """Equal-weight IV least squares, three starts, identical settings for every method.

    SLSQP works in [0,1]^3; central differences resolve float32 ANN outputs
    without the bias of a large one-sided step.
    Runtime and evaluation counts include all starts and final validation.
    """
    if method not in ("hagan", "integration", "ann"):
        raise ValueError(f"Unknown method: {method}")
    if method == "ann" and net is None:
        raise ValueError("Load the integration ANN before fitting.")
    lo, width = BOUNDS[:, 0], np.diff(BOUNDS, axis=1).ravel()
    evaluations = 0
    started = perf_counter()

    def objective(x):
        nonlocal evaluations
        evaluations += 1
        try:
            # Optimization may briefly step outside the coverage constraint. Permit that
            # for the direct methods; reject ANN extrapolation with a finite penalty.
            fitted = predict(quotes, lo + x * width, method, net)
            return np.mean(((fitted - quotes.ivs) / 0.01) ** 2)
        except (ValueError, FloatingPointError):
            return 1e8 + 1e6 * np.square(np.minimum(coverage(quotes, lo + x * width), 0)).sum()

    constraint = {"type": "ineq", "fun": lambda x: coverage(quotes, lo + x * width)}
    candidates, messages = [], []
    for start in STARTS:
        opt = minimize(objective, (start - lo) / width, method="SLSQP", jac="3-point", bounds=[(0, 1)] * 3,
                       constraints=constraint,
                       options={"maxiter": maxiter, "ftol": 1e-8, "finite_diff_rel_step": 1e-3})
        parameters = lo + opt.x * width
        messages.append(str(opt.message))
        if opt.success and np.all(coverage(quotes, parameters) >= 0):
            evaluations += 1
            try:
                fitted = predict(quotes, parameters, method, net)
                candidates.append((np.mean((fitted - quotes.ivs) ** 2), parameters, fitted))
            except (ValueError, FloatingPointError):
                pass
    elapsed = perf_counter() - started
    if not candidates:
        return FitResult(method, np.full(3, np.nan), np.full_like(quotes.ivs, np.nan),
                         np.nan, np.nan, False, "No valid converged fit. Starts: " + "; ".join(messages),
                         evaluations, elapsed)
    mse, parameters, fitted = min(candidates, key=lambda item: item[0])
    return FitResult(method, parameters, fitted, np.mean(np.abs(fitted - quotes.ivs)) * 1e4,
                     np.sqrt(mse) * 1e4, True, "Converged", evaluations, elapsed)


def synthetic_quotes(seed=2026, noise_bp=1.0):
    """Six moderate smiles, each with clean and noisy versions; no global RNG state."""
    rng = np.random.default_rng(seed)
    cases = []
    for i, T in enumerate((14 / 365, 0.25, 0.5, 1.0, 0.25, 1.0)):
        parameters = np.array([rng.uniform(0.14, 0.30), rng.uniform(0.6, 1.6), rng.uniform(-0.55, 0.35)])
        alpha, nu1m, rho = parameters
        K = np.exp(np.linspace(-0.8, 0.8, 7) * alpha * np.sqrt(T))
        clean = I.label_row(np.r_[T, alpha, nu1m / np.sqrt(12 * T), rho, K])
        for kind, ivs in (("noiseless", clean), ("noisy", clean + rng.normal(0, noise_bp / 1e4, len(K)))):
            cases.append((f"smile-{i + 1}", kind, Quotes(T, 100.0, 100 * K, ivs), parameters.copy()))
    return cases


def benchmark(repeats=3, output=Path("wiki/research/results/calibration.csv")):
    """Fit six noisy smiles; write median wall time across repeated fits per case/method."""
    torch.set_num_threads(1)
    net = load_ann()
    cases = [case for case in synthetic_quotes() if case[1] == "noisy"]
    for method in ("hagan", "integration", "ann"):
        predict(cases[0][2], STARTS[1], method, net)  # untimed quadrature/Torch warm-up

    rows = []
    for case, kind, quotes, truth in cases:
        for method in ("hagan", "integration", "ann"):
            fits = [calibrate(quotes, method, net) for _ in range(repeats)]
            result = fits[0]  # deterministic fit; time varies between repetitions
            success = all(f.success for f in fits)
            row = {"case": case, "quotes": kind, "maturity": quotes.maturity, "method": method,
                   "success": success,
                   "message": result.message if success else "; ".join(f.message for f in fits if not f.success),
                   "mae_bp": result.mae_bp, "rmse_bp": result.rmse_bp,
                   "seconds": float(np.median([f.seconds for f in fits])),
                   "evaluations": result.evaluations, "repeats": repeats}
            row.update(zip(("alpha", "nu1m", "rho"), result.parameters))
            row.update(zip(("true_alpha", "true_nu1m", "true_rho"), truth))
            rows.append(row)
            print(f"{case} {kind:9s} {method:11s}: success={row['success']} "
                  f"RMSE={row['rmse_bp']:.3f} bp, {row['seconds']:.3f} s", flush=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    benchmark()
