"""Hagan (2002) SABR implied vol and the McGhee (2018) training-set generator.

Run `uv run python -m src.hagan` to regenerate data/hagan_1d_1y.npz and the CSVs.
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
N_STRIKES = 10
COLUMNS_X = ["T", "alpha", "nu", "rho"] + [f"K{i}" for i in range(1, N_STRIKES + 1)]
COLUMNS_Y = [f"iv{i}" for i in range(1, N_STRIKES + 1)]


def z_over_x(z, rho):
    z, rho = np.broadcast_arrays(np.asarray(z, float), np.asarray(rho, float))
    out = np.empty_like(z)
    small = np.abs(z) < 1e-6
    zs, rs = z[small], rho[small]
    out[small] = 1 - rs * zs / 2 + (2 - 3 * rs**2) * zs**2 / 12  # Taylor expansion around z=0
    zl, rl = z[~small], rho[~small]
    out[~small] = zl / np.log((np.sqrt(1 - 2 * rl * zl + zl**2) + zl - rl) / (1 - rl))
    return out


def hagan_iv(F, K, T, alpha, beta, rho, nu):
    K = np.asarray(K, float)
    L = np.log(F / K)
    p = (F * K) ** ((1 - beta) / 2)
    z = nu / alpha * p * L
    D = p * (1 + (1 - beta) ** 2 / 24 * L**2 + (1 - beta) ** 4 / 1920 * L**4)
    A = (1 - beta) ** 2 * alpha**2 / (24 * p**2) + rho * beta * nu * alpha / 4 / p + (2 - 3 * rho**2) * nu**2 / 24
    return alpha / D * z_over_x(z, rho) * (1 + A * T)


def check_against_paper():
    """Max abs error (bp) vs. McGhee (2018) Table 6."""
    K = np.array([0.4419, 0.5152, 0.6007, 0.7004, 0.8166, 0.9520, 1.1100, 1.2941, 1.5088, 1.7591, 2.0509])
    iv_paper = np.array([33.47, 29.66, 25.70, 21.54, 17.14, 12.64, 10.49, 12.82, 16.00, 19.19, 22.31]) / 100
    iv = hagan_iv(1.0, K, 1.6316, 0.108266, 1.0, -0.464053, 0.887903)
    return np.max(np.abs(iv - iv_paper)) * 1e4


def make_strikes(F, T, alpha, rho, nu):
    """10 strikes spanning ±3.5 sd of the unconditional variance (paper Tables 3-4)."""
    u = nu**2 * T
    v_bar = alpha**2 * np.expm1(u) / u
    delta_sigma = alpha * np.expm1(-u / 2 + 1.5 * nu * T**0.5)
    n = np.linspace(-3.5, 3.5, N_STRIKES)
    return F * np.exp(-0.5 * v_bar * T + rho / nu * delta_sigma + n * np.sqrt(v_bar * T))


def sample_parameters(n, t_min, t_max, seed):
    rng = np.random.default_rng(seed)
    T = rng.uniform(t_min, t_max, n)
    alpha = rng.uniform(0.05, 0.5, n)
    nu_1m = rng.uniform(0.05, 4, n)
    rho = rng.uniform(-0.9, 0.9, n)
    return np.column_stack((T, alpha, nu_1m * np.sqrt(1 / 12 / T), rho))


def grid_parameters(t_min, t_max):
    T, alpha, nu_1m, rho = np.meshgrid(
        np.linspace(t_min, t_max, 100),
        np.linspace(0.05, 0.5, 10),
        np.linspace(0.05, 4, 10),
        np.linspace(-0.9, 0.9, 10),
        indexing="ij",
    )
    return np.column_stack((T.ravel(), alpha.ravel(), (nu_1m * np.sqrt(1 / 12 / T)).ravel(), rho.ravel()))


def build_dataset(P):
    """P columns (T, alpha, nu, rho) -> X = [P | K1..K10], y = Hagan IVs (F=1, beta=1)."""
    T, alpha, nu, rho = (P[:, [i]] for i in range(4))
    K = make_strikes(1.0, T, alpha, rho, nu)
    return np.hstack((P, K)), hagan_iv(1.0, K, T, alpha, 1.0, rho, nu)


def generate(t_min=1 / 365, t_max=1.0, seeds=(42, 43, 44)):
    train_seed, test_seed, shuffle_seed = seeds
    P_train = np.vstack((grid_parameters(t_min, t_max), sample_parameters(150_000, t_min, t_max, train_seed)))
    order = np.random.default_rng(shuffle_seed).permutation(len(P_train))
    P_train[order] = P_train
    X_train, y_train = build_dataset(P_train)
    X_test, y_test = build_dataset(sample_parameters(50_000, t_min, t_max, test_seed))
    return {
        "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
        "t_range": np.array([t_min, t_max]), "seeds": np.array(seeds),
        "strike_rule": "table_3_4_unconditional_variance",
    }


def to_csv(data, out_dir=DATA_DIR):
    out_dir.mkdir(exist_ok=True)
    header = ",".join(COLUMNS_X + COLUMNS_Y)
    for split in ("train", "test"):
        table = np.hstack((data[f"X_{split}"], data[f"y_{split}"]))
        np.savetxt(out_dir / f"hagan_1d_1y_{split}.csv", table, delimiter=",", header=header, comments="", fmt="%.17g")


def load_csv(split, data_dir=DATA_DIR):
    table = np.loadtxt(data_dir / f"hagan_1d_1y_{split}.csv", delimiter=",", skiprows=1)
    return table[:, : len(COLUMNS_X)], table[:, len(COLUMNS_X) :]


if __name__ == "__main__":
    print(f"Max error vs. paper Table 6: {check_against_paper():.2f} bp")
    data = generate()
    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(DATA_DIR / "hagan_1d_1y.npz", **data)
    to_csv(data)
