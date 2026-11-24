"""Figures and tables for wiki/report/ann-section.md.

uv run python figures/report_figures.py   ->  wiki/report/figures/*.png, *.md
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODEL_DIR, DATA_DIR, SIZES
from src.model import SABRNet

OUT = Path(__file__).resolve().parents[1] / "wiki" / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TABLES = OUT.parents[1] / "research" / "results" / "report"
TABLES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
torch.set_num_threads(2)


def load(name, hidden, log):
    ckpt = torch.load(MODEL_DIR / f"{name}.pt", map_location="cpu", weights_only=True)
    net = SABRNet(hidden, log_inputs=log)
    net.load_state_dict(ckpt["state"])
    return net.eval(), ckpt


def predict(net, X):
    return np.vstack([net.predict(X[i:i + 4096]) for i in range(0, len(X), 4096)])


def metrics(e):
    e = np.abs(e[np.isfinite(e)])
    return {"MAE": e.mean(), "RMSE": np.sqrt((e ** 2).mean()), "p99": np.percentile(e, 99), "max": e.max()}


def minmax(x):
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))


# 1. Input distributions after min-max scaling: raw vs transformed --------------------------------
with np.load(DATA_DIR / "hagan_1d_1y.npz") as data:
    X = data["X_train"]
    shuffle_seed = int(data["seeds"][2])
# The generator assigns original rows to their shuffled positions.
order = np.random.default_rng(shuffle_seed).permutation(len(X))
random_rows = order[100_000:]
assert len(random_rows) == 150_000
T, nu, K = X[:, 0], X[:, 2], X[:, 4:]
pairs = [("Maturity", T, "$T$", np.sqrt(T), r"$\sqrt{T}$"),
         ("Vol of vol", nu, r"$\nu$", nu * np.sqrt(12 * T), r"$\nu_{1M}=\nu\sqrt{12T}$"),
         ("Strikes (scaled per column, then pooled)", K, "$K/F$", np.log(K), r"$\ln(K/F)$")]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
for ax, (title, raw, rl, tr, tl) in zip(axes, pairs):
    bins = np.linspace(0, 1, 51)
    ax.hist(minmax(raw)[random_rows].ravel(), bins, density=True, alpha=0.6, color="tab:gray", label=f"raw {rl}")
    ax.hist(minmax(tr)[random_rows].ravel(), bins, density=True, histtype="step", lw=1.8, color="tab:blue", label=f"transformed {tl}")
    ax.set(title=title, xlabel="min–max scaled value", yscale="linear")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("density")
fig.suptitle("Random training samples only (150,000 smiles); full-training-set min–max scaling")
fig.savefig(OUT / "input_distributions_random_linear.png")
plt.close(fig)

# share of data squeezed into the bottom 5% of the scaled range
rows = []
for title, raw, rl, tr, tl in pairs:
    rows.append({"input": title, "raw: share in lowest 5% of range": (minmax(raw) < 0.05).mean(),
                 "transformed: share in lowest 5% of range": (minmax(tr) < 0.05).mean()})
squeeze = pd.DataFrame(rows).set_index("input")

# 2. Training curves: Hagan pretraining and integration fine-tuning, raw vs log -------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
colors = dict(zip(SIZES, plt.cm.viridis(np.linspace(0, 0.85, 4))))
for ax, suffix, title in [(axes[0], "", "Hagan pretraining (1D–1Y)"),
                          (axes[1], "_ln", "Integration fine-tuning (1D–1Y)")]:
    for h in SIZES:
        for prefix, ls, lab in [("ann_log", "-", "log"), ("ann", "--", "raw")]:
            losses = torch.load(MODEL_DIR / f"{prefix}_{h}{suffix}.pt", map_location="cpu",
                                weights_only=True)["losses"]
            rmse_bp = np.sqrt(np.asarray(losses)) * 1e4
            ax.semilogy(np.arange(1, len(rmse_bp) + 1), rmse_bp, ls, color=colors[h], lw=1,
                        label=f"N={h} {lab}")
    ax.set(title=title, xlabel="epoch", ylabel="training RMSE (IV bp)")
    ax.grid(alpha=0.3, which="both")
axes[0].legend(fontsize=7, ncol=2)
fig.suptitle("Training loss as RMSE (solid: transformed inputs, dashed: raw inputs)")
fig.savefig(OUT / "training_curves.png")
plt.close(fig)

# 3. Test-set error tables -------------------------------------------------------------------------
def table(data, suffix):
    with np.load(DATA_DIR / data) as d:
        Xt, yt = d["X_test"], d["y_test"]
    out, preds = {}, {}
    for prefix, log, lab in [("ann", False, "raw"), ("ann_log", True, "transformed")]:
        for h in SIZES:
            net, ckpt = load(f"{prefix}_{h}{suffix}", h, log)
            p = predict(net, Xt)
            preds[(lab, h)] = p
            out[(lab, h)] = {**metrics((p - yt) * 1e4), "epochs": len(ckpt["losses"])}
    return pd.DataFrame(out).T.rename_axis(["inputs", "N"]), Xt, yt, preds


hagan_tab, *_ = table("hagan_1d_1y.npz", "")
ln_tab, Xi, yi, preds = table("ln_parity_1d_1y.npz", "_ln")
H = metrics((np.load(DATA_DIR / "hagan_1d_1y.npz")["y_test"] - yi) * 1e4)  # same inputs row for row


def to_md(df):
    df = df.reset_index()
    rows = [list(map(str, df.columns)), ["---"] * len(df.columns)] + df.astype(str).values.tolist()
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def md(df):
    df = df.copy()
    for c in ("MAE", "RMSE", "p99", "max"):
        df[c] = df[c].map(lambda v: f"{v:.2f}" if v < 100 else f"{v:.0f}")
    df["epochs"] = df["epochs"].map(lambda v: f"{int(v):,}")
    return to_md(df)


(OUT / "tables.md").write_text(
    "# Generated tables (figures/report_figures.py)\n\n"
    "## Input squeeze after min–max scaling\n\n" + to_md(squeeze.map(lambda v: f"{v:.1%}")) + "\n\n"
    "## Hagan stage: ANN vs Hagan formula, 1D–1Y test set (IV bp)\n\n" + md(hagan_tab) + "\n\n"
    "## Fine-tuned: ANN vs integration, 1D–1Y test set (IV bp)\n\n" + md(ln_tab) + "\n\n"
    f"Reference row, Hagan formula vs integration on the same labels: MAE {H['MAE']:.1f}, RMSE {H['RMSE']:.1f}, "
    f"p99 {H['p99']:.0f}, max {H['max']:.0f}.\n")

# 4. Error across the parameter domain: transformed vs raw, N=750 ---------------------------------
mask = np.isfinite(yi)
err = {lab: np.where(mask, np.abs(preds[(lab, 750)] - yi) * 1e4, np.nan) for lab in ("transformed", "raw")}
nu1m = Xi[:, 2] * np.sqrt(12 * Xi[:, 0])
params = [(Xi[:, 0], np.linspace(1 / 365, 1, 9), "Maturity $T$ (years)"),
          (100 * Xi[:, 1], np.linspace(5, 50, 9), r"Initial vol $\alpha$ (%)"),
          (100 * nu1m, np.linspace(5, 400, 9), r"Vol of vol $\nu_{1M}$ (%)"),
          (Xi[:, 3], np.linspace(-0.9, 0.9, 9), r"Correlation $\rho$")]
style = {"transformed": {"color": "tab:blue"}, "raw": {"color": "tab:gray"}}
fig = plt.figure(figsize=(12, 6.4), constrained_layout=True)
grid = fig.add_gridspec(2, 6)
axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]), fig.add_subplot(grid[1, 0:3]),
        fig.add_subplot(grid[1, 3:6])]
for ax, (values, edges, label) in zip(axes, params):
    centers = (edges[:-1] + edges[1:]) / 2
    for lab, e in err.items():
        g = pd.DataFrame({"b": pd.cut(np.repeat(values, 10), edges, include_lowest=True),
                          "e": e.ravel()}).groupby("b", observed=False)["e"]
        ax.plot(centers, g.mean(), "o-", ms=3, **style[lab], label=f"{lab}: mean")
        ax.plot(centers, g.quantile(0.99), "o--", ms=3, alpha=0.7, **style[lab], label=f"{lab}: p99")
    ax.set(xlabel=label, ylabel="|ANN − integration| (IV bp)", yscale="linear")
    ax.grid(alpha=0.3, which="both")
axes[0].legend(fontsize=8)
ax = axes[4]
pos = np.arange(1, 11)
for lab, e in err.items():
    ax.plot(pos, np.nanmean(e, 0), "o-", ms=3, **style[lab])
    ax.plot(pos, np.nanpercentile(e, 99, 0), "o--", ms=3, alpha=0.7, **style[lab])
ax.set(xlabel="strike position, low → high", ylabel="|ANN − integration| (IV bp)", xticks=pos, yscale="linear")
ax.grid(alpha=0.3, which="both")
fig.savefig(OUT / "error_by_parameter_linear.png")
plt.close(fig)

print((OUT / "tables.md").read_text())
print("saved:", *sorted(p.name for p in OUT.iterdir()))

# 5. Completed second-window models: the same held-out evaluation as above.
hagan_long, *_ = table("hagan_1y_2y.npz", "_1y2y")
ln_long, Xl, yl, pl = table("ln_parity_1y_2y.npz", "_ln_1y2y")
assert Xl.shape == (50000, 14)
assert np.isfinite(yl).all(), "Report test-set coverage before quoting pooled errors."
for prefix in ("ann", "ann_log"):
    for h in SIZES:
        _, c = load(f"{prefix}_{h}_ln_1y2y", h, prefix == "ann_log")
        assert c["stopped"] or len(c["losses"]) >= 20000
with np.load(DATA_DIR / "hagan_1y_2y.npz") as d:
    assert np.array_equal(d["X_test"], Xl)
    Hl = metrics((d["y_test"] - yl) * 1e4)
with (OUT / "tables.md").open("a") as f:
    f.write("\n## Hagan stage: 1Y–2Y (IV bp)\n\n" + md(hagan_long))
    f.write("\n\n## Fine-tuned: 1Y–2Y (IV bp), 50,000 complete test smiles\n\n" + md(ln_long))
    f.write(f"\n\nHagan vs integration, same test inputs: {Hl}.\n")
ln_long.to_csv(TABLES / "integration_1y2y.csv")

# Paper Figure 7: compare the ten learned nodes and the interpolated smile separately.
from src import integration as I
from src.hagan import hagan_iv, make_strikes
from scipy.interpolate import CubicSpline
T, alpha, nu, rho = 540 / 365, 0.30, 0.60, -0.35
K = make_strikes(1., T, alpha, rho, nu)
row = np.r_[T, alpha, nu, rho, K]
node_ref = I.label_row(row)
assert np.isfinite(node_ref).all()
Kf = np.geomspace(K[0], K[-1], 160)
dense_row = np.r_[T, alpha, nu, rho, Kf]
ref128, ref256 = I._ivs(dense_row, 128), I._ivs(dense_row, 256)
assert np.isfinite(ref256).all()
assert np.max(np.abs(ref256 - ref128)) * 1e4 <= I.IV_TOL_BP
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), constrained_layout=True)
axes[0].plot(Kf, 100 * ref256, color="black", label="Integration (direct)")
axes[0].plot(Kf, 100 * hagan_iv(1., Kf, T, alpha, 1., rho, nu), ":", color="tab:orange", label="Hagan")
example_rows = []
for prefix, log, lab, color in [("ann", False, "raw", "tab:gray"), ("ann_log", True, "transformed", "tab:blue")]:
    net, _ = load(f"{prefix}_750_ln_1y2y", 750, log)
    nodes = net.predict(row[None])[0]
    curve = CubicSpline(K, nodes)(Kf)
    axes[0].plot(Kf, 100 * curve, color=color, label=f"ANN N=750 ({lab}, spline)")
    axes[0].scatter(K, 100 * nodes, color=color, s=12)
    axes[1].plot(Kf, (curve - ref256) * 1e4, color=color, label=f"{lab}: spline vs direct")
    axes[1].scatter(K, (nodes - node_ref) * 1e4, color=color, s=12)
    example_rows.append({"inputs": lab, "node_mae_bp": metrics((nodes-node_ref)*1e4)["MAE"],
                         "node_max_bp": metrics((nodes-node_ref)*1e4)["max"],
                         "spline_mae_bp": metrics((curve-ref256)*1e4)["MAE"],
                         "spline_max_bp": metrics((curve-ref256)*1e4)["max"]})
axes[1].axhline(0, color="black", lw=.7)
for ax in axes:
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 3])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Normalized strike K/F (log scale)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=7)
axes[0].set_ylabel("Implied volatility (%)")
axes[1].set_ylabel("ANN − integration (IV bp)")
fig.suptitle("Paper Figure 7 case: T=540D, α=30%, ν=60%, ρ=−35%; dots: ten network strikes")
fig.savefig(OUT / "paper_figure7.png")
plt.close(fig)
pd.DataFrame(example_rows).to_csv(TABLES / "paper_figure7.csv", index=False)
print("SECOND WINDOW RESULTS\n", ln_long.to_string())
print("HAGAN BASELINE", Hl)
print("FIGURE 7\n", pd.DataFrame(example_rows).to_string(index=False))
