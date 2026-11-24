import argparse
import contextlib
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import torch

from src.config import DATA_DIR, MODEL_DIR, LOG_DIR, SIZES
from src.model import MAX_EPOCHS, SABRNet, train


@dataclass
class Run:
    data: str  # data/datasets/<data>.npz
    tag: str  # output ann_[log_]<N>[_<tag>].pt
    init: str | None = None  # warm start from the checkpoint with this tag (§5.3)
    lr: float = 1e-3


RUNS = {
    "hagan_1d_1y": Run("hagan_1d_1y", tag=""),
    "hagan_1y_2y": Run("hagan_1y_2y", tag="1y2y"),
    "ln_1d_1y": Run("ln_parity_1d_1y", tag="ln", init="", lr=1e-4),
    "ln_1y_2y": Run("ln_parity_1y_2y", tag="ln_1y2y", init="1y2y", lr=1e-4),
}


def name(h: int, log_inputs: bool, tag: str) -> str:
    return (f"ann_log_{h}" if log_inputs else f"ann_{h}") + (f"_{tag}" if tag else "")


def finished(h: int, log_inputs: bool, run: Run) -> bool:
    path = MODEL_DIR / f"{name(h, log_inputs, run.tag)}.pt"
    if not path.exists():
        return False
    c = torch.load(path)
    return c["stopped"] or len(c["losses"]) >= MAX_EPOCHS


def train_one(h: int, log_inputs: bool, run: Run) -> str:
    torch.set_num_threads(1)  # batch-32 steps are overhead-bound; extra threads only add contention
    torch.manual_seed(h)
    d = np.load(DATA_DIR / f"{run.data}.npz")

    net = SABRNet(h, log_inputs=log_inputs)
    net.set_input_range(d["X_train"])
    if run.init is not None:  # keeps the input scaling of the warm-start network
        net.load_state_dict(torch.load(MODEL_DIR / f"{name(h, log_inputs, run.init)}.pt")["state"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = name(h, log_inputs, run.tag)
    with open(LOG_DIR / f"{out}.log", "a", buffering=1) as log, contextlib.redirect_stdout(log):
        losses = train(net, d["X_train"], d["y_train"], lr=run.lr, ckpt=MODEL_DIR / f"{out}.pt")
    return f"N={h}: done at epoch {len(losses)}, final loss {losses[-1]:.3e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=RUNS)
    ap.add_argument("sizes", type=int, nargs="*", default=SIZES)
    ap.add_argument("--log-inputs", action="store_true", help="train on (sqrt T, alpha, nu_1M, rho, ln K)")
    args = ap.parse_args()

    run = RUNS[args.run]
    todo = [h for h in args.sizes if not finished(h, args.log_inputs, run)]
    print(f"training {todo}, skipping {sorted(set(args.sizes) - set(todo))} (already finished)")
    with Pool(max(len(todo), 1)) as pool:
        for msg in pool.starmap(train_one, [(h, args.log_inputs, run) for h in todo]):
            print(msg)
