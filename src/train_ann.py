"""Train several hidden sizes in parallel, one process (one core) per size.

    uv run python -m src.train_ann 250 500 750 1000
    tail -f data/ann_*.log        # progress

data/ann_{N}.pt is a full checkpoint (weights, Adam state, schedule), written every 10 epochs, in the
format the notebook loads. Rerunning resumes from it; sizes that already auto-stopped are skipped.
Delete the file to retrain from scratch. Recipe and schedule: see src/model.py.
"""

import argparse
import contextlib
from multiprocessing import Pool

import numpy as np
import torch

from src.hagan import DATA_DIR
from src.model import SABRNet, train


def finished(h: int, epochs: int) -> bool:
    path = DATA_DIR / f"ann_{h}.pt"
    if not path.exists():
        return False
    c = torch.load(path)
    return c["stopped"] or len(c["losses"]) >= epochs


def train_one(h: int, epochs: int) -> str:
    torch.set_num_threads(1)  # batch-32 steps are overhead-bound; extra threads only add contention
    torch.manual_seed(h)
    d = np.load(DATA_DIR / "hagan_1d_1y.npz")
    net = SABRNet(h)
    net.set_input_range(d["X_train"])  # overwritten by the checkpoint on resume (same values)
    with open(DATA_DIR / f"ann_{h}.log", "a", buffering=1) as log, contextlib.redirect_stdout(log):
        losses = train(net, d["X_train"], d["y_train"], epochs=epochs, log_every=1, ckpt=DATA_DIR / f"ann_{h}.pt")
    return f"N={h}: done at epoch {len(losses)}, final loss {losses[-1]:.3e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", type=int, nargs="+")
    ap.add_argument("--epochs", type=int, default=20000, help="cap on total epochs; runs normally auto-stop before")
    args = ap.parse_args()
    todo = [h for h in args.sizes if not finished(h, args.epochs)]
    print(f"training {todo}, skipping {sorted(set(args.sizes) - set(todo))} (already finished)")
    with Pool(max(len(todo), 1)) as pool:
        for msg in pool.starmap(train_one, [(h, args.epochs) for h in todo]):
            print(msg)
