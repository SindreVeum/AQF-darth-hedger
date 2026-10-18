import os
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch import nn


class SABRNet(nn.Module):
    def __init__(self, hidden: int = 1000, n_in: int = 14, n_out: int = 10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.Softplus(), nn.Linear(hidden, n_out), nn.Softplus())
        # min-max input scaling, stored in the state dict; identity until set_input_range()
        self.register_buffer("x_lo", torch.zeros(n_in))
        self.register_buffer("x_hi", torch.ones(n_in))

    def set_input_range(self, X: np.ndarray) -> None:
        """Scale inputs to [0, 1] using the min and max of X (the training set)."""
        self.x_lo.copy_(torch.as_tensor(X.min(0), dtype=torch.float32))
        self.x_hi.copy_(torch.as_tensor(X.max(0), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.x_lo) / (self.x_hi - self.x_lo))

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self(torch.as_tensor(X, dtype=torch.float32)).numpy()


def train(model: SABRNet, X: np.ndarray, Y: np.ndarray, epochs: int = 20000, batch_size: int = 32,
          lr: float = 1e-3, patience: int = 100, stop_patience: int = 500, log_every: int = 0,
          ckpt: Path | None = None, ckpt_every: int = 10) -> list[float]:
    X, Y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)
    # fused: one kernel for the whole update; same algorithm, ~2x faster batch-32 steps on CPU
    opt = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
    # halve the rate after `patience` epochs without a 1e-4 relative improvement
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=patience, threshold=1e-4)
    losses, best_epoch = [], 0
    if ckpt and ckpt.exists():
        c = torch.load(ckpt)
        model.load_state_dict(c["state"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        losses, best_epoch = c["losses"], c["best_epoch"]
        print(f"resumed at epoch {len(losses)}, lr {opt.param_groups[0]['lr']:.1e}", flush=True)

    def save(stopped: bool = False) -> None:
        tmp = ckpt.with_suffix(".tmp")  # write then rename: an interrupt never leaves a broken file
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "losses": losses, "best_epoch": best_epoch, "stopped": stopped}, tmp)
        os.replace(tmp, ckpt)

    stopped = False
    while len(losses) < epochs:
        perm, total = torch.randperm(len(X)), 0.0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            loss = nn.functional.mse_loss(model(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        losses.append(total / len(X))
        if log_every and len(losses) % log_every == 0:
            print(f"epoch {len(losses):4d}/{epochs}  loss {losses[-1]:.3e}  lr {opt.param_groups[0]['lr']:.1e}", flush=True)

        sched.step(losses[-1])
        if sched.best == losses[-1]:  # the scheduler just recorded a new best
            best_epoch = len(losses)
        if len(losses) - best_epoch >= stop_patience:  # no new best despite the cuts: converged
            print(f"stopped at epoch {len(losses)}, best loss {sched.best:.3e}", flush=True)
            stopped = True
            break
        if ckpt and len(losses) % ckpt_every == 0:
            save()
    if ckpt:
        save(stopped)
    return losses


def smile(strikes: np.ndarray, vols: np.ndarray) -> CubicSpline:
    """§4.3: spline for one smile, strikes (10,) and vols (10,) -> callable K -> vol."""
    return CubicSpline(strikes, vols)
