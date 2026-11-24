import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

N_IN, N_OUT = 14, 10
LR_PATIENCE = 100  # epochs without improvement before the learning rate is halved
STOP_PATIENCE = 500  # epochs without improvement before training stops
CKPT_EVERY = 10
MAX_EPOCHS = 20000


def log_features(x: torch.Tensor) -> torch.Tensor:
    """(T, alpha, nu, rho, K1..K10) -> (sqrt T, alpha, nu_1M, rho, ln K1..ln K10), F = 1.

    Spreads the inputs that min-max scaling otherwise squeezes near 0: nu reaches 22 at T = 1D and the top
    strikes 20, so short-dated smiles used a few % of each column's range. nu_1M inverts eq. 43.
    """
    T = x[:, :1]
    return torch.cat((T.sqrt(), x[:, 1:2], x[:, 2:3] * (12 * T).sqrt(), x[:, 3:4], x[:, 4:].log()), dim=1)


class SABRNet(nn.Module):
    def __init__(self, hidden: int = 1000, log_inputs: bool = False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_IN, hidden), nn.Softplus(), nn.Linear(hidden, N_OUT), nn.Softplus())
        self.features = log_features if log_inputs else nn.Identity()
        # min-max input scaling (of the features), stored in the state dict; identity until set_input_range()
        self.register_buffer("x_lo", torch.zeros(N_IN))
        self.register_buffer("x_hi", torch.ones(N_IN))

    def set_input_range(self, X: np.ndarray) -> None:
        """Scale inputs to [0, 1] using the min and max of X (the training set)."""
        F = self.features(torch.as_tensor(X, dtype=torch.float32))
        self.x_lo.copy_(F.min(0).values)
        self.x_hi.copy_(F.max(0).values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((self.features(x) - self.x_lo) / (self.x_hi - self.x_lo))

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self(torch.as_tensor(X, dtype=torch.float32)).numpy()


def train(model: SABRNet, X: np.ndarray, Y: np.ndarray, epochs: int = MAX_EPOCHS, batch_size: int = 32,
          lr: float = 1e-3, ckpt: Path | None = None) -> list[float]:
    X, Y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)
    M = Y.isfinite()  # missing labels (NaN) are left out of the loss; all-finite data gives plain MSE
    Y = Y.nan_to_num(0.0)

    opt = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=LR_PATIENCE, threshold=1e-4)
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
            m = M[idx]
            loss = ((model(X[idx]) - Y[idx]) ** 2 * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        losses.append(total / len(X))
        print(f"epoch {len(losses):4d}/{epochs}  loss {losses[-1]:.3e}  lr {opt.param_groups[0]['lr']:.1e}", flush=True)

        sched.step(losses[-1])
        if sched.best == losses[-1]:  # the scheduler just recorded a new best
            best_epoch = len(losses)
        if len(losses) - best_epoch >= STOP_PATIENCE:  # no new best despite the cuts: converged
            print(f"stopped at epoch {len(losses)}, best loss {sched.best:.3e}", flush=True)
            stopped = True
            break
        if ckpt and len(losses) % CKPT_EVERY == 0:
            save()
    if ckpt:
        save(stopped)
    return losses

