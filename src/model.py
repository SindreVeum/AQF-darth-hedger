"""McGhee (2018) §4.1: (T, σ₀, ξ, ρ, K₁..K₁₀) → (σ₁..σ₁₀) with one softplus hidden layer."""

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch import nn


class SABRNet(nn.Module):
    def __init__(self, hidden: int = 1000, n_in: int = 14, n_out: int = 10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.Softplus(), nn.Linear(hidden, n_out))
        # Scaling is not specified in the paper; we standardise inputs and outputs.
        # Stored as buffers so it is saved with the weights and reused when fine-tuning.
        for name, n in [("x_mu", n_in), ("x_sd", n_in), ("y_mu", n_out), ("y_sd", n_out)]:
            self.register_buffer(name, torch.zeros(n))
        self.scaled = False

    def set_scaling(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.x_mu, self.x_sd = X.mean(0), X.std(0) + 1e-8
        self.y_mu, self.y_sd = Y.mean(0), Y.std(0) + 1e-8
        self.scaled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.x_mu) / self.x_sd) * self.y_sd + self.y_mu

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self(torch.as_tensor(X, dtype=torch.float32)).numpy()


def train(model: SABRNet, X: np.ndarray, Y: np.ndarray, epochs: int = 200,
          batch_size: int = 1024, lr: float = 1e-3, boost: float = 4.0,
          bump_patience: int = 15) -> list[float]:
    """Adam + MSE. Call once on Hagan data, then again on integration data to fine-tune (same weights).

    Adaptive learning rate (§4.1): ReduceLROnPlateau halves the rate when the epoch loss stops
    improving. If there is still no new best after `bump_patience` epochs, the cuts have not
    helped, so the rate is multiplied by `boost` (capped at the starting `lr`).
    """
    X, Y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)
    if not model.scaled:  # scaling is fixed by the first (Hagan) stage
        model.set_scaling(X, Y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    best, since_best = float("inf"), 0
    losses = []
    for _ in range(epochs):
        perm, total = torch.randperm(len(X)), 0.0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            # MSE in standardised units; raw vols give tiny gradients
            loss = nn.functional.mse_loss(model(X[idx]) / model.y_sd, Y[idx] / model.y_sd)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        losses.append(total / len(X))
        sched.step(losses[-1])

        # same improvement test as the scheduler (relative threshold 1e-4)
        if losses[-1] < best * (1 - sched.threshold):
            best, since_best = losses[-1], 0
        else:
            since_best += 1
        if since_best >= bump_patience:  # stuck despite the cuts: bump the rate back up
            for g in opt.param_groups:
                g["lr"] = min(g["lr"] * boost, lr)
            since_best, sched.num_bad_epochs = 0, 0
    return losses


def smile(strikes: np.ndarray, vols: np.ndarray) -> CubicSpline:
    """§4.3: spline for one smile, strikes (10,) and vols (10,) -> callable K -> vol."""
    return CubicSpline(strikes, vols)
