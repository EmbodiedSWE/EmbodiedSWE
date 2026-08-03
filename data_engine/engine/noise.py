"""Action-noise wrapper — L4 dynamics randomization, transparent to any solve(env).

The EXECUTED action = commanded + noise; the commanded (clean) action stays the
training label (DART-style). The solve stays closed-loop on the true state — its
own feedback control absorbs the perturbations, and that recovery is the data.

One mechanism: burst-gated Gaussian. Per env, a noise window starts with prob
`prob` per step and lasts `duration` steps; inside a window every step gets
N(0, sigma) on `dims`. The corner cases are the classic models:

    sigma=0                  off — the nominal/gate default
    prob=1, duration=1       continuous white noise
    prob=0.01, duration=30   occasional multi-step disturbances to recover from

Rules: only `dims` are noised (gripper dims never — noise there corrupts pinch
calibration); a row whose dims are all zero is untouched (the batch_solve hold
convention for finished envs, so completed goals are never wiggled); seeded, so
a batch's noise replays from its recorded seed; executed dims clamp to +-1.
"""

from __future__ import annotations

import torch


class NoisyActionEnv:
    """Every attribute delegates to the wrapped env; only step() perturbs."""

    def __init__(self, env, dims: slice, sigma: float = 0.0,
                 prob: float = 1.0, duration: int = 1, seed: int = 0) -> None:
        self._env, self._dims, self._sigma = env, dims, float(sigma)
        self._prob, self._duration = float(prob), int(duration)
        self._rng = torch.Generator().manual_seed(seed)
        self._left = None  # (E,) steps remaining in each env's noise window
        self.last_clean = self.last_executed = None

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def step(self, action, render: bool = False):
        if self._sigma == 0.0:
            self.last_clean = self.last_executed = action
            return self._env.step(action, render)

        clean = action.detach().clone()
        a = clean[:, self._dims].cpu()
        active = a.abs().sum(dim=1) > 0  # zero rows = held envs: never noised
        if self._left is None:
            self._left = torch.zeros(len(a), dtype=torch.long)
        start = (self._left == 0) & active & \
                (torch.rand(len(a), generator=self._rng) < self._prob)
        self._left[start] = self._duration
        noisy = (self._left > 0) & active
        self._left -= (self._left > 0).long()

        noise = torch.randn(*a.shape, generator=self._rng) * self._sigma * noisy.unsqueeze(1)
        executed = clean.clone()
        executed[:, self._dims] = (a + noise).clamp(-1, 1).to(clean.device, clean.dtype)
        self.last_clean, self.last_executed = clean, executed
        return self._env.step(executed, render)
