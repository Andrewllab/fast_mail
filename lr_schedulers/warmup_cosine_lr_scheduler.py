from __future__ import annotations

import math

from torch.optim import Optimizer

from .lr_scheduler import LearningRateScheduler


class WarmupCosineLRScheduler(LearningRateScheduler):
    """
    Linear warmup learning rate until `warmup_steps`, then cosine decay until `total_steps`.

    This scheduler is intended to be stepped *per optimizer update* (i.e., per training step).

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        peak_lr (float): Maximum learning rate reached at the end of warmup.
        init_lr (float): Initial learning rate at step 0.
        warmup_steps (int): Number of warmup steps (linear ramp).
        total_steps (int): Total number of training steps (warmup + decay).
        final_lr (float | None): Optional absolute final learning rate at `total_steps`.
            If provided, cosine decays from `peak_lr` to `final_lr`.
        min_lr_ratio (float): Used only if `final_lr` is None. Then final_lr = peak_lr * min_lr_ratio.
            Must be in [0, 1]. Default: 0.1

    Notes:
        - If warmup_steps == 0, warmup is skipped and schedule starts at peak_lr immediately.
        - If total_steps <= warmup_steps, the schedule becomes pure warmup (or constant).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *args,
        peak_lr: float,
        init_lr: float,
        warmup_ratio: float,
        total_steps: int,
        final_lr: float | None = None,
        min_lr_ratio: float = 0.1,
        **kwargs,
    ) -> None:

        if warmup_ratio < 0:
            raise ValueError(f"warmup_ratio must be >= 0, got {warmup_ratio}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        if warmup_ratio > 1.0:
            # We allow it, but it's almost certainly a config mistake.
            # The behavior will just be warmup (or constant) without decay.
            pass
        if final_lr is None:
            if not (0.0 <= min_lr_ratio <= 1.0):
                raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
            final_lr = peak_lr * float(min_lr_ratio)
        if peak_lr < 0 or init_lr < 0 or final_lr < 0:
            raise ValueError("Learning rates must be non-negative.")

        self.peak_lr = float(peak_lr)
        self.init_lr = float(init_lr)
        self.final_lr = float(final_lr)

        self.warmup_steps = int(warmup_ratio * total_steps)
        self.total_steps = int(total_steps)

        # Linear warmup increment (per step). If warmup_steps==0 -> unused.
        if self.warmup_steps > 0:
            self.warmup_rate = (self.peak_lr - self.init_lr) / float(self.warmup_steps)
        else:
            self.warmup_rate = 0.0

        # Following the example: start counting at 1 (first call to step uses update_steps=1).
        self.update_steps = 1

        super().__init__(optimizer, *args, **kwargs)

        # Set initial LR immediately.
        self.lr = self.init_lr
        self.set_lr(self.lr)

    def _cosine_lr(self, step: int) -> float:
        """
        Cosine decay from peak_lr to final_lr over [warmup_steps, total_steps].

        step: current update step (1-indexed to match this scheduler's semantics).
        """
        # If no decay region, just return peak_lr (or whatever makes sense).
        if self.total_steps <= self.warmup_steps:
            return self.peak_lr

        # Map step into [0, 1] progress for the decay phase.
        decay_steps = self.total_steps - self.warmup_steps
        # Convert to a 0-based index within decay.
        t = step - self.warmup_steps
        # Clamp to [0, decay_steps]
        t = max(0, min(t, decay_steps))
        progress = t / float(max(1, decay_steps))

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_lr + (self.peak_lr - self.final_lr) * cosine

    def step(self, epoch: int | None = None):
        # Warmup: linearly ramp from init_lr toward peak_lr.
        if self.warmup_steps > 0 and self.update_steps <= self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * float(self.update_steps)
            # Numerical safety: do not overshoot peak_lr due to rounding.
            lr = (
                min(lr, self.peak_lr)
                if self.peak_lr >= self.init_lr
                else max(lr, self.peak_lr)
            )
        else:
            # Cosine decay (or constant if decay phase doesn't exist).
            lr = self._cosine_lr(self.update_steps)

        self.set_lr(lr)
        self.lr = lr
        self.update_steps += 1
        return self.lr
