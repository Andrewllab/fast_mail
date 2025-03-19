# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from typing import Sequence

from torch.optim import Optimizer

from .lr_scheduler import LearningRateScheduler


class TriStageLRScheduler(LearningRateScheduler):
    r"""
    Tri-Stage Learning Rate Scheduler. Implement the learning rate scheduler in "SpecAugment"

    Similar to inverse_squre_root scheduler, but tri_stage learning rate employs
    three stages LR scheduling:

        - warmup stage, starting from `lr` * `init_lr_scale`, linearly
          increased to `lr` in `warmup_steps` iterations
        - hold stage, after `warmup_steps`, keep the LR as `lr` for `hold_steps`
          iterations
        - decay stage, after hold stage, decay LR exponetially to
          `lr` * `final_lr_scale` in `decay_steps`;
          after that LR is keep as `final_lr_scale` * `lr`

    During warmup::
      init_lr = cfg.init_lr_scale * cfg.lr
      lrs = torch.linspace(init_lr, cfg.lr, cfg.warmup_steps)
      lr = lrs[update_num]

    During hold::
      lr = cfg.lr

    During decay::
      decay_factor = - math.log(cfg.final_lr_scale) / cfg.decay_steps
      lr = cfg.lr * exp(- (update_num - warmup_steps - decay_steps) * decay_factor)

    Updated it to be CosineAnneaLLR Scheduler

    After that::
      lr = cfg.lr * cfg.final_lr_scale

    Args:
        optimizer (Optimizer): Wrapped optimizer
        init_lr (float): Initial learning rate
        init_lr_scale (float): Initial learning rate scale
        final_lr_scale (float): Final learning rate scale
        phase_ratio (Sequence[float]): Automatically sets warmup/hold/decay steps to the ratio
            specified here from max_updates. the ratios must add up to 1.0
        total_steps (int): Total training steps
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *args,
        lr: float,
        init_lr_scale: float,
        final_lr_scale: float,
        phase_ratio: Sequence[float],
        total_steps: int,
        **kwargs,
    ):
        super().__init__(optimizer, *args, **kwargs)

        phase_ratio = tuple(phase_ratio)
        self.warmup_steps = int(total_steps * phase_ratio[0])
        self.hold_steps = int(total_steps * phase_ratio[1])
        self.decay_steps = int(total_steps * phase_ratio[2])

        self.peak_lr = lr
        self.init_lr = init_lr_scale * lr
        self.final_lr = final_lr_scale * lr

        self.warmup_rate = (
            (self.peak_lr - self.init_lr) / self.warmup_steps
            if self.warmup_steps != 0
            else 0
        )
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps
        self.update_step = 0
        self.lr = self.init_lr

    def _decide_stage(self):
        if self.update_step < self.warmup_steps:
            return 0, self.update_step

        offset = self.warmup_steps

        if self.update_step < offset + self.hold_steps:
            return 1, self.update_step - offset

        offset += self.hold_steps

        if self.update_step <= offset + self.decay_steps:
            # decay stage
            return 2, self.update_step - offset

        offset += self.decay_steps

        return 3, self.update_step - offset

    def step(self, epoch: int | None = None):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            # self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
            self.lr = self.final_lr + 0.5 * (self.peak_lr - self.final_lr) * (
                1 + math.cos(steps_in_stage / self.decay_steps * math.pi)
            )
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.lr)
        self.update_step += 1

        return self.lr
