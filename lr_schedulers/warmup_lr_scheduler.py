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

from torch.optim import Optimizer

from .lr_scheduler import LearningRateScheduler


class WarmupLRScheduler(LearningRateScheduler):
    """
    Warmup learning rate until `total_steps`

    Args:
        optimizer (Optimizer): Wrapped optimizer
        peak_lr (float): Maximum learning rate
        init_lr (float): Initial learning rate
        warmup_steps (int): Warmup the learning rate linearly for the first N updates
        total_steps (int): Total training steps
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *args,
        peak_lr: float,
        init_lr: float,
        warmup_steps: int,
        total_steps: int,
        **kwargs,
    ) -> None:
        super().__init__(optimizer, *args, **kwargs)
        if warmup_steps != 0:
            warmup_rate = peak_lr - init_lr
            self.warmup_rate = warmup_rate / warmup_steps
        else:
            self.warmup_rate = 0
        self.update_steps = 1
        self.init_lr = self.lr = init_lr
        self.warmup_steps = warmup_steps

    def step(self, epoch: int | None = None):
        if self.update_steps < self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * self.update_steps
            self.set_lr(lr)
            self.lr = lr
        self.update_steps += 1
        return self.lr
