import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np


class LRScheduler:
    def __init__(
        self, initial_lr, warmup_lr, warmup_iters, max_iters, min_lr, decay_iters
    ):
        """
        Initialize the learning rate scheduler.

        Args:
            initial_lr (float): The initial learning rate for the first warm-up phase.
            warmup_lr (float): The target learning rate for subsequent warm-up phases.
            warmup_iters (int): The number of iterations to warm up.
            max_iters (int): The number of iterations for one warmup-decay cycle.
            min_lr (float): The minimum learning rate.
            decay_iters (int): The number of iterations over which to decay the learning rate.
        """
        self.initial_lr = initial_lr
        self.warmup_lr = warmup_lr
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.decay_iters = decay_iters
        self.cycle_iterations = max_iters
        self.lr_step = 0
        self.current_cycle = 0

    def get_lr(self, current_iter, lr_prev):
        """
        Compute the learning rate at the given iteration.

        Args:
            current_iter (int): The current iteration number.
            lr_prev (float): The learning rate from the previous iteration.

        Returns:
            float: The computed learning rate.
        """
        # Total iterations passed in all cycles
        total_iter = current_iter + (self.current_cycle * self.cycle_iterations)

        # Find the effective iteration within the current cycle
        effective_iter = total_iter % self.cycle_iterations

        # Determine the correct target learning rate during warmup
        target_lr = self.initial_lr if self.current_cycle == 0 else self.warmup_lr

        # Phase 1: Warmup phase
        if effective_iter < self.warmup_iters:
            current_lr = target_lr * (effective_iter / self.warmup_iters)
        # Phase 2: Decay phase
        else:
            decay_phase_iter = effective_iter - self.warmup_iters
            total_decay_phase_iters = self.decay_iters - self.warmup_iters
            if decay_phase_iter < total_decay_phase_iters:
                decay_step = (target_lr - self.min_lr) / total_decay_phase_iters
                current_lr = target_lr - decay_step * decay_phase_iter
            else:
                current_lr = self.min_lr

        # Ensure learning rate does not drop below the minimum learning rate
        current_lr = max(current_lr, self.min_lr)

        # Check if this completes a cycle
        if effective_iter + 1 == self.cycle_iterations:
            self.current_cycle += 1

        return current_lr


if __name__ == "__main__":
    scheduler = LRScheduler(
        initial_lr=3e-3,
        warmup_lr=3e-4,
        warmup_iters=200,
        max_iters=5000,
        min_lr=1e-7,
        decay_iters=2000,
    )
    lrs = []
    lr = 3e-3
    for it in range(100_000):
        lr = scheduler.get_lr(it, lr)
        lrs.append(lr)
    plt.plot(np.array(lrs))
    plt.show()
    t = 9
