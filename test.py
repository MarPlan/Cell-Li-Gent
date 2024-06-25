import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

learning_rate = 0.01  # initial learning rate
min_lr = 1e-4
warmup_iters = 200
interval_iters = 500
norm_history = deque([0] * interval_iters, maxlen=interval_iters)

# Global variables for tracking the state of change
new_lr = learning_rate
current_lr = learning_rate
change_iters = 0  # Counter for how many iterations we've been changing the lr
steps = 0
check_change = 0
averaged_norm = 1
exp_2 = 0
max_iters = 5000


# def get_lr(it, lr_prev):
#     global current_lr, decay, lr_step
#     if it < warmup_iters:
#         decay = True
#         current_lr = learning_rate * it / warmup_iters
#         return current_lr
#     if it > (max_iters - warmup_iters):
#         if decay:
#             lr_step = lr_prev / warmup_iters
#             decay = False
#         current_lr -= lr_step
#         return current_lr
#     if current_lr > min_lr:
#         lr_step = (learning_rate - min_lr) / (2_060 - warmup_iters)
#         current_lr -= lr_step
#         return current_lr
#     else:
#         return current_lr

decay = True
lr_step = 0


def get_lr(it, lr_prev):
    global decay, lr_step
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > (max_iters - warmup_iters):
        if decay:
            lr_step = lr_prev / warmup_iters
            decay = False
        lr_prev -= lr_step
        return lr_prev
    if lr_prev > min_lr:
        lr_step = (learning_rate - min_lr) / (2_060 - warmup_iters)
        lr_prev -= lr_step
        return lr_prev
    else:
        return lr_prev


class LRScheduler:
    def __init__(self, learning_rate, warmup_iters, max_iters, min_lr):
        self.learning_rate = learning_rate
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.decay = True
        self.lr_step = 0

    def get_lr(self, it, lr_prev):
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        if it > (self.max_iters - self.warmup_iters):
            if self.decay:
                self.lr_step = lr_prev / self.warmup_iters
                self.decay = False
            lr_prev -= self.lr_step
            return lr_prev
        if lr_prev > self.min_lr:
            self.lr_step = (self.learning_rate - self.min_lr) / (
                2_060 - self.warmup_iters
            )
            lr_prev -= self.lr_step
            return lr_prev
        else:
            return lr_prev


def get_lr_trapz(it, norm):
    global \
        current_lr, \
        new_lr, \
        change_iters, \
        steps, \
        lr_step, \
        averaged_norm, \
        check_change, \
        exp_2
    norm_history.append(norm)

    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        current_lr = learning_rate * it / warmup_iters
        steps = interval_iters
        return current_lr

    # 2) Maintain learning rate for interval_iters
    if steps < 1:
        steps = interval_iters
        if math.ceil(math.log10(1 / averaged_norm)) == math.ceil(
            math.log10(1 / (sum(norm_history) / len(norm_history)))
        ):
            check_change += 1
        else:
            check_change = 0
        averaged_norm = sum(norm_history) / len(norm_history)
        exponent = math.ceil(math.log10(1 / averaged_norm))

        if exponent > 0:
            if check_change == 4:
                exp_2 += 1 if round(current_lr, 15) > min_lr else -1
                check_change = 0
            new_lr = learning_rate / (10 ** (exponent + exp_2))
            new_lr = new_lr if new_lr > min_lr else min_lr
            lr_step = (new_lr - current_lr) / warmup_iters
            change_iters = warmup_iters  # We will transition over warmup_iters

    # If a change is queued, adjust linearly over warmup_iters iterations
    steps -= 1
    if change_iters > 0:
        steps = interval_iters
        change_iters -= 1
        current_lr += lr_step

    return current_lr


if __name__ == "__main__":
    while True:
        try:
            t=3
            break
        except RuntimeError:
            continue
    scheduler = LRScheduler(learning_rate, warmup_iters, max_iters, min_lr)
    lrs = []
    lr = learning_rate
    for it in range(0, 10000):
        if it > max_iters:
            break
        lr = scheduler.get_lr(it, lr)
        # lr = get_lr(it, lr)
        lrs.append(lr)

    plt.plot(np.array(lrs))
    plt.show()
    t = 9
