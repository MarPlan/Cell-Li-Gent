import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

learning_rate = 0.001  # initial learning rate
min_lr = 1e-6
warmup_iters = 200
interval_iters = 1500
norm_history = deque([0]*interval_iters, maxlen=interval_iters)

# Global variables for tracking the state of change
new_lr = learning_rate
current_lr = learning_rate
change_iters = 0  # Counter for how many iterations we've been changing the lr
steps = 0


def get_lr(it, norm):
    global current_lr, new_lr, change_iters, steps, lr_step
    norm_history.append(norm)

    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        current_lr = learning_rate * it / warmup_iters
        steps = interval_iters
        return current_lr

    # 2) Maintain learning rate for interval_iters
    if steps < 1:
        steps = interval_iters
        averaged_norm = sum(norm_history) / len(norm_history)
        exponent = math.ceil(math.log10(1 / averaged_norm))

        if exponent > 0:
            new_lr = learning_rate / (10**exponent)
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
    # Example usage
    lrs = []
    norm = np.linspace(0, 1, 100000)
    for it in range(0, 100000):
        lr = get_lr(it, norm[it])
        lrs.append(lr)
        # print(f"Iteration {it}: Learning Rate = {lr}")

    plt.plot(np.array(lrs))
    # plt.plot(np.array(avgs))
    plt.show()
    t = 9

### Explanation:
"""
# 1. **Warmup Phase:**
#    - During the first warmup_iters iterations, the learning rate increases linearly from 0 to the initial learning_rate.
#
# 2. **Steady Phase and Interval Check:**
#    - After warming up, the learning rate remains constant for interval_iters - 1.
#    - On every `interval_iters` step, the average norm is computed and the new learning rate is calculated based on the exponent of the average norm.
#
# 3. **Linear Change:**
#    - If a change is queued (when `change_iters` is set), the learning rate linearly transitions from the current value to the new value over `warmup_iters` iterations.
#    - `change_iters` decrements with each iteration until it reaches 0, completing the transition.
#
# 4. **Usage:**
#    - The `get_lr` function adjusts the learning rate based on the specified scheduling strategy.
#    - Replace `norm` with the actual norm values observed during your training loop.
#
# This code ensures that the learning rate undergoes a controlled adjustment phase, providing a smooth transition between intervals, and remains constant otherwise.
"""
