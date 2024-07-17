import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

# class LRScheduler:
#     def __init__(
#         self, initial_lr, warmup_lr, warmup_iters, max_iters, min_lr, decay_iters
#     ):
#         """
#         Initialize the learning rate scheduler.
#
#         Args:
#             initial_lr (float): The initial learning rate for the first warm-up phase.
#             warmup_lr (float): The target learning rate for subsequent warm-up phases.
#             warmup_iters (int): The number of iterations to warm up.
#             max_iters (int): The number of iterations for one warmup-decay cycle.
#             min_lr (float): The minimum learning rate.
#             decay_iters (int): The number of iterations over which to decay the learning rate.
#         """
#         self.initial_lr = initial_lr
#         self.warmup_lr = warmup_lr
#         self.warmup_iters = warmup_iters
#         self.max_iters = max_iters
#         self.min_lr = min_lr
#         self.decay_iters = decay_iters
#         self.cycle_iterations = max_iters
#         self.lr_step = 0
#         self.current_cycle = 0
#
#     def get_lr(self, current_iter, lr_prev):
#         """
#         Compute the learning rate at the given iteration.
#
#         Args:
#             current_iter (int): The current iteration number.
#             lr_prev (float): The learning rate from the previous iteration.
#
#         Returns:
#             float: The computed learning rate.
#         """
#         # Total iterations passed in all cycles
#         total_iter = current_iter + (self.current_cycle * self.cycle_iterations)
#
#         # Find the effective iteration within the current cycle
#         effective_iter = total_iter % self.cycle_iterations
#
#         # Determine the correct target learning rate during warmup
#         target_lr = self.initial_lr if self.current_cycle == 0 else self.warmup_lr
#
#         # Phase 1: Warmup phase
#         if effective_iter < self.warmup_iters:
#             current_lr = target_lr * (effective_iter / self.warmup_iters)
#         # Phase 2: Decay phase
#         else:
#             decay_phase_iter = effective_iter - self.warmup_iters
#             total_decay_phase_iters = self.decay_iters - self.warmup_iters
#             if decay_phase_iter < total_decay_phase_iters:
#                 decay_step = (target_lr - self.min_lr) / total_decay_phase_iters
#                 current_lr = target_lr - decay_step * decay_phase_iter
#             else:
#                 current_lr = self.min_lr
#
#         # Ensure learning rate does not drop below the minimum learning rate
#         current_lr = max(current_lr, self.min_lr)
#
#         # Check if this completes a cycle
#         if effective_iter + 1 == self.cycle_iterations:
#             self.current_cycle += 1
#
#         return current_lr


def estimate_loss_print(file_path, dataset_name, splits=[]):
    seq_len = 4096
    for split in splits:
        X, Y = train_data.get_batch(split)
        if split == "pred":
            y_hat = []
            input = X[:, :seq_len]
            print(input[:, -1] - X[:, seq_len-1])

            for i in range(2*8192 - seq_len):
                input[:, -1, 0] = X[:, seq_len + i, 0]
                y_hat.append(Y[:,  i].unsqueeze(1))
            y_hat = torch.concatenate(y_hat, dim=1).to(Y.device)
            losses = F.mse_loss(Y[:, -4095:-1, :], y_hat[:, -4096:, :])
            plt.plot(Y[0, seq_len-1:-1, 1])
            plt.plot(y_hat[0,:,1])
            plt.grid()
            plt.show()
            t = 5
            # fig.savefig(out_path, dpi=300)


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

        # Phase 1: Warmup phase
        if effective_iter < self.warmup_iters:
            warmup_factor = (self.warmup_lr / self.min_lr) ** (
                effective_iter / self.warmup_iters
            )
            current_lr = self.min_lr * warmup_factor
        # Phase 2: Decay phase
        else:
            decay_phase_iter = effective_iter - self.warmup_iters
            total_decay_phase_iters = self.decay_iters - self.warmup_iters
            if decay_phase_iter < total_decay_phase_iters:
                decay_factor = (self.min_lr / self.warmup_lr) ** (
                    decay_phase_iter / total_decay_phase_iters
                )
                current_lr = self.warmup_lr * decay_factor
            else:
                current_lr = self.min_lr

        # Ensure learning rate does not drop below the minimum learning rate
        current_lr = max(current_lr, self.min_lr)

        # Check if this completes a cycle
        if effective_iter + 1 == self.cycle_iterations:
            self.current_cycle += 1

        return current_lr


def plot_outputs_readme():

    ckpt_path = "ckpt/transformer/final/8.6e-06_pred_loss.npy"
    data = np.load(ckpt_path, allow_pickle=False)
    x, y, y_hat, y_hat_pseudo = data[0], data[1], data[2], data[3]
    t = np.arange(0, x.shape[-2])/60/60

    fig = plt.figure()
    fig_size_big = (10, 12)
    # fig_size_big = (5.5, 4)
    fig.set_size_inches(fig_size_big)
    ax = fig.subplots(4, 1, sharex=True)

    batch_nr = 0
    for i in [3]:
        ax[0].plot(t,y[batch_nr, :, i], label="Y")
        ax[0].plot(t, y_hat[batch_nr, :, i], "--", label="y_hat")
        ax[0].plot(t, y_hat_pseudo[batch_nr, :, i], ":", label="y_hat_pseudo")
        ax[0].set_ylabel("SoC [·]")
        ax[0].legend(loc="lower right")

    for i in [2, 4]:
        ax[1].plot(t, y[batch_nr, :, i], label="Y")
        ax[1].plot(t, y_hat[batch_nr, :, i], "--", label="y_hat")
        ax[1].plot(t, y_hat_pseudo[batch_nr, :, i], ":", label="y_hat_pseudo")
        ax[1].set_ylabel("Temperature [°C]")

    for i in [1, 5]:
        ax[2].plot(t, y[batch_nr, :, i], label="Y")
        ax[2].plot(t, y_hat[batch_nr, :, i], "--", label="y_hat")
        ax[2].plot(t, y_hat_pseudo[batch_nr, :, i], ":", label="y_hat_pseudo")
        ax[2].set_ylabel("Voltage [V]")

    for i in [0]:
        ax[3].plot(t, y[batch_nr, :, i], label="Y")
        ax[3].plot(t, y_hat[batch_nr, :, i], "--", label="y_hat")
        ax[3].plot(t, y_hat_pseudo[batch_nr, :, i], ":", label="y_hat_pseudo")
        ax[3].set_ylabel("Current [A]")
        ax[3].set_xlabel("Time [h]")

    plt.tight_layout()
    plt.savefig(
        ckpt_path.replace(".npy", "_readme.png"),
        format="png",
        bbox_inches="tight",
        # pad_inches=[0, 0, 1, 0]
        # pad_inches="tight"
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":

    plot_outputs_readme()
    t=5
    # import os
    #
    # import torch
    #
    # from util.prepare_data import BatteryData
    # batch_size = 4
    # seq_len = 400
    # device = "cpu"
    # dataset = "spme_training_scaled"
    # data_file = os.path.abspath("data/train/battery_data.h5")
    # train_data = BatteryData(data_file, dataset, batch_size, seq_len, device)
    # estimate_loss_print(data_file, dataset, splits=["pred"])
    # import os
    #
    # from util.prepare_data import BatteryData
    # batch_size = 4
    # seq_len = 400
    # device = "cpu"
    # dataset = "spme_training_scaled"
    # data_file = os.path.abspath("data/train/battery_data.h5")
    # train_data = BatteryData(data_file, dataset, batch_size, seq_len, device)
    # split = "train"
    # X, Y = train_data.get_batch(split)
    # plt.plot(X[0,:,1])
    # plt.plot(Y[0,:,1])
    # plt.show()
    # t=9

    data = np.load(
        "/Users/markus/Projects/Cell-Li-Gent/ckpt/transformer/v_best/1.4e-04_pred_loss.npy"
    )
    x, y, y_hat, y_hat_pseudo = data[0, :, :-1], data[1, :, :-1], data[2, :, 1:], data[3, :, 1:]
    fig = plt.figure()
    fig_size_big = (15, 15)
    fig.set_size_inches(fig_size_big)
    ax = fig.subplots(5, 1, sharex=True)

    batch_nr = 0
    for i in range(x.shape[-1]):
        ax[0].plot(x[batch_nr, :, i], label="X")

    for i in [1, 5]:
        ax[1].plot(y[batch_nr, :, i], label="Y")
        ax[1].plot(y_hat[batch_nr, :, i], "--", label="y_hat")
        ax[1].plot(y_hat_pseudo[batch_nr, :, i], ":", label="y_hat_pseudo")

    for i in [2, 4]:
        ax[2].plot(y[batch_nr, :, i], label="Y")
        ax[2].plot(y_hat[batch_nr, :, i], "--", label="y_hat")
        ax[2].plot(y_hat_pseudo[batch_nr, :, i], ":", label="y_hat_pseudo")

    for i in [3]:
        ax[3].plot(y[batch_nr, :, i], label="Y")
        ax[3].plot(y_hat[batch_nr, :, i], "--", label="y_hat")
        ax[3].plot(y_hat_pseudo[batch_nr, :, i], ":", label="y_hat_pseudo")

    for i in [0]:
        ax[4].plot(y[batch_nr, :, i], label="Y")
        ax[4].plot(y_hat[batch_nr, :, i], "--", label="y_hat")
        ax[4].plot(y_hat_pseudo[batch_nr, :, i], ":", label="y_hat_pseudo")


    plt.tight_layout()
    plt.savefig(
        "/Users/markus/Projects/Cell-Li-Gent/ckpt/transformer/v_best/local_1.4e-04_pred_loss.pdf",
        format="pdf",
        bbox_inches="tight",
        # pad_inches=[0, 0, 1, 0]
        # pad_inches="tight"
        dpi=300,
    )

    # fig.savefig(
    #     "/Users/markus/Projects/Cell-Li-Gent/ckpt/transformer/v_best/local_1.4e-04_pred_loss.png",
    #     dpi=300,
    # )
    plt.show()
    t = 5
    # plt.show()

    import os

    import h5py
    import torch
    import torch.nn.functional as F

    dataset_name = "spme_training_scaled"
    file_path = os.path.abspath("data/train/battery_data.h5")
    with h5py.File(file_path, "r") as file:
        data_scaled = file[dataset_name]
        mins, maxs = (
            data_scaled.attrs["min_values"],
            data_scaled.attrs["max_values"],
        )
    maxs_expanded = torch.tensor(maxs)
    mins_expanded = torch.tensor(mins)
    scaling = F.mse_loss(maxs_expanded, mins_expanded, reduction="none")
    scaling_ = torch.abs(maxs_expanded - mins_expanded)
    t = 5
    # import os
    #
    # import h5py
    # import torch
    #
    # from util.prepare_data import BatteryData
    # dataset = "spme_training_scaled"
    # data_file = os.path.abspath("data/train/battery_data.h5")
    # estimate_loss(data_file, dataset)

    # scheduler = LRScheduler(
    #     initial_lr=3e-3,
    #     warmup_lr=3e-3,
    #     warmup_iters=200,
    #     max_iters=2000,
    #     min_lr=1e-9,
    #     decay_iters=2000,
    # )
    # lrs = []
    # lr = 3e-3
    # for it in range(100_000):
    #     lr = scheduler.get_lr(it, lr)
    #     lrs.append(lr)
    # plt.plot(np.array(lrs))
    # plt.show()
    # t = 9

    # Sample usage and plotting
    scheduler = LRScheduler(
        initial_lr=3e-3,
        warmup_lr=3e-3,
        warmup_iters=200,
        max_iters=2000,
        min_lr=1e-7,
        decay_iters=2000,
    )

    lrs = []
    lr = 3e-3
    for it in range(10_000):
        lr = scheduler.get_lr(it, lr)
        lrs.append(lr)

    # Plot on linear scale
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(lrs)), lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate on Linear Scale")

    # Plot on log scale
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(lrs)), lrs)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate on Logarithmic Scale")

    plt.tight_layout()
    plt.show()
    t = 4
