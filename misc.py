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
        ax[0].legend(loc="upper right")

    ax[1].plot(t, y[batch_nr, :, 2], label="Y_surf")
    ax[1].plot(t, y_hat[batch_nr, :, 2], "--", label="y_hat_surf")
    ax[1].plot(t, y_hat_pseudo[batch_nr, :, 2], ":", label="y_hat_pseudo_surf")

    ax[1].plot(t, y[batch_nr, :, 4], label="Y_core")
    ax[1].plot(t, y_hat[batch_nr, :, 4], "--", label="y_hat_core")
    ax[1].plot(t, y_hat_pseudo[batch_nr, :, 4], ":", label="y_hat_pseudo_core")
    ax[1].set_ylabel("Temperature [°C]")
    ax[1].legend(loc="upper right")

    ax[2].plot(t, y[batch_nr, :, 1], label="Y_term")
    ax[2].plot(t, y_hat[batch_nr, :, 1], "--", label="y_term")
    ax[2].plot(t, y_hat_pseudo[batch_nr, :, 1], ":", label="y_hat_pseudo_term")

    ax[2].plot(t, y[batch_nr, :, 5], label="Y_OCV")
    ax[2].plot(t, y_hat[batch_nr, :, 5], "--", label="y_hat_OCV")
    ax[2].plot(t, y_hat_pseudo[batch_nr, :, 5], ":", label="y_hat_pseudo_OCV")
    ax[2].set_ylabel("Voltage [V]")
    ax[2].legend(loc="upper right")

    for i in [0]:
        ax[3].plot(t, y[batch_nr, :, i], label="Y")
        ax[3].plot(t, y_hat[batch_nr, :, i], "--", label="y_hat")
        ax[3].plot(t, y_hat_pseudo[batch_nr, :, i], ":", label="y_hat_pseudo")
        ax[3].set_ylabel("Current [A]")
        ax[3].set_xlabel("Time [h]")
        ax[3].legend(loc="upper right")

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




def plot_error():
    from matplotlib.cm import get_cmap
    from scipy.ndimage import uniform_filter1d
    ckpt_path = "ckpt/transformer/final/8.6e-06_pred_loss.npy"
    data = np.load(ckpt_path, allow_pickle=False)
    x, y, y_hat, y_hat_pseudo = data[0], data[1], data[2], data[3]
    t = np.arange(0, x.shape[-2]) / 60 / 60

    # Compute the errors
    error = np.abs(y - y_hat)
    error_pseudo = np.abs(y - y_hat_pseudo)

    # Overall mean error
    error_mean = error.mean(axis=(0, 2))
    error_pseudo_mean = error_pseudo.mean(axis=(0, 2))

    # Individual mean errors across the last dimension
    error_individual = error.mean(axis=0)  # Shape: [seq_len, features]
    error_pseudo_individual = error_pseudo.mean(axis=0)  # Shape: [seq_len, features]

    # Smoothing function
    def smooth(data, window_size=500):
        return uniform_filter1d(data, size=window_size)

    fig_size_big = (10, 12)
    fig, ax = plt.subplots(2, 1, figsize=fig_size_big, sharex=True)

    # Get a colormap
    cmap = get_cmap("tab10")


    labels=["Current [A]","Volt_term [V]","Temp_surf [°C]","SoC [·]","Temp_core [°C]","OVC [V]"]
    # Plotting the error subplot
    for i in range(error_individual.shape[1]):
        color = cmap(i % 10)  # Get a color for each feature
        label = labels[i]
        # Plot the raw data with lower opacity
        ax[0].plot(t, error_individual[:, i], color=color, alpha=0.2)
        # Plot the smoothed data
        ax[0].plot(t, smooth(error_individual[:, i]), color=color, label=label)

    # Plot overall mean error
    ax[0].plot(t, error_mean, color='red', alpha=0.2)
    ax[0].plot(t, smooth(error_mean), color='red', label='Overall Mean Error')
    ax[0].set_ylabel('Mean squared error y_hat')
    ax[0].legend()

    # Plotting the error_pseudo subplot
    for i in range(error_pseudo_individual.shape[1]):
        color = cmap(i % 10)  # Get a color for each feature
        label = labels[i]
        # Plot the raw data with lower opacity
        ax[1].plot(t, error_pseudo_individual[:, i], color=color, alpha=0.2)
        # Plot the smoothed data
        ax[1].plot(t, smooth(error_pseudo_individual[:, i]), color=color, label=label)

    # Plot overall mean error pseudo
    ax[1].plot(t, error_pseudo_mean, color='red', alpha=0.2)
    ax[1].plot(t, smooth(error_pseudo_mean), color='red', label='Overall Mean Error Pseudo')
    ax[1].set_xlabel('Time [h]')
    ax[1].set_ylabel('Mean squared error y_hat_pseudo')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(
        ckpt_path.replace(".npy", "_error_comparison_smoothed.png"),
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":

    plot_error()
    # plot_outputs_readme()
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
