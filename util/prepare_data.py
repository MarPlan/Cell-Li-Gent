import h5py
import numpy as np
import torch


class BatteryData:
    def __init__(self, file_path, dataset, batch_size, seq_len, device):
        self.file = h5py.File(file_path, "r")
        self.file_path = file_path
        self.dataset = dataset

        self.batch_size = batch_size
        self.sub_seq_len = seq_len + 1
        self.device = device
        self.first = True

        self.n_series = self.file[dataset].shape[0]
        self.total_seq_len = self.file[dataset].shape[1]

    def get_batch(self, split):
        """Prepare a batch of data tensors for the given 'split'."""
        if split == "train":
            # We dont sample here because some data generation might have patterns that
            # repeat over a few consecutive time series
            # Only 80 percent of the training data rest is for validation
            seq_indices = np.random.randint(
                0, np.ceil(self.n_series * 0.8), self.batch_size
            )
            start_indices = np.random.randint(
                0, self.total_seq_len - self.sub_seq_len, self.batch_size
            )
        if split == "val":
            # Only 20 percent of training data for validation
            seq_indices = np.random.randint(
                np.ceil(self.n_series * 0.8) - 1, self.n_series, self.batch_size
            )
            start_indices = np.random.randint(
                0, self.total_seq_len - self.sub_seq_len, self.batch_size
            )
        if split == "pred":
            # Using validation data
            # pred_horizon = 6  # factor for seq_len prediction horizon
            pred_horizon = 4 * 2048 + 1
            seq_indices = np.random.randint(
                np.ceil(self.n_series * 0.8) - 1, self.n_series, 32
            )
            start_indices = np.random.randint(0, self.total_seq_len - pred_horizon, 32)
            if self.first:
                seq_indices[0:4] = np.array([2900, 2901, 2902, 2903])
                start_indices[0:4] = np.array([2_000] * 4)
                self.first = False

        with h5py.File(self.file_path, "r") as file:
            data = file[self.dataset]  # Access the specified dataset

            # Fetch the subsequences for `x` using list comprehension
            # Inputs [,,:3] >> I_terminal, U_terminal, T_surface
            # Use all parameters as inputs, at least for pre-training
            x = torch.stack(
                [
                    torch.from_numpy(
                        data[
                            seq_idx,
                            start_idx : start_idx
                            + (self.sub_seq_len if split != "pred" else pred_horizon),
                        ]
                    )
                    for seq_idx, start_idx in zip(seq_indices, start_indices)
                ]
            ).to(torch.float32)

            # Fetch the subsequences for `y` (targets) using list comprehension
            # Outputs [,,1:] >> U_terminal, T_surface, SoC, T_core, OCV
            # Not I_terminal because random, maybe interesting for operating prediction
            y = torch.stack(
                [
                    torch.from_numpy(
                        data[
                            seq_idx,
                            start_idx + 1 : start_idx
                            + 1
                            + (self.sub_seq_len if split != "pred" else pred_horizon),
                        ]
                    )
                    for seq_idx, start_idx in zip(seq_indices, start_indices)
                ]
            ).to(torch.float32)

        y = torch.cat([x[:, 1:, 0:1], y[:, :-1, 1:]], dim=2)
        x = torch.cat([x[:, 1:, 0:1], x[:, :-1, 1:]], dim=2)
        # If using CUDA, use pinned memory for asynchronous transfers to GPU
        if self.device == "cuda":
            x, y = (
                x.pin_memory().to(self.device, non_blocking=True),
                y.pin_memory().to(self.device, non_blocking=True),
            )
        else:
            x, y = (
                x.to(self.device),
                y.to(self.device),
            )  # Standard tensor transfer to the specified device

        return torch.round(x, decimals=4), torch.round(y, decimals=4)

