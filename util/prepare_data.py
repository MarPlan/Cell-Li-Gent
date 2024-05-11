# prepare data to a python binary pickle file for fast read speed
# dull dataset size determination: llama3: 8B params = 15T token (1 token = 1 [U,I,V] datapoint)
# dull dataset size determination: chinchilla: 8B params = 200B token (1 token = 1 [U,I,V] datapoint) (Meta homepage: https://ai.meta.com/blog/meta-llama-3/)

# My model: 500M params -> 12M token -> 12B datapoint of [U,I,V] ,respectively ~~ calc 120GB data (Chinchilla, considering as lower bound for sufficient results)
# My model: 500M params -> 1T token (based on llama) -> 1T datapoint of [U,I,V] ,respectively ~~ calc 12TB data even beyond chinchilla log-linear improvemtne if more tokens trained on!

# My model realistic: 100M params -> 2B Token -> 24GB data, but karpathys 9B~17GB binary data

import os

import h5py
import numpy as np
import torch


class BatteryData():
    def __init__(self, dir_name: str, mode, batch_size, seq_len, device):
        # TODO: when loading check for dataset signature and corresponding scaler if they match if not raise error
        self.file =h5py.File(os.path.join(dir_name, mode, ".hdf5"), "r")

        self.batch_size = batch_size
        self.sub_seq_len = seq_len
        self.device = device

        self.n_series =self.file["Dataset"].shape[0]
        self.total_seq_len = self.file["Dataset"].shape[1]

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
                0, self.total_seq_len - self.sub_seq_len + 1, self.batch_size
            )
        if split == "val":
            # Only 20 percent of training data for validation
            seq_indices = np.random.randint(
                0, np.floor(self.n_series * 0.8), self.batch_size
            )
            start_indices = np.random.randint(
                0, self.total_seq_len - self.sub_seq_len + 1, self.batch_size
            )
        if split in ["test", "predict"]:
            seq_indices = np.arange(self.batch_size)
            start_indices = np.zeros(self.batch_size)
        else:
            # Handle non-training/validation cases or return appropriate messaging/error
            print(f"No data handling defined for split: {split}")
            return None, None

        # with h5py.File(self.file, "r") as file:
        data = self.file  # Access the specified dataset

        # Fetch the subsequences for `x` using list comprehension
        x = torch.stack(
            [
                torch.from_numpy(
                    data[seq_idx, start_idx : start_idx + self.sub_seq_len, :]
                )
                for seq_idx, start_idx in zip(seq_indices, start_indices)
            ]
        ).to(torch.int64)

        # Fetch the subsequences for `y` (targets) using list comprehension
        y = torch.stack(
            [
                torch.from_numpy(
                    data[
                        seq_idx, start_idx + 1 : start_idx + 1 + self.sub_seq_len, :
                    ]
                )
                for seq_idx, start_idx in zip(seq_indices, start_indices)
            ]
        ).to(torch.int64)

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

        return x, y
