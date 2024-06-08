import os
from contextlib import nullcontext

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from model.transformer import ModelArgs, Transformer
from util.prepare_data import BatteryData


def check_in_out(x, y, y_hat, y_hat_pseudo, file_path=None, dataset_name=None):
    with h5py.File(file_path, "r") as file:
        data_scaled = file[dataset_name]
        mins, maxs = data_scaled.attrs["min_values"], data_scaled.attrs["max_values"]
    fig = plt.figure()
    ax = fig.subplots(2, 1, sharex=True)

    batch_nr = 0
    for i in range(x.shape[-1]):
        x[:, :, i] = x[:, :, i] * (maxs[i] - mins[i]) + mins[i]
        ax[0].plot(x[batch_nr, :, i], label="X")
    ax[0].legend()

    for i in range(y.shape[-1]):
        y[:, :, i] = y[:, :, i] * (maxs[i + 1] - mins[i + 1]) + mins[i + 1]
        y_hat[:, :, i] = y_hat[:, :, i] * (maxs[i + 1] - mins[i + 1]) + mins[i + 1]
        y_hat_pseudo[:, :, i] = (
            y_hat_pseudo[:, :, i] * (maxs[i + 1] - mins[i + 1]) + mins[i + 1]
        )

        ax[1].plot(y[batch_nr, :, i], label="Y")
        ax[1].plot(y_hat[batch_nr, :, i], "--", label="y_hat")
        ax[1].plot(y_hat_pseudo[batch_nr, :, i], ":", label="y_hat_pseudo")
    ax[1].legend()
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def estimate_loss():
    model.eval()
    X, Y = train_data.get_batch(split="pred")
    y_hat = []
    y_hat_pseudo = []
    with ctx:
        input = X[:, :seq_len]
        for i in range(seq_len * 11):
            y, _ = model(input)
            y_hat.append(y.cpu().numpy())
            input = torch.roll(input, -1, 1)
            input[:, -1, 0] = X[:, seq_len + i, 0]
            input[:, -1, 1:3] = y[:, -1, :2]

            y, _ = model(X[:, i : seq_len + i])
            y_hat_pseudo.append(y.cpu().numpy())

        y_hat = np.concatenate(y_hat, axis=1)
        y_hat_pseudo = np.concatenate(y_hat_pseudo, axis=1)
        loss = np.mean(np.sum(Y[:, seq_len:].cpu().numpy() - y_hat) ** 2)
        loss_pseudo = np.mean(np.sum(Y[:, seq_len:].cpu().numpy() - y_hat_pseudo) ** 2)
        print(loss, loss_pseudo)
        check_in_out(
            X[:, seq_len:].cpu().numpy(),
            Y[:, seq_len:].cpu().numpy(),
            y_hat,
            y_hat_pseudo,
            file_path=data_file,
            dataset_name=dataset,
        )


if __name__ == "__main__":
    ckpt_path = "ckpt/transformer/v_9/8.9e-04_val_loss.pt"
    data_file = os.path.abspath("data/train/battery_data.h5")
    dataset = "spme_training_scaled"
    batch_size = 1  # if gradient_accumulation_steps > 1, this is the micro-batch size
    device = "mps"
    model_args = ModelArgs()
    print(f"Resuming training from {ckpt_path}")
    # resume training from a checkpoint.
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_heads", "dim_model", "seq_len", "bias"]:
        model_args.__setattr__(k, checkpoint_model_args.__getattribute__(k))
    # create the model
    model_args.max_seq_len = model_args.seq_len
    seq_len = model_args.seq_len
    model = Transformer(model_args)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
    model.to(device)

    train_data = BatteryData(data_file, dataset, batch_size, seq_len, device)

    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )

    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.autocast(device_type=device_type, dtype=ptdtype)
    )
    estimate_loss()
