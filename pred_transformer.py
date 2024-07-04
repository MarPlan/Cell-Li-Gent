import os
from contextlib import nullcontext

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from model.transformer import ModelArgs, Transformer
from util.prepare_data import BatteryData


def check_in_out(x, y, y_hat, y_hat_pseudo, file_path=None, dataset_name=None, ckpt_path=None):
    fig = plt.figure(dpi=300)
    ax = fig.subplots(4, 1, sharex=True)

    batch_nr = 0
    for i in range(x.shape[-1]):
        ax[0].plot(x[batch_nr, :, i], label="X")

    for i in [0, 4]:
        ax[1].plot(y[batch_nr, :, i], label="Y")
        ax[1].plot(y_hat[batch_nr, :, i], "--", label="y_hat")
        ax[1].plot(y_hat_pseudo[batch_nr, :, i], ":", label="y_hat_pseudo")

    for i in [1,3]:
        ax[2].plot(y[batch_nr, :, i], label="Y")
        ax[2].plot(y_hat[batch_nr, :, i], "--", label="y_hat")
        ax[2].plot(y_hat_pseudo[batch_nr, :, i], ":", label="y_hat_pseudo")

    for i in [2]:
        ax[3].plot(y[batch_nr, :, i], label="Y")
        ax[3].plot(y_hat[batch_nr, :, i], "--", label="y_hat")
        ax[3].plot(y_hat_pseudo[batch_nr, :, i], ":", label="y_hat_pseudo")

    plt.tight_layout()
    plt.show()
    fig.savefig(ckpt_path.replace(".pt", ".png"), dpi=300)
    plt.show()


@torch.no_grad()
def estimate_loss(file_path=None, dataset_name=None):
    out = {}
    model.eval()
    for split in ["pred"]:
        #losses = torch.zeros(eval_iters)
        for k in range(1):
            X, Y = train_data.get_batch(split)
            y_hat = []
            y_hat_pseudo = []
            with ctx:
                input = X[:, :seq_len]
                input_pseudo = X[:, :seq_len]
                for i in range(seq_len*3):
                    y, _ = model(input)
                    y_hat.append(y)
                    input = torch.roll(input, -1, 1)
                    input[:, -1, 0] = X[:, seq_len + i, 0]
                    input[:, -1, 1:] = y[:, -1]

                    y, _ = model(input_pseudo)
                    y_hat_pseudo.append(y)
                    input_pseudo = torch.roll(input_pseudo, -1, 1)
                    input_pseudo[:, -1, :3] = X[:, seq_len + i, :3]
                    input_pseudo[:, -1, 3:] = y[:, -1, 2:]
                y_hat = torch.concatenate(y_hat, dim=1).to(Y.device)
                y_hat_pseudo = torch.concatenate(y_hat_pseudo, dim=1).to(Y.device)
                # Perform the rescaling using broadcasting
                with h5py.File(file_path, "r") as file:
                    data_scaled = file[dataset_name]
                    mins, maxs = (
                        data_scaled.attrs["min_values"],
                        data_scaled.attrs["max_values"],
                    )
                maxs_expanded = torch.tensor(
                    maxs[np.newaxis, np.newaxis, :], device=X.device
                )
                mins_expanded = torch.tensor(
                    mins[np.newaxis, np.newaxis, :], device=X.device
                )
                X = X * (maxs_expanded - mins_expanded) + mins_expanded
                Y = Y * (maxs_expanded - mins_expanded) + mins_expanded
                y_hat = (
                    y_hat * (maxs_expanded[:, :, 1:] - mins_expanded[:, :, 1:])
                        + mins_expanded[:, :, 1:]
                )

                y_hat_pseudo = (
                    y_hat_pseudo * (maxs_expanded[:, :, 1:] - mins_expanded[:, :, 1:])
                        + mins_expanded[:, :, 1:]
                )
                y_hat_loss = F.l1_loss(Y[:, -4096:, 1:], y_hat[:, -4096:])
                y_hat_pseudo_loss = F.l1_loss(Y[:, -4096:, 1:], y_hat_pseudo[:, -4096:])
                print(f"Loss: {y_hat_loss}, Loss pseudo: {y_hat_pseudo_loss}")

        check_in_out(
            X[:, seq_len:].cpu().numpy(),
            Y[:, seq_len:, 1:].cpu().numpy(),
            y_hat.cpu().numpy(),
            y_hat_pseudo.cpu().numpy(),
            file_path=data_file,
            dataset_name=dataset,
            ckpt_path=ckpt_path

        )
        #out[split] = losses.mean().to("cpu").item()
    #model.train()
    return out



if __name__ == "__main__":

    ckpt_path = "ckpt/transformer/v_1/7.0e-06_val_loss.pt"
    model_args = ModelArgs()

    torch.cuda.empty_cache()
    torch.manual_seed(422)
    np.random.seed(422)

    data_file = os.path.abspath("data/train/battery_data.h5")
    dataset = "spme_training_scaled"
    batch_size = 1  # if gradient_accumulation_steps > 1, this is the micro-batch size
    device = "cuda"
    
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

    estimate_loss(data_file, dataset)

