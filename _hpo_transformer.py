"""
Train for half an epoch -> should continue till end
Use max possible memory, catch mem error and reduce batch size, overall 500k, if too big loss inf
Keep warmup, min lr, scheduler fixed, linear decay to min lr within one epoch
Use auto regressive for predciton/test/objective loss, meaning only current external
"""

import fcntl
import gc
import multiprocessing
import os
import time
from contextlib import nullcontext

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ConfigSpace import (
    ConfigurationSpace,
)

from model.transformer import ModelArgs, Transformer
from util.prepare_data import BatteryData


def train(
    config: ConfigurationSpace,
    seed: int = 420,
    budget=55,
    result_queue=None,
    batch_size_new=None,
    grad_acc=None,
):
    def closest_size(batch_size, possible_sizes):
        # Filter out sizes greater than batch_size and find the maximum remaining sizes
        lower_sizes = [size for size in possible_sizes if size <= batch_size]
        if not lower_sizes:
            raise ValueError("No valid batch size found.")
        return max(lower_sizes)

    while True:
        try:
            with open("gpu_list.txt", "r+") as file:
                # Try to lock the file
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                gpu_list = file.readlines()
                # Get the first GPU
                device = gpu_list.pop(0).strip()
                # Truncate the file and write the updated GPU list back to the file
                file.seek(0)
                file.truncate()
                for g in gpu_list:
                    file.write(g)
                # Unlock the file
                fcntl.flock(file, fcntl.LOCK_UN)
                break
        except (IOError, BlockingIOError):
            # If file is already open, wait for 0.3 seconds and try again
            time.sleep(0.1)

    # TODO: Add slice for Y to select out dim
    # slice = slice(:,:,1:)
    seq_len = config["seq_len"]
    n_layer = config["n_layer"]
    n_heads = config["n_heads"]
    dim_model = config["dim_model"]
    pe_type = config["pe_type"]
    rope_theta = config["rope_theta"] if pe_type == "RoPE" else 666

    bias = False
    learning_rate = 2e-3
    loss_type = "MSE"
    norm_type = config["norm_type"]
    reduction = "mean"
    act_type = "SwiGLU"
    max_iters = np.floor(budget)

    eval_interval = max_iters
    eval_iters = 1
    dataset = "spme_training_scaled"
    data_file = os.path.abspath("data/train/battery_data.h5")

    batch_size = divmod(524_288, seq_len)[0]
    # Note more than 3000 based on dataloader and dataset size
    possible_sizes = [
        # 16,
        # 24,
        # 32,
        # 48,
        # 64,
        # 96,
        # 128,
        192,
        256,
        384,
        512,
        768,
        1024,
        1536,
        2048,
    ]
    batch_size = (
        closest_size(batch_size, possible_sizes)
        if batch_size_new is None
        else batch_size_new
    )
    gradient_accumulation_steps = 1 if seq_len != 128 else 2
    gradient_accumulation_steps = (
        gradient_accumulation_steps if grad_acc is None else grad_acc
    )
    lr_decay_iter = 3240 // 2
    min_lr = 1e-9
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    warmup_iters = 100
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    # -----------------------------------------------------------------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    eval('setattr(torch.backends.cudnn, "allow_tf32", True)')
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

    @torch.no_grad()
    def estimate_loss(file_path=data_file, dataset_name=dataset):
        out = {}
        model.eval()
        for split in ["pred"]:
            seed = 420
            torch.manual_seed(seed)
            np.random.seed(seed)
            train_data.first = True
            for k in range(eval_iters):
                X, Y = train_data.get_batch(split)
                if split == "pred":
                    y_hat = []
                    with ctx:
                        input = X[:, :seq_len]
                        for i in range(8192 - seq_len):
                            y, _ = model(input)
                            y_hat.append(y)
                            input = torch.roll(input, -1, 1)
                            input[:, -1, 0] = X[:, seq_len + i, 0]
                            input[:, -1, 1:] = y[:, -1, 1:]
                        y_hat = torch.concatenate(y_hat, dim=1).to(Y.device)
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
                        Y_re = Y * (maxs_expanded - mins_expanded) + mins_expanded
                        y_hat_re = (
                            y_hat * (maxs_expanded - mins_expanded) + mins_expanded
                        )
                    losses_re = F.mse_loss(Y_re[:, -4096:, :], y_hat_re[:, -4096:, :])
                    losses = F.mse_loss(Y[:, -4096:, :], y_hat[:, -4096:, :])
            out[split] = losses.to("cpu").item()
            out[split + "_re"] = losses_re.to("cpu").item()

        fig = plt.figure()
        ax = fig.subplots(5, 1, sharex=True)

        X = X[:, seq_len:].to(torch.float32).cpu().numpy()
        Y = Y[:, seq_len:].to(torch.float32).cpu().numpy()
        y_hat_re = y_hat_re.to(torch.float32).cpu().numpy()

        batch_nr = 0
        for i in range(X.shape[-1]):
            ax[0].plot(X[batch_nr, :, i], label="X")

        for i in [1, 5]:
            ax[1].plot(Y[batch_nr, :, i], label="Y")
            ax[1].plot(y_hat_re[batch_nr, :, i], "--", label="y_hat")

        for i in [2, 4]:
            ax[2].plot(Y[batch_nr, :, i], label="Y")
            ax[2].plot(y_hat_re[batch_nr, :, i], "--", label="y_hat")

        for i in [3]:
            ax[3].plot(Y[batch_nr, :, i], label="Y")
            ax[3].plot(y_hat_re[batch_nr, :, i], "--", label="y_hat")

        for i in [0]:
            ax[4].plot(Y[batch_nr, :, i], label="Y")
            ax[4].plot(y_hat_re[batch_nr, :, i], "--", label="y_hat")

        import time

        plt.tight_layout()

        out_path = f"hpo/loss_re_{out["pred_re"]}_time_{time.time()}.png"
        fig.savefig(out_path, dpi=300)
        model.train()
        return out

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

    lr_scheduler = LRScheduler(
        initial_lr=learning_rate,
        warmup_lr=learning_rate,
        warmup_iters=warmup_iters,
        max_iters=lr_decay_iter,
        min_lr=min_lr,
        decay_iters=lr_decay_iter,
    )
    memory_allocated = torch.cuda.memory_allocated(device=device)
    memory_reserved = torch.cuda.memory_reserved(device=device)
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_data = BatteryData(data_file, dataset, batch_size, seq_len, device)
        model_args = ModelArgs(
            n_layer=n_layer,
            n_heads=n_heads,
            dim_model=dim_model,
            seq_len=seq_len,
            max_seq_len=seq_len,
            bias=bias,
            dropout=0,
            pe_type=pe_type,
            loss=loss_type,
            norm_type=norm_type,
            rope_theta=rope_theta,
            reduction=reduction,
            act_type=act_type,
            device=device,
        )
        model = Transformer(model_args)
        model.to(device)
        model = torch.compile(model)
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
        optimizer = model.configure_optimizers(
            weight_decay, learning_rate, (beta1, beta2), device_type
        )
        model.train()
        X, Y = train_data.get_batch("train")
        lr = learning_rate
        iter_num = 0
        best_pred_loss = 1e9
        while True:
            lr = lr_scheduler.get_lr(iter_num, lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            for micro_step in range(gradient_accumulation_steps):
                with ctx:
                    _, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps
                X, Y = train_data.get_batch("train")
                scaler.scale(loss).backward()
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if (iter_num % eval_interval == 0) and (iter_num > 0):
                losses = estimate_loss()
                best_pred_loss = losses["pred"]
                best_pred_loss_re = losses["pred_re"]

            iter_num += 1
            if iter_num > max_iters:
                break

        n_params = sum(p.numel() for p in model.parameters())
        del loss, _

        print(
            f"device: {device}, loss: {best_pred_loss:.2e}, loss_re: {best_pred_loss_re:.2e}, "
            f"n_params: {n_params / 1e6:.2f}M, batch_size: {batch_size}, "
            f"mem_alloc: {memory_allocated / (1024 ** 2):.2f} MB, "
            f"mem_res: {memory_reserved / (1024 ** 2):.2f} MB"
        )
        result_queue.put(
            {
                "status": "success",
                "loss": best_pred_loss,
                "loss_re": best_pred_loss_re,
                "n_params": n_params,
            }
        )

    except RuntimeError as e:
        if "out of memory" in str(e) and batch_size > 4:
            batch_size //= 2
            gradient_accumulation_steps *= 2
            result_queue.put(
                {
                    "status": "oom",
                    "batch_size": batch_size,
                    "grad_acc": gradient_accumulation_steps,
                }
            )
        else:
            print(config)
            print(e)
            print(
                f"CUDA OOM, device: {device},  mem_alloc: {memory_allocated / (1024 ** 2):.2f} MB, mem_res: {memory_reserved / (1024 ** 2):.2f} MB"
            )
            result_queue.put(
                {
                    "status": "crash",
                    "loss": 1e7,
                }
            )
    finally:
        del X, Y, model, optimizer, scaler, param_group
        while True:
            try:
                with open("gpu_list.txt", "a+") as file:
                    # Try to lock the file
                    fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    file.write(f"{device}\n")
                    # Unlock the file
                    fcntl.flock(file, fcntl.LOCK_UN)
                    break
            except (IOError, BlockingIOError):
                # If file is already open, wait for 0.3 seconds and try again
                time.sleep(0.1)

        torch.cuda.synchronize(device)


def train_wrapper(config: ConfigurationSpace, seed: int = 420, budget=55):
    result_queue = multiprocessing.Queue()

    grad_acc = None
    batch_size_new = None
    while True:
        process = multiprocessing.Process(
            target=train,
            args=(config, seed, budget, result_queue, batch_size_new, grad_acc),
        )
        process.start()
        process.join()  # Wait for the process to complete

        result = result_queue.get()  # Retrieve the result from the queue

        if result["status"] == "success":
            return result["loss"], {
                "loss_re": result["loss_re"],
                "n_params": result["n_params"],
            }
        elif result["status"] == "oom":
            batch_size_new = result["batch_size"]
            grad_acc = result["grad_acc"]
            gc.collect()
            torch.cuda.empty_cache()
        else:
            return result["loss"]


def dask_wrapper(config: ConfigurationSpace, seed: int = 420, budget=55):
    # This function is what Dask will call
    return train_wrapper(config, seed, budget)
