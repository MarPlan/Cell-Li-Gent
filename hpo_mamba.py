"""
Train for half an epoch -> should continue till end
Use max possible memory, catch mem error and reduce batch size, overall 500k, if too big loss inf
Keep warmup, min lr, scheduler fixed, linear decay to min lr within one epoch
Use auto regressive for predciton/test/objective loss, meaning only current external
"""

import fcntl
import gc
import os
import time
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Integer,
)
from model.mamba import MambaLMHeadModel, ModelArgs
from smac import Callback, Scenario
from smac import MultiFidelityFacade as MFFacade
from smac.intensifier.hyperband import Hyperband
from util.prepare_data import BatteryData


def train(config: ConfigurationSpace, seed: int = 420, budget=55):
    def closest_size(batch_size, possible_sizes):
        # Filter out sizes greater than batch_size and find the maximum of the remaining sizes
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
    dim_model = config["dim_model"]
    d_intermediate = config["d_intermediate"]


    learning_rate = 2e-3
    max_iters = np.floor(budget)
    eval_interval = np.floor(max_iters / 4)
    eval_iters = 1
    dataset = "spme_training_scaled"
    data_file = os.path.abspath("data/train/battery_data.h5")

    batch_size = divmod(524_288, seq_len)[0]
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
        # 1536,
        # 2048,
        # 3072,
        # 4096,
        # 6144,
        # 8192,
    ]
    batch_size = closest_size(batch_size, possible_sizes)
    gradient_accumulation_steps = 1
    lr_decay_iter = 2060
    min_lr = learning_rate / 10
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    warmup_iters = 10
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
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = train_data.get_batch(split)
                y_hat = []
                with ctx:
                    input = X[:, :seq_len]
                    for i in range(4096):
                        y, _ = model(input)
                        y_hat.append(y)
                        input = torch.roll(input, -1, 1)
                        input[:, -1, :3] = X[:, seq_len + i, :3]
                        input[:, -1, 3:] = y[:, -1, 2:]
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
                    # X = X * (maxs_expanded - mins_expanded) + mins_expanded
                    Y = Y * (maxs_expanded - mins_expanded) + mins_expanded
                    y_hat = (
                        y_hat * (maxs_expanded[:, :, 1:] - mins_expanded[:, :, 1:])
                        + mins_expanded[:, :, 1:]
                    )
                    losses[k] = F.mse_loss(Y[:, -4096:, 1:], y_hat[:, -4096:])
            out[split] = losses.mean().to("cpu").item()
        model.train()
        return out

    # learning rate decay scheduler (warmup, linear decay, cool down to 0)
    class LRScheduler:
        def __init__(
            self, learning_rate, warmup_iters, max_iters, min_lr, lr_decay_iter
        ):
            self.learning_rate = learning_rate
            self.warmup_iters = warmup_iters
            self.max_iters = max_iters
            self.min_lr = min_lr
            self.decay = True
            self.lr_step = 0
            self.lr_decay_iter = lr_decay_iter

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
                    self.lr_decay_iter - self.warmup_iters
                )
                lr_prev -= self.lr_step
                return lr_prev
            else:
                return lr_prev

    lr_schedul = LRScheduler(
        learning_rate, warmup_iters, max_iters, min_lr, lr_decay_iter
    )
    while True:
        torch.cuda.synchronize(device)
        gc.collect()
        # torch._dynamo.reset()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated(device=device)
        memory_reserved = torch.cuda.memory_reserved(device=device)
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
            train_data = BatteryData(data_file, dataset, batch_size, seq_len, device)
            model_args = ModelArgs(
                    dim_out = 5,
                    dim_inp = 6,
                    device = device,
                    dtype = ptdtype,
                    dim_model = dim_model, # hidden size
                    n_layer = n_layer,
                    d_intermediate = d_intermediate,  # MLP after mixer
            )
            model = MambaLMHeadModel(model_args)
            model.to(device)
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
                lr = lr_schedul.get_lr(iter_num, lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                if (iter_num % eval_interval == 0) and (iter_num > 0):
                    losses = estimate_loss()
                    if losses["pred"] < best_pred_loss:
                        best_pred_loss = losses["pred"]
                for micro_step in range(gradient_accumulation_steps):
                    with ctx:
                        _, loss = model(X, Y[:, :, 1:])
                        loss = loss / gradient_accumulation_steps
                    X, Y = train_data.get_batch("train")
                    scaler.scale(loss).backward()
                if grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                iter_num += 1
                if iter_num > max_iters:
                    break
            del loss, _, param_group
            del optimizer, scaler
            del X, Y, model
            del train_data, model_args
            torch.cuda.synchronize(device)
            gc.collect()
            # torch._dynamo.reset()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
            break
        except RuntimeError as e:
            if "out of memory" in str(e) and batch_size > 16:
                del memory_allocated
                del memory_reserved
                del scaler, lr, micro_step
                del param_group
                del optimizer
                del (
                    X,
                    Y,
                    model,
                )
                torch.cuda.synchronize(device)
                gc.collect()
                # torch._dynamo.reset()
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.empty_cache()
                batch_size //= 2
                gradient_accumulation_steps *= 2
                print(f"{device} reduced batch size to {batch_size}")
            else:
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
                del X, Y, model, optimizer, scaler, train_data, model_args, param_group
                torch.cuda.synchronize(device)
                gc.collect()
                # torch._dynamo.reset()
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.empty_cache()
                print(
                    f"CUDA OOM, device: {device},  mem_alloc: {memory_allocated / (1024 ** 2):.2f} MB, mem_res: {memory_reserved / (1024 ** 2):.2f} MB"
                )
                print(e)
                return 1e7

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

    gc.collect()
    # torch._dynamo.reset()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()

    print(
        f"Loss: {best_pred_loss:.2f}, device: {device},  mem_alloc: {memory_allocated / (1024 ** 2):.2f} MB, mem_res: {memory_reserved / (1024 ** 2):.2f} MB"
    )

    return best_pred_loss


if __name__ == "__main__":
    gpu_list = [
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
        # "cuda:4",
        # "cuda:5",
        # "cuda:6",
        # "cuda:7",
    ]

    with open("gpu_list.txt", "w") as file:
        for gpu in gpu_list:
            file.write(f"{gpu}\n")

    seed = 420
    torch.manual_seed(seed)
    np.random.seed(seed)

    cs = ConfigurationSpace(
        name="mamba",
        seed=seed,
        space={
            "d_intermediate" : Categorical("d_intermediate", [0,1], ordered=True),
            "dim_model": Categorical(
                "dim_model", [256, 512, 768], ordered=True
            ),
            "seq_len": Categorical("seq_len", [512, 768, 1024, 1536, 2048], ordered=True),
            "n_layer": Integer("n_layer", bounds=(6, 30)),
        },
    )

    # Scenario object specifying the optimization environment
    scenario = Scenario(
        configspace=cs,
        name="mamba",
        output_directory=Path(f"{Path.cwd()}/hpo"),
        deterministic=True,
        n_trials=150,
        termination_cost_threshold=0.01,
        min_budget=20,
        max_budget=150,
        n_workers=1,
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

    # Create our intensifier
    intensifier = Hyperband(scenario, incumbent_selection="highest_budget")

    class CustomCallback(Callback):
        def __init__(self) -> None:
            pass

        def on_iteration_start(self, smbo) -> None:
            gc.collect()
            # torch._dynamo.reset()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
            return None

    # Create our SMAC object and pass the scenario and the train method
    smac = MFFacade(
        scenario,
        train,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=False,
        logging_level=20,
        callbacks=[CustomCallback()],
    )

    # Let's optimize
    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(cs.get_default_configuration())
    print(f"Default cost ({intensifier.__class__.__name__}): {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost ({intensifier.__class__.__name__}): {incumbent_cost}")
