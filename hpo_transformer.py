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
    EqualsCondition,
    Float,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause,
    Integer,
    Normal,
)
from smac import Callback, Scenario
from smac import MultiFidelityFacade as MFFacade
from smac.intensifier.hyperband import Hyperband

from model.transformer import ModelArgs, Transformer
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

    seq_len = config["seq_len"]
    n_layer = config["n_layer"]
    n_heads = config["n_heads"]
    dim_model = config["dim_model"]
    bias = config["bias"]
    learning_rate = config["learning_rate"]
    pe_type = config["pe_type"]
    rope_theta = config["rope_theta"] if pe_type == "RoPE" else 10000.0
    loss = config["loss"]
    norm_type = config["norm_type"]
    reduction = config["reduction"]
    act_type = config["act_type"]
    max_iters = int(budget)

    eval_interval = 10
    eval_iters = 1
    dataset = "spme_training_scaled"
    data_file = os.path.abspath("data/train/battery_data.h5")

    batch_size = divmod(524_288, seq_len)[0]
    possible_sizes = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
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
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    eval('setattr(torch.backends.cudnn, "allow_tf32", True)')
    # torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
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
    while True:
        gc.collect()
        torch.cuda.empty_cache()
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
                loss=loss,
                norm_type=norm_type,
                rope_theta=rope_theta,
                reduction=reduction,
                act_type=act_type,
                device=device,
            )
            model = Transformer(model_args)
            model.to(device)
            scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
            optimizer = model.configure_optimizers(
                weight_decay, learning_rate, (beta1, beta2), device_type
            )
            model = torch.compile(model)
            model.train()
            X, Y = train_data.get_batch("train")
            with ctx:
                _ = model(X, Y[:, :, 1:])
            del _
            break
        except RuntimeError as e:
            if "out of memory" in str(e) and batch_size > 16:
                del X, Y, model, optimizer, scaler, train_data, model_args
                batch_size //= 2
                gradient_accumulation_steps *= 2
                continue
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
                del X, Y, model, optimizer, scaler, train_data, model_args
                print(
                    f"CUDA OOM, device: {device},  mem_alloc: {memory_allocated / (1024 ** 2):.2f} MB, mem_res: {memory_reserved / (1024 ** 2):.2f} MB"
                )
                return 1e9

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
                        input[:, -1, 0] = X[:, seq_len + i, 0]
                        input[:, -1, 1:] = y[:, -1, :]
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
            out[split] = losses.mean().item()
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

    del X, Y, model, optimizer, scaler, train_data, model_args, loss, _

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
        "cuda:4",
        "cuda:5",
        "cuda:6",
        "cuda:7",
    ]

    with open("gpu_list.txt", "w") as file:
        for gpu in gpu_list:
            file.write(f"{gpu}\n")

    seed = 420
    torch.manual_seed(seed)
    np.random.seed(seed)
    cs = ConfigurationSpace(
        name="transformer",
        seed=seed,
        space={
            "pe_type": Categorical("pe_type", ["RoPE", "ALiBi", "APE"]),
            "norm_type": Categorical("norm_type", ["RMSNorm", "LayerNorm"]),
            "rope_theta": Integer("rope_theta", bounds=(10, 200_000), log=True),
            "act_type": Categorical("act_type", ["SwiGLU", "GeLU"]),
            "loss": Categorical("loss", ["MSE", "MAE", "LogCosh"]),
            "reduction": Categorical("reduction", ["sum", "mean"]),
            "dim_model": Categorical(
                "dim_model", [32, 64, 128, 256, 512, 768], ordered=True
            ),
            "n_heads": Categorical(
                "n_heads",
                [4, 8, 12, 16, 32, 64, 96, 128, 192],
                ordered=True,
            ),
            "seq_len": Categorical(
                "seq_len", [128, 256, 512, 768, 1024, 1536, 2048], ordered=True
            ),
            "n_layer": Integer("n_layer", bounds=(4, 16)),
            "bias": Categorical("bias", [True, False], default=False),
            "learning_rate": Float(
                "learning_rate",
                bounds=(1e-5, 1e-2),
                log=True,
                default=1e-3,
                distribution=Normal(mu=5e-3, sigma=3),
            ),
        },
    )

    cs.add_condition(EqualsCondition(cs["rope_theta"], cs["pe_type"], "RoPE"))

    forbidden_clause_1a = ForbiddenEqualsClause(cs["dim_model"], 32)
    forbidden_clause_1b = ForbiddenInClause(cs["n_heads"], [12, 32, 64, 96, 128, 192])
    forbidden_clause_1 = ForbiddenAndConjunction(
        forbidden_clause_1a, forbidden_clause_1b
    )

    forbidden_clause_2a = ForbiddenEqualsClause(cs["dim_model"], 64)
    forbidden_clause_2b = ForbiddenInClause(cs["n_heads"], [12, 64, 96, 128, 192])
    forbidden_clause_2 = ForbiddenAndConjunction(
        forbidden_clause_2a, forbidden_clause_2b
    )

    forbidden_clause_3a = ForbiddenEqualsClause(cs["dim_model"], 128)
    forbidden_clause_3b = ForbiddenInClause(cs["n_heads"], [12, 96, 128, 192])
    forbidden_clause_3 = ForbiddenAndConjunction(
        forbidden_clause_3a, forbidden_clause_3b
    )

    forbidden_clause_4a = ForbiddenEqualsClause(cs["dim_model"], 256)
    forbidden_clause_4b = ForbiddenInClause(cs["n_heads"], [12, 96, 192])
    forbidden_clause_4 = ForbiddenAndConjunction(
        forbidden_clause_4a, forbidden_clause_4b
    )

    forbidden_clause_5a = ForbiddenEqualsClause(cs["dim_model"], 512)
    forbidden_clause_5b = ForbiddenInClause(cs["n_heads"], [12, 96, 192])
    forbidden_clause_5 = ForbiddenAndConjunction(
        forbidden_clause_5a, forbidden_clause_5b
    )

    cs.add_forbidden_clauses(
        [
            forbidden_clause_1,
            forbidden_clause_2,
            forbidden_clause_3,
            forbidden_clause_4,
            forbidden_clause_5,
        ]
    )

    # Scenario object specifying the optimization environment
    scenario = Scenario(
        configspace=cs,
        name="transformer",
        output_directory=Path(f"{Path.cwd()}/hpo"),
        deterministic=True,
        n_trials=500,
        termination_cost_threshold=0.01,
        min_budget=30,
        max_budget=250,
        n_workers=8,
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
    )

    # Let's optimize
    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(cs.get_default_configuration())
    print(f"Default cost ({intensifier.__class__.__name__}): {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost ({intensifier.__class__.__name__}): {incumbent_cost}")
