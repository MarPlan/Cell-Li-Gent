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
)
from smac import Callback, Scenario
from smac import MultiFidelityFacade as MFFacade
from smac.intensifier.hyperband import Hyperband

from model.transformer import ModelArgs, Transformer
from util.prepare_data import BatteryData

# torch._dynamo.config.cache_size_limit = 512


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
    n_heads = config["n_heads"]
    dim_model = config["dim_model"]
    pe_type = config["RoPE"]
    rope_theta = config["rope_theta"]

    bias = False
    learning_rate = 2e-3
    loss_type = "MSE"
    norm_type = "RMSNorm"
    reduction = "mean"
    act_type = "SwiGLU"
    max_iters = np.floor(budget)

    eval_interval = 50
    eval_iters = 25
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
    batch_size = closest_size(batch_size, possible_sizes)
    gradient_accumulation_steps = 1
    lr_decay_iter = 3000
    min_lr = 1e-7
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
        # for split in ["train", "val", "pred"]:
        for split in ["val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = train_data.get_batch(split)
                if split == "pred":
                    y_hat = []
                    with ctx:
                        input = X[:, :seq_len]
                        for i in range(6144):
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
                    if k == 1:
                        break
                else:
                    with ctx:
                        _, loss = model(X, Y[:, :, 1:])
                    losses[k] = loss
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
            scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
            optimizer = model.configure_optimizers(
                weight_decay, learning_rate, (beta1, beta2), device_type
            )
            model = torch.compile(model)
            model.train()
            X, Y = train_data.get_batch("train")

            lr = learning_rate
            iter_num = 0
            best_val_loss = 1e9
            while True:
                lr = lr_schedul.get_lr(iter_num, lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                if (iter_num % eval_interval == 0) and (iter_num > 0):
                    losses = estimate_loss()
                    if losses["val"] < best_val_loss:
                        best_val_loss = losses["val"]
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
            if "out of memory" in str(e) and batch_size > 1:
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
                print(config)
                print(e)
                print(
                    f"CUDA OOM, device: {device},  mem_alloc: {memory_allocated / (1024 ** 2):.2f} MB, mem_res: {memory_reserved / (1024 ** 2):.2f} MB"
                )
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
        f"Loss: {best_val_loss:.2f}, device: {device},  mem_alloc: {memory_allocated / (1024 ** 2):.2f} MB, mem_res: {memory_reserved / (1024 ** 2):.2f} MB"
    )

    return best_val_loss


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
        name="transformer",
        seed=seed,
        space={
            "pe_type": Categorical("pe_type", ["RoPE", "APE", "ALiBi"]),
            # "norm_type": Categorical("norm_type", ["RMSNorm", "LayerNorm"]),
            "rope_theta": Float("rope_theta", bounds=(500, 200_000)),
            # "loss": Categorical("loss", ["MSE", "MAE"]),
            # "reduction": Categorical("reduction", ["sum", "mean"]),
            "dim_model": Categorical(
                "dim_model", [64, 128, 256, 384, 512, 768], ordered=True
            ),
            "n_heads": Categorical(
                "n_heads",
                [2, 4, 8, 12, 16, 32, 64, 128, 256, 384, 512],
                ordered=True,
            ),
            "seq_len": Categorical("seq_len", [256, 512, 1024, 2048], ordered=True),
            "n_layer": Integer("n_layer", bounds=(8, 25)),
            # "bias": Categorical("bias", [True, False], default=False),
            # "learning_rate": Float(
            #    "learning_rate",
            #    bounds=(1e-5, 1e-2),
            #    # log=True,
            #    default=1.5e-3,
            #    distribution=Normal(mu=5e-3, sigma=2),
            # ),
        },
    )

    cs.add_condition(EqualsCondition(cs["rope_theta"], cs["pe_type"], "RoPE"))

    # Function to find the forbidden heads for a given dim_model.
    def forbidden_heads_for_dim_model(dim_model, n_heads):
        return [head for head in n_heads if head >= dim_model or dim_model % head != 0]

    # Creating all forbidden clauses.
    forbidden_clauses = []
    for dim_model in cs["dim_model"].sequence:
        forbidden_heads = forbidden_heads_for_dim_model(
            dim_model, cs["n_heads"].sequence
        )
        if forbidden_heads:
            forbidden_dim_clause = ForbiddenEqualsClause(cs["dim_model"], dim_model)
            forbidden_heads_clause = ForbiddenInClause(cs["n_heads"], forbidden_heads)
            forbidden_clauses.append(
                ForbiddenAndConjunction(forbidden_dim_clause, forbidden_heads_clause)
            )

    cs.add_forbidden_clauses(forbidden_clauses)

    forbidden_dim_flash_attn = ForbiddenEqualsClause(cs["dim_model"], 768)
    forbidden_head_flash_attn = ForbiddenEqualsClause(cs["n_heads"], 2)
    forbidden_flash_attn = ForbiddenAndConjunction(
        forbidden_dim_flash_attn, forbidden_head_flash_attn
    )
    cs.add_forbidden_clauses([forbidden_flash_attn])

    # Scenario object specifying the optimization environment
    scenario = Scenario(
        configspace=cs,
        name="transformer_20",
        output_directory=Path(f"{Path.cwd()}/hpo"),
        deterministic=True,
        n_trials=150,
        termination_cost_threshold=0.01,
        min_budget=50,
        max_budget=500,
        n_workers=4,
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
