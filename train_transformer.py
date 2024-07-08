"""
This training scirpt is basically an exact copy of:
https://github.com/karpathy/nanoGPT/blob/master/train.py
*thank you*

Minor adjustments for a different model, data and single GPU only
"""

import argparse
import os
import time
from contextlib import nullcontext

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from model.transformer import ModelArgs, Transformer
from util.prepare_data import BatteryData

parser = argparse.ArgumentParser()
# file system input / output
parser.add_argument(
    "--data_file",
    type=str,
)
parser.add_argument(
    "--out_dir",
    type=str,
)
parser.add_argument(
    "--dataset",
    type=str,
)

parser.add_argument(
    "--batch_size",
    type=int,
)
parser.add_argument(
    "--seq_len",
    type=int,
)
parser.add_argument(
    "--n_layer",
    type=int,
)
parser.add_argument(
    "--n_heads",
    type=int,
)
parser.add_argument(
    "--dim_model",
    type=int,
)
parser.add_argument("--sequence_length", type=int)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
)

parser.add_argument(
    "--max_iters",
    type=int,
)

# optimization
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--warmup_iters", type=int)
parser.add_argument("--weight_decay", type=float)
parser.add_argument("--grad_clip", type=float)
# evaluation
parser.add_argument("--wandb_log", type=int)
parser.add_argument("--eval_interval", type=str)
# # memory management
parser.add_argument("--device", type=str)
parser.add_argument("--compile", type=int)
parser.add_argument("--dtype", type=str)
parser.add_argument("--wandb_api_key", type=str)


# -----------------------------------------------------------------------------
# default config values designed to train a Transformer with 124M params
# I/O
wandb_api_key = ""
out_dir = "ckpt/transformer/"
eval_interval = 250
log_interval = 1
eval_iters = 200
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = 1  # disabled by default
wandb_project = "Cell-Li-Gent"
wandb_run_name = "transformer"  # 'run' + str(time.time())
# data
dataset = "spme_training_scaled"
data_file = os.path.abspath("data/train/battery_data.h5")

pe_type = "APE"
rope_theta = 666
seq_len = 512
n_layer = 10
n_heads = 8
dim_model = 386

gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = (
    524_288 // seq_len // gradient_accumulation_steps
)  # 524_288 if gradient_accumulation_steps > 1, this is the micro-batch size
max_iters = (
    np.floor(
        3_000 * 0.8 * 360_000 // (gradient_accumulation_steps * batch_size * seq_len)
    )
    * 40
)  # total number of training iterations

learning_rate = 2e-3  # max learning rate
min_lr = 0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
warmup_iters = 200  # how many steps to warm up for
decay_iters = np.floor(
    3_000 * 0.8 * 360_000 / seq_len / gradient_accumulation_steps / batch_size
)  # how many steps to decay for ~1 epoch to min_lr
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+

bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
# step =  batch_size * seq_len * gradient_accumulation_steps # 32_768 datapoints per iteration
# iterations = 3_000*360_000 / step # iterations for one epoch
# batches * time series resulting in iteration for one epoch
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps'
# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
# use PyTorch 2.0 to compile the model to be faster
compile = True
flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
configs = parser.parse_args()
parser_dict = vars(configs)
for key in parser_dict:
    if parser_dict[key] is not None:
        config[key] = parser_dict[key]
        globals()[key] = parser_dict[key]
# -----------------------------------------------------------------------------
# various inits, derived attributes, I/O setup
# consider the input is of shape [batch_size, seq_len, number_inputs]
inp_values_per_iter = gradient_accumulation_steps * batch_size * seq_len
print(f"tokens per iteration will be: {inp_values_per_iter:,}")

# Ensure the base directory does exist by creating if it does not
os.makedirs(out_dir, exist_ok=True)
# Check if the directory exists
if os.path.exists(out_dir):
    version = 1
    # Try new subdirectories with an increasing version number
    while True:
        new_out_dir = os.path.join(out_dir, f"v_{version}")
        if not os.path.exists(new_out_dir):
            os.makedirs(new_out_dir)
            print(
                f"Created new directory {new_out_dir} because {out_dir} already exists."
            )
            out_dir = new_out_dir
            break
        version += 1

torch.manual_seed(420)
np.random.seed(420)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
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

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
best_pred_loss = 1e9
# -----------------------------------------------------------------------------
# data init
# verify_dataset_limits(data_file = data_file, dataset=dataset)
# scale_data(file_path=data_file, dataset_name=dataset)
train_data = BatteryData(data_file, dataset, batch_size, seq_len, device)
# model init
model_args = ModelArgs(
    rope_theta=rope_theta,
    n_layer=n_layer,
    n_heads=n_heads,
    dim_model=dim_model,
    seq_len=seq_len,
    max_seq_len=seq_len,
    bias=bias,
    dropout=dropout,
    pe_type=pe_type,
    device=device,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = Transformer(model_args)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "dim_model", "seq_len", "bias"]:
        setattr(model_args, k, checkpoint_model_args[k])
    # create the model
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

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)  # requires PyTorch 2.0


@torch.no_grad()
def estimate_loss(file_path=data_file, dataset_name=dataset):
    out = {}
    model.eval()
    for split in ["train", "val", "pred"]:
        losses = torch.zeros(eval_iters)
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
                        input[:, -1, :3] = X[:, seq_len + i, :3]
                        input[:, -1, 3:] = y[:, -1, 3:]
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
                    Y_re = Y * (maxs_expanded - mins_expanded) + mins_expanded
                    y_hat_re = y_hat * (maxs_expanded - mins_expanded) + mins_expanded
                losses_re = F.mse_loss(Y_re[:, -4096:, :], y_hat_re[:, -4096:, :])
                losses = F.mse_loss(Y[:, -4096:, :], y_hat[:, -4096:, :])
                if k == 1:
                    break
            else:
                with ctx:
                    _, loss = model(X, Y[:, :, 1:])
                losses[k] = loss
        out[split] = losses.mean().to("cpu").item()
        if split == "pred":
            out[split + "_re"] = losses_re.mean().to("cpu").item()
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
    warmup_iters=200,
    max_iters=decay_iters,
    min_lr=min_lr,
    decay_iters=decay_iters,
)


if wandb_log:
    wandb.login(key=wandb_api_key)
    wandb.init(dir=out_dir, project=wandb_project, name=wandb_run_name, config=config)


# training loop
X, Y = train_data.get_batch("train")  # fetch the very first batch
local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0
lr = 0
t0 = time.time()
while True:
    # determine and set the learning rate for this iteration
    lr = lr_scheduler.get_lr(iter_num, lr) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"EVAL: "
            f"step {iter_num}: train loss {losses['train']:.3e}, "
            f"val loss {losses['val']:.3e}, "
            f"pred loss {losses['pred']:.3e}, "
            f"pred loss re {losses['pred_re']:.3e}, "
        )
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "best_pred_loss": best_pred_loss,
                    "best_pred_loss_re": losses["pred_re"],
                    "config": config,
                }
                print(f"saving val checkpoint to {out_dir}")

                torch.save(
                    checkpoint,
                    os.path.join(
                        out_dir,
                        f"{checkpoint['best_val_loss']:.1e}_val_loss.pt",
                    ),
                )

        if losses["pred"] < best_pred_loss:
            best_pred_loss = losses["pred"]
            if iter_num > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "best_pred_loss": best_pred_loss,
                    "best_pred_loss_re": losses["pred_re"],
                    "config": config,
                }
                print(f"saving pred checkpoint to {out_dir}")

                torch.save(
                    checkpoint,
                    os.path.join(
                        out_dir,
                        f"{checkpoint['best_pred_loss']:.1e}_pred_loss.pt",
                    ),
                )
    # forward backward update, with optional gradient accumulation to simulate larger
    # batch size and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            _, loss = model(X, Y)
            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while
        # model is doing the forward pass on the GPU
        X, Y = train_data.get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss
        # (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            # GPU peak flops
            mfu = model.estimate_mfu(
                batch_size * gradient_accumulation_steps, dt, flops_promised
            )
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: train loss {lossf:.1e}, time {dt*1000:.2f}ms, "
            f"norm {norm:.1e}, "
            f"lr {lr:.1e}, "
            f"mfu {running_mfu*100:.2f}%"
        )

        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "loss_train": lossf,
                    "val/loss_train": losses["train"],
                    "val/loss_eval": losses["val"],
                    "val/loss_pred": losses["pred"],
                    "val/loss_pred_re": losses["pred_re"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                    "norm": norm.item(),  # convert to percentage
                }
            )

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
