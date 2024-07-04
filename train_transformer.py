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
parser.add_argument("--print_gpu", type=int)
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
print_gpu = False
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

gradient_accumulation_steps = 2  # used to simulate larger batch sizes
batch_size = 256 // gradient_accumulation_steps # 524_288 if gradient_accumulation_steps > 1, this is the micro-batch size
seq_len = 2048
# model
n_layer = 18
n_heads = 8
dim_model = 384
learning_rate = 1e-3 # max learning rate
min_lr = 1e-7  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
warmup_iters = 200  # how many steps to warm up for
max_iters = np.floor(
    3_000*0.8 * 360_000 // (gradient_accumulation_steps * batch_size * seq_len)
) * 2 # total number of training iterations
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla

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
# -----------------------------------------------------------------------------
# data init
# verify_dataset_limits(data_file = data_file, dataset=dataset)
# scale_data(file_path=data_file, dataset_name=dataset)
train_data = BatteryData(data_file, dataset, batch_size, seq_len, device)
# model init
model_args = ModelArgs(
    n_layer=n_layer,
    n_heads=n_heads,
    dim_model=dim_model,
    seq_len=seq_len,
    max_seq_len=seq_len,
    bias=bias,
    dropout=dropout,
    pe_type = "RoPE",
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
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0


# FIX: add val and train loss
# helps estimate an arbitrarily accurate loss over either split using many batches
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

lr_scheduler = LRScheduler(
    learning_rate, warmup_iters, max_iters, min_lr, lr_decay_iters
)


if wandb_log:
    wandb.login(key=wandb_api_key)
    wandb.init(dir=out_dir, project=wandb_project, name=wandb_run_name, config=config)


# training loop
X, Y = train_data.get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0
lr = 0
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
            f"val loss {losses['val']:.3e}"
        )
        # FIX: Track pred loss
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")

        # FIX: save on pred loss
                torch.save(
                    checkpoint,
                    os.path.join(
                        out_dir,
                        f"{checkpoint['best_val_loss']:.1e}_val_loss.pt",
                    ),
                )

    # forward backward update, with optional gradient accumulation to simulate larger
    # batch size and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            _, loss = model(X, Y[:,:,1:])
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
        # FIX: track pred loss
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "loss_train": lossf,
                    "val/loss_train": losses["train"],
                    "val/loss_eval": losses["val"],
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
