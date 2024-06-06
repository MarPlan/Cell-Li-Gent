"""
This training scirpt is basically an exact copy of:
https://github.com/karpathy/nanoGPT/blob/master/train.py
*thank you*

Minor adjustments for a different model, data and single GPU only
"""

import math
import os
import time
from contextlib import nullcontext

import torch
import wandb

from model.transformer import ModelArgs, Transformer
from tests.data_limits import verify_dataset_limits
from util.config_data import scale_data
from util.prepare_data import BatteryData

# -----------------------------------------------------------------------------
# default config values designed to train a Transformer with 124M params
# I/O
out_dir = "ckpt/transformer/"
eval_interval = 2000
log_interval = 1
eval_iters = 200
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "Cell-Li-Gent"
wandb_run_name = "transformer"  # 'run' + str(time.time())
# data
dataset = "spme_training_scaled"
data_file = os.path.abspath("data/train/battery_data.h5")
gradient_accumulation_steps = 1 * 4  # used to simulate larger batch sizes
batch_size = 32  # if gradient_accumulation_steps > 1, this is the micro-batch size
seq_len = 256
# model
n_layer = 8
n_heads = 4
dim_model = 512
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
device = "mps"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps'
# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
# TODO: use PyTorch 2.0 to compile the model to be faster CRASHING
compile = False
flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
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
model_args = ModelArgs( n_layer=n_layer, n_heads=n_heads, dim_model=dim_model, seq_len=seq_len, max_seq_len=seq_len, bias=bias, dropout=dropout)  # start with model_args from command line
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

# TODO: CRASHES compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = train_data.get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# TODO: chrashed
# wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# training loop
X, Y = train_data.get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )
        if wandb_log:
            # TODO: save file to out_dir, dont override existing files! sub dir or file name
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
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
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")

                torch.save(
                    checkpoint,
                    os.path.join(
                        out_dir, f"{checkpoint['best_val_loss']:.1e}_val_loss.pt"
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
            f"mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# TODO: make init script function!
if __name__ == "__main__":
    pass
