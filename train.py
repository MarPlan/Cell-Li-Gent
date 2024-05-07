from dataclasses import dataclass

import lightning as L
import torch
from data.prepare_data import BatteryData
from model.model_wrapper import LitModel
from model.transformer.transformer import ModelArgs, Transformer


@dataclass
class TrainingArgs:
    # default config values designed to train a Transformer 124M parameter
    # I/O
    out_dir = "out"
    eval_interval = 2000
    log_interval = 1
    eval_iters = 200
    init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
    # wandb logging
    wandb_log = False  # disabled by default
    wandb_project = "Cell-Li-Gent"
    wandb_run_name = "transformer"  # 'run' + str(time.time())
    # data
    data_dir = "/data"
    batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
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
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps'
    # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    # -----------------------------------------------------------------------------
    # various inits, derived attributes, I/O setup
    # we are running on a single gpu, and one process
    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]


def training():
    L.seed_everything(42, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # -----------------------------------------------------------------------------
    # for all model args look into /model/<architecture>/<model_name>.py
    train_args = TrainingArgs()
    model_args = ModelArgs(
        n_layer=12,
        n_heads=12,
        dim_model=768,
        dropout=0.0,  # for pretraining 0 is good, for finetuning try 0.1+
        bias=False,  # do we use bias inside LayerNorm and Linear layers?
    )
    # -----------------------------------------------------------------------------
    data = BatteryData(train_args)
    model = Transformer(model_args)
    model = LitModel(model_args, model)
    # saves checkpoints to out_dir at every epoch end
    trainer = L.Trainer(default_root_dir=train_args.out_dir, deterministic=True)

    if train_args == "resume":
        print(f"Resuming training from {train_args.out_dir}")
        # automatically restores model, epoch, step, LR schedulers, etc...
        trainer.fit(model, datamodule=data, ckpt_path=train_args.out_dir)

    else:
        print("Initializing a new model from scratch")

        # NOTE: see if lightning can handle it if not use:
        # unwanted_prefix = "_orig_mod."
        # for k, v in list(state_dict.items()):
        #     if k.startswith(unwanted_prefix):
        #         state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)


if __name__ == "__main__":
    training()
