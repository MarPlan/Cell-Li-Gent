"""
Basically a copy of:
https://github.com/karpathy/nanoGPT/blob/master/model.py
https://github.com/meta-llama/llama3/blob/main/llama/model.py
with only minor adjustments
*thank you*
"""

import inspect
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    pe_type: str = "default"
    norm_type: str = "default"
    dim_model: int = 128
    dim_out: int = 128
    dim_inp: int = 1
    n_heads: int = 16
    seq_len: int = 256
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    dropout: float = 1.0
    n_layer: int = 10
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?


def precompute_abs_pos(config: ModelArgs):
    # Compute the positional encodings in advance
    position = torch.arange(0, config.max_seq_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, config.dim_model, 2)
        * -(torch.log(torch.tensor(10000.0)) / config.dim_model)
    )
    positional_encoding = torch.zeros(config.max_seq_len, config.dim_model)
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)
    return positional_encoding


def apply_absolute_emb(x, positional_encoding):
    # Add positional encoding to input
    x = x + positional_encoding[: x.size(0), :]
    return x


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.pe_type = config.pe_type
        if config.pe_type == "RoPE":
            self.freqs_cis = precompute_freqs_cis(
                config.dim_model // config.n_heads,
                config.max_seq_len * 2,
                config.rope_theta,
            )
            self.rope = apply_rotary_emb
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move
        # head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # causal self-attention; Self-attend:
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        if self.pe_type == "RoPE":
            q, k = self.rope(q, k, freqs_cis=self.freqs_cis)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.act_type == "SwiGLU":
            self.act = SwiGLU()
        else:
            self.act = nn.GELU()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, config, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(config.dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.norm_type == "RMSNorm":
            self.norm = RMSNorm(config.d_model)
        else:
            self.norm = nn.LayerNorm(config.d_model, bias=False)
        self.ln_1 = self.norm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = self.norm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        self.config = config
        if config.pe_type != "RoPE":
            self.abs_pos = precompute_abs_pos(config)
            self.abs_pe = apply_absolute_emb

        self.transformer = nn.ModuleDict(
            dict(
                inp_emb=nn.Embedding(config.dim_inp, config.dim_model),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.dim_model)
                if config.norm_type == "RMSNorm"
                else nn.LayerNorm(config.dim_model, bias=False),
            )
        )
        self.output = nn.Linear(config.dim_model, config.dim_out, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, y=None):
        x = self.transformer.inp_emb(x)
        if self.config.pe_type != "RoPe":
            x = self.abs_pe(x, self.abs_pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if y is not None:
            # if we are given some desired targets also calculate the loss
            out = self.output(x)
            loss = F.mse_loss(out, y)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            out = self.output(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return out, loss

    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed,
        # otherwise no. i.e. all weight tensors in matmuls + embeddings decay,
        # all biases and layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt, flops_promised):
        """estimate model flops utilization (MFU) in units of bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        # scale up according to input dim of the time series, perhaps conversion to
        L, H, Q, T = (
            self.config.n_layer,
            self.config.n_heads,
            self.config.dim_model // self.config.n_heads,
            self.config.seq_len,
        )
        # a single token is not correct this way
        D = self.config.dim_inp
        flops_per_token = 6 * N * D + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        mfu = flops_achieved / flops_promised
        return mfu
