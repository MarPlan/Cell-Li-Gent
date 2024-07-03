"""
https://github.com/state-spaces/mamba
"""

import inspect
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights
from mamba_ssm.utils.generation import GenerationMixin

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


@dataclass
class ModelArgs:
    dim_out: int = 5
    dim_inp: int = 6
    device: str = "cpu"
    dtype: str = "float16"
    dim_model: int = 256 # hidden size
    n_layer: int = 24
    d_intermediate: int = 1  # MLP after mixer


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,  # hidden size
        n_layer: int,
        d_intermediate: int,  # MLP after mixer
        vocab_size: int,  # input dim
        ssm_cfg=None,  # config for mamba init, Mamba1 default specify {layer: Mamba2}
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,  # for weight init
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Linear(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1
                if d_intermediate == 0
                else 2,  # 2 if we have MLP
            )
        )

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                inference_params=inference_params,
                **mixer_kwargs,
            )
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: ModelArgs,
    ) -> None:
        self.config = config
        initializer_cfg=None,
        device=config.device
        dtype=config.dtype,
        d_model = config.dim_model  # hidden size
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate  # MLP after mixer
        vocab_size = config.dim_inp  # input dim
        dim_out = config.dim_out
        ssm_cfg = {"layer":"Mamba2"}  # config for mamba init, Mamba1 default specify {layer: Mamba2}
        attn_layer_idx = None
        attn_cfg = None
        rms_norm = True
        residual_in_fp32 = False
        fused_add_norm = False
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, dim_out, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(
        self, x, y=None, inference_params=None, num_last_tokens=0, **mixer_kwargs
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(
            x, inference_params=inference_params, **mixer_kwargs
        )
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]

        if y is not None:
            # if we are given some desired targets also calculate the loss
            out = self.lm_head(hidden_states)
            loss = F.mse_loss(out, y, reduction="mean")
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            out = self.lm_head(
                hidden_states[:, [-1], :]
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
        # num_decay_params = sum(p.numel() for p in decay_params)
        # num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(
        #     f"num decayed parameter tensors: {len(decay_params)}, "
        #     f"with {num_decay_params:,} parameters"
        # )
        # print(
        #     f"num non-decayed parameter tensors: {len(nodecay_params)}, "
        #     f"with {num_nodecay_params:,} parameters"
        # )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        # print(f"using fused AdamW: {use_fused}")

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
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        mfu = flops_achieved / flops_promised
        return mfu
