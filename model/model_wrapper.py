import inspect

import lightning as L
import torch
import torch.nn.functional as F


class LitModel(L.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and
        # layernorms don't.
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.config.device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            params=optim_groups,
            lr=self.config.lr,
            betas=self.config.betas,
            **extra_args,
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
