import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(
            config.model.n_embd,
            4 * config.model.n_embd,
        )

        self.gelu = nn.GELU(approximate="tanh")

        self.c_proj = nn.Linear(
            4 * config.model.n_embd,
            config.model.n_embd,
        )

        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x