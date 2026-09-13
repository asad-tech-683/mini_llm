import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.model.n_embd % config.model.n_head == 0

        self.c_attn = nn.Linear(config.model.n_embd, 3 * config.model.n_embd)
        self.c_proj = nn.Linear(config.model.n_embd, config.model.n_embd)

        self.n_head = config.model.n_head
        self.n_embd = config.model.n_embd

        # self.register_buffer(
        #     "bias",
        #     torch.tril(
        #         torch.ones(config.sequence_length, config.sequence_length)
        #     ).view(1, 1, config.sequence_length, config.sequence_length),
        # )

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(
            B,
            T,
            self.n_head,
            C // self.n_head,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_head,
            C // self.n_head,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            C // self.n_head,
        ).transpose(1, 2)
        
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = torch.softmax(att, dim=-1)
        # y = att @ v
        
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            # attn_mask=self.bias[:, :, :T, :T],
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y