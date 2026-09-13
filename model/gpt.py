import torch
import torch.nn as nn

from .block import Block


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.model.n_embd),
                wpe=nn.Embedding(config.sequence_length, config.model.n_embd),
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.model.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.model.n_embd),
            )
        )

        self.lm_head = nn.Linear(
            config.model.n_embd,
            config.vocab_size,
            bias=False,
        )
        
        # Tie input and output embeddings.
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02

            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.model.n_layer) ** -0.5

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=std,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, idx, targets=None):
        B, T = idx.size()

        assert T <= self.config.sequence_length, (
            f"Cannot forward sequence of length {T}, "
            f"block size is only {self.config.sequence_length}"
        )

        pos = torch.arange(
            0,
            T,
            dtype=torch.long,
            device=idx.device,
        )

        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss