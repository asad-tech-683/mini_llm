import torch
import sentencepiece as spm
from config import Config

class Generator:
    def __init__(self, model):
        self.model = model
        self.tokenizer = sp = spm.SentencePieceProcessor(model_file=str(Config().tokenizer.model_path))

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens,
        top_k=None,
        temperature=1.0,
    ):

        tokens = self.tokenizer.encode(prompt)

        x = torch.tensor(
            tokens,
            dtype=torch.long,
            device=next(self.model.parameters()).device,
        ).unsqueeze(0)

        for _ in range(max_new_tokens):

            x_cond = x[:, -self.model.config.sequence_length:]

            logits, _ = self.model(x_cond)

            logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:
                topk_logits, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1)),
                )

                logits[
                    logits < topk_logits[:, [-1]]
                ] = float("-inf")

            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

            x = torch.cat(
                (x, next_token),
                dim=1,
            )

        return self.tokenizer.decode(
            x[0].tolist()
        )

    @torch.no_grad()
    def generate_many(
        self,
        prompt,
        num_generations,
        max_new_tokens,
        top_k=None,
        temperature=1.0,
    ):
        generations = []

        for _ in range(num_generations):
            generations.append(
                self.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    temperature=temperature,
                )
            )

        return generations
