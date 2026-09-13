import torch
from data.dataset import TokenDataset

class Evaluator:
    def __init__(
        self,
        *,
        model,
        batch_size,
        sequence_length,
        distributed,
        data_dir,
        languages=("hindi", "urdu"),
        max_batches=None,
    ):
        self.model = model
        self.dataset = TokenDataset(
            data_dir=data_dir,
            split="val",
            languages=languages,
        )
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.distributed = distributed
        self.languages = tuple(languages)
        self.max_batches = max_batches

        self.batch_tokens = batch_size * sequence_length
        self.input_tokens = self.batch_tokens + 1

    @torch.no_grad()
    def evaluate(self):
        try:
            results = {
                lang: self._evaluate_language(lang)
                for lang in self.languages
            }
            results["overall"] = self._combine_results(results)
            return results
        finally:
            # code here
            pass

    def _evaluate_language(self, language):
        self.dataset.set_language(language)

        total = (self.dataset.num_tokens - 1) // self.input_tokens
        if total == 0:
            raise ValueError(
                f"Validation dataset for '{language}' is too small "
                "for one evaluation batch."
            )

        rank = self.distributed.rank
        world = self.distributed.world_size
        batches = range(rank, total, world)

        if self.max_batches is not None:
            batches = list(batches)[: self.max_batches]

        loss_sum = tokens = 0

        for batch_index in batches:
            x, y = self._get_batch(batch_index)

            _, loss = self.model(
                x.to(self.distributed.device),
                targets=y.to(self.distributed.device),
            )

            loss_sum += loss.item() * self.batch_tokens
            tokens += self.batch_tokens

        loss_sum, tokens = self._reduce(loss_sum, tokens)

        if not tokens:
            raise RuntimeError(
                f"No validation tokens evaluated for '{language}'."
            )

        return self._metrics(loss_sum / tokens, tokens)

    def _get_batch(self, batch_index):
        start = (
            batch_index * self.distributed.world_size
            + self.distributed.rank
        ) * self.input_tokens

        buf = self.dataset.get_slice(
            start,
            start + self.input_tokens,
        )

        buf = torch.from_numpy(buf.astype("int64"))

        return (
            buf[:-1].view(self.batch_size, self.sequence_length),
            buf[1:].view(self.batch_size, self.sequence_length),
        )

    def _reduce(self, loss, tokens):
        if not self.distributed.is_distributed:
            return loss, tokens

        values = torch.tensor(
            [loss, tokens],
            dtype=torch.float64,
            device=self.distributed.device,
        )
        torch.distributed.all_reduce(values)

        return values[0].item(), int(values[1].item())

    def _metrics(self, loss, tokens):
        return {
            "loss": loss,
            "perplexity": torch.exp(torch.tensor(loss)).item(),
            "tokens": tokens,
            "batches": tokens // self.batch_tokens,
        }

    def _combine_results(self ,results):
        results = [r for k, r in results.items() if k != "overall"]
        tokens = sum(r["tokens"] for r in results)

        if not tokens:
            return {
                "loss": 0.0,
                "perplexity": 0.0,
                "tokens": 0,
                "batches": 0,
            }

        loss = sum(
            r["loss"] * r["tokens"]
            for r in results
        ) / tokens

        return {
            "loss": loss,
            "perplexity": torch.exp(torch.tensor(loss)).item(),
            "tokens": tokens,
            "batches": tokens // self.batch_tokens,
        }
