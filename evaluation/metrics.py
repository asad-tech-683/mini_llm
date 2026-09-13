import csv
from pathlib import Path


class EvaluationMetrics:
    FIELDS = (
        ("loss", "Loss", ".6f"),
        ("perplexity", "Perplexity", ".4f"),
        ("tokens", "Tokens", ","),
        ("batches", "Batches", ","),
    )

    def __init__(
        self,
        results=None,
        *,
        step=None,
        elapsed=None,
        log_path="logs/validation.csv",
    ):
        self.results = results
        self.step = step
        self.elapsed = elapsed
        self.log_path = Path(log_path) if log_path else None

        if self.log_path:
            self._initialize_log()

    def _initialize_log(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # if self.log_path.exists():
        #     return

        fields = [
            "step",
            "language",
            "loss",
            "perplexity",
            "tokens",
            "batches",
            "elapsed_time",
        ]

        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    @property
    def languages(self):
        return tuple(k for k in self.results if k != "overall")

    @property
    def overall(self):
        return self.results["overall"]

    def log_step(self):
        """Append the current evaluation results to the CSV log."""
        if not self.log_path:
            return

        fields = [
            "step",
            "language",
            "loss",
            "perplexity",
            "tokens",
            "batches",
            "elapsed_time",
        ]

        rows = []

        for language in self.languages:
            result = self.results[language]

            rows.append({
                "step": self.step,
                "language": language,
                "loss": result["loss"],
                "perplexity": result["perplexity"],
                "tokens": result["tokens"],
                "batches": result["batches"],
                "elapsed_time": self.elapsed,
            })

        result = self.overall

        rows.append({
            "step": self.step,
            "language": "overall",
            "loss": result["loss"],
            "perplexity": result["perplexity"],
            "tokens": result["tokens"],
            "batches": result["batches"],
            "elapsed_time": self.elapsed,
        })

        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerows(rows)
            
    def report(self, *, step=None, elapsed=None):
        title = "Eval" if step is None else f"Eval @ {step}"
        self.step = step

        parts = [title]

        for language in self.languages:
            result = self.results[language]
            parts.append(
                f"{language.capitalize()} "
                f"Loss {result['loss']:.4f} "
                f"PPL {result['perplexity']:.2f}"
            )

        parts.append(
            f"Overall "
            f"Loss {self.overall['loss']:.4f} "
            f"PPL {self.overall['perplexity']:.2f}"
        )

        parts.append(f"{self.overall['tokens'] / 1e3:.2f}K toks")
        parts.append(f"{self.overall['batches']:,} batches")

        if elapsed is not None:
            parts.append(f"{elapsed:.2f}s")

        print("[" + " | ".join(parts) + "]")
        self.log_step()


    def _print_result(self, result):
        for key, label, fmt in self.FIELDS:
            print(f"    {label:<13}: {result[key]:{fmt}}")