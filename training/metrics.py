import csv
import time
from pathlib import Path


class TrainingMetrics:
    def __init__(
        self,
        *,
        total_tokens_by_language,
        batch_size,
        sequence_length,
        world_size,
        grad_accumulation_steps,
        max_steps,
        log_path="logs/training.csv",
    ):
        self.total_tokens_by_language = dict(total_tokens_by_language)
        self.languages = tuple(self.total_tokens_by_language)
        if not self.languages:
            raise ValueError("At least one language must be provided.")

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.world_size = world_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_steps = max_steps

        self.tokens_per_micro_batch = batch_size * sequence_length
        self.tokens_per_global_micro_batch = (
            self.tokens_per_micro_batch * world_size
        )
        self.tokens_per_optimizer_step = (
            self.tokens_per_global_micro_batch * grad_accumulation_steps
        )

        self.optimizer_steps_per_epoch = {
            lang: self.total_tokens_by_language[lang] // self.tokens_per_optimizer_step
            for lang in self.languages
        }
        self.tokens_per_epoch = {
            lang: steps * self.tokens_per_optimizer_step
            for lang, steps in self.optimizer_steps_per_epoch.items()
        }
        self.total_training_steps = max_steps

        self.current_step = 0
        self.total_tokens_processed = 0
        self.tokens_processed_by_language = {lang: 0 for lang in self.languages}
        self.total_training_time = 0.0
        self._step_start_time = None

        self.last_language = None
        self.last_loss = None
        self.last_learning_rate = None
        self.last_grad_norm = None
        self.last_step_time = None
        self.last_tokens_per_second = None

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
            "learning_rate",
            "grad_norm",
            "step_time",
            "tokens_per_second",
            "avg_tokens_per_second",
            "total_tokens_processed",
            "elapsed_time",
            "eta_seconds",
        ] + [f"{lang}_tokens_processed" for lang in self.languages]

        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def summary(self):
        print()
        print("=" * 72)
        print("Training data")
        print("=" * 72)
        for lang in self.languages:
            print(f"  {lang.capitalize():<12}: {self.total_tokens_by_language[lang]:,} tokens")

        print()
        print(f"Tokens / GPU micro-batch:    {self.tokens_per_micro_batch:,}")
        print(f"Global tokens / micro-batch: {self.tokens_per_global_micro_batch:,}")
        print(f"Gradient accumulation:       {self.grad_accumulation_steps:,}")
        print(f"Global tokens / optimizer:   {self.tokens_per_optimizer_step:,}")
        print()

        for lang in self.languages:
            print(
                f"{lang.capitalize():<12} steps / epoch: "
                f"{self.optimizer_steps_per_epoch[lang]:,}"
            )
            print(
                f"{lang.capitalize():<12} tokens / epoch: "
                f"{self.tokens_per_epoch[lang]:,}"
            )

        print()
        print(f"Training steps:              {self.max_steps:,}")
        print(
            f"Total tokens to process:     "
            f"{self.tokens_per_optimizer_step * self.max_steps:,}"
        )
        print("=" * 72)
        print()

    def start_step(self):
        if self._step_start_time is not None:
            raise RuntimeError("A training step is already being timed.")
        self._step_start_time = time.perf_counter()

    def end_step(self, *, language, loss, learning_rate, grad_norm):
        if self._step_start_time is None:
            raise RuntimeError("start_step() must be called before end_step().")
        if language not in self.languages:
            raise ValueError(
                f"Unknown language: {language}. "
                f"Available languages: {', '.join(self.languages)}"
            )

        elapsed = time.perf_counter() - self._step_start_time
        self._step_start_time = None

        self.current_step += 1
        self.total_training_time += elapsed
        self.total_tokens_processed += self.tokens_per_optimizer_step
        self.tokens_processed_by_language[language] += self.tokens_per_optimizer_step

        self.last_language = language
        self.last_loss = loss
        self.last_learning_rate = learning_rate
        self.last_grad_norm = grad_norm
        self.last_step_time = elapsed
        self.last_tokens_per_second = (
            self.tokens_per_optimizer_step / elapsed if elapsed > 0 else 0.0
        )

        self.log_step()
        return elapsed

    def log_step(self):
        """Append the latest completed step to the CSV log."""
        if not self.log_path or self.current_step == 0:
            return

        fields = [
            "step",
            "language",
            "loss",
            "learning_rate",
            "grad_norm",
            "step_time",
            "tokens_per_second",
            "avg_tokens_per_second",
            "total_tokens_processed",
            "elapsed_time",
            "eta_seconds",
        ] + [f"{lang}_tokens_processed" for lang in self.languages]

        row = {
            "step": self.current_step,
            "language": self.last_language,
            "loss": self.last_loss,
            "learning_rate": self.last_learning_rate,
            "grad_norm": self.last_grad_norm,
            "step_time": self.last_step_time,
            "tokens_per_second": self.last_tokens_per_second,
            "avg_tokens_per_second": self.tokens_per_second,
            "total_tokens_processed": self.total_tokens_processed,
            "elapsed_time": self.total_training_time,
            "eta_seconds": self.estimate_time_remaining(),
        }
        row.update({
            f"{lang}_tokens_processed": self.tokens_processed_by_language[lang]
            for lang in self.languages
        })

        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

    @property
    def tokens_per_second(self):
        return (
            self.total_tokens_processed / self.total_training_time
            if self.total_training_time > 0
            else 0.0
        )

    @property
    def progress(self):
        return self.current_step / self.max_steps if self.max_steps > 0 else 0.0

    @property
    def tokens_remaining(self):
        total = self.tokens_per_optimizer_step * self.max_steps
        return max(total - self.total_tokens_processed, 0)

    def estimate_epoch_time(self, language=None):
        language = language or self.last_language
        if language is None or self.tokens_per_second <= 0:
            return None
        return self.tokens_per_epoch[language] / self.tokens_per_second

    def estimate_total_training_time(self):
        if self.tokens_per_second <= 0:
            return None
        return (
            self.tokens_per_optimizer_step * self.max_steps
            / self.tokens_per_second
        )

    def estimate_time_remaining(self):
        if self.tokens_per_second <= 0:
            return None
        return self.tokens_remaining / self.tokens_per_second

    @staticmethod
    def format_time(seconds):
        if seconds is None:
            return "unknown"
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.1f}m"
        return f"{minutes / 60:.2f}h"
    
    def report_step(self):
        if self.current_step == 0:
            return

        epoch_time = self.estimate_epoch_time()
        remaining = self.estimate_time_remaining()

        parts = [
            f"Step {self.current_step}/{self.max_steps}",
            self.last_language.capitalize(),
            f"Loss {self.last_loss:.4f}",
            f"LR {self.last_learning_rate:.2e}",
            f"Grad {self.last_grad_norm:.4f}",
            self.format_time(self.last_step_time),
            f"Avg {self.tokens_per_second:,.0f} tok/s",
            f"Processed {self.total_tokens_processed / 1e6:.2f}M toks",
            f"{self.last_language.capitalize()} "
            f"{self.tokens_processed_by_language[self.last_language] / 1e6:.2f}M",
            f"Elapsed {self.format_time(self.total_training_time)}",
            f"ETA {self.format_time(remaining)}",
        ]

        if epoch_time is not None:
            parts.append(
                f"Epoch {self.format_time(epoch_time)}"
            )

        print("[" + " | ".join(parts) + "]")


    def state_dict(self):
        return {
            "current_step": self.current_step,
            "total_tokens_processed": self.total_tokens_processed,
            "tokens_processed_by_language": dict(self.tokens_processed_by_language),
            "total_training_time": self.total_training_time,
        }

    def load_state_dict(self, state):
        self.current_step = state["current_step"]
        self.total_tokens_processed = state["total_tokens_processed"]
        self.tokens_processed_by_language = dict(
            state["tokens_processed_by_language"]
        )
        self.total_training_time = state["total_training_time"]
