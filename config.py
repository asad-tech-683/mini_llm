from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenizerConfig:
    name: str
    model_path: Path
    vocab_size: int
    data_path: Path


# Available tokenizers. Each entry points to the SentencePiece model and the
# directory containing the pre-tokenized data.
TOKENIZERS = {
    "sp_32k": TokenizerConfig(
        name="sp_32k",
        model_path=Path("data/tokenizer_32k.model"),
        vocab_size=32_000,
        data_path=Path("data/tokenized/sp_32k"),
    ),
    "sp_50k": TokenizerConfig(
        name="sp_50k",
        model_path=Path("data/tokenizer_50k.model"),
        vocab_size=51_200,
        data_path=Path("data/tokenized/sp_50k"),
    ),
}


@dataclass
class ModelConfig:
    n_layer: int = 5
    n_head: int = 8
    n_embd: int = 512

    # Filled in automatically from the selected tokenizer.
    vocab_size: int = 0


@dataclass
class TrainConfig:
    sequence_length: int

    batch_size: int = 32

    # Total tokens processed before each optimizer step.
    total_batch_size: int = 102_400

    @property
    def grad_accumulation_steps(self) -> int:
        micro_batch_tokens = self.batch_size * self.sequence_length

        if self.total_batch_size % micro_batch_tokens != 0:
            raise ValueError(
                "total_batch_size must be divisible by "
                "batch_size * sequence_length"
            )

        return self.total_batch_size // micro_batch_tokens

    max_lr: float = 1e-4
    min_lr: float = 1e-5

    warmup_steps: int = 100
    max_steps: int = 1000

    weight_decay: float = 0.01
    grad_clip: float = 1.0

    seed: int = 1337

    log_interval: int = 1
    eval_interval: int = 50
    checkpoint_interval: int = 500


@dataclass
class Config:
    """Top-level experiment configuration."""

    # Which tokenizer to use; must be a key in TOKENIZERS.
    tokenizer_name: str = "sp_32k"

    # Sequence length is defined here and passed down to the trainer.
    sequence_length: int = 128

    # Sub-configs are built in __post_init__ so they can depend on the
    # tokenizer and sequence length chosen above.
    model: ModelConfig = None
    train: TrainConfig = None

    def __post_init__(self):
        if self.tokenizer_name not in TOKENIZERS:
            raise ValueError(
                f"Unknown tokenizer: {self.tokenizer_name}. "
                f"Available: {list(TOKENIZERS)}"
            )

        self.tokenizer = TOKENIZERS[self.tokenizer_name]

        if self.model is None:
            self.model = ModelConfig()

        # The model's vocab size always matches the tokenizer.
        self.model.vocab_size = self.tokenizer.vocab_size

        if self.train is None:
            self.train = TrainConfig(
                sequence_length=self.sequence_length
            )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size