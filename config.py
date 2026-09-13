from dataclasses import dataclass
from pathlib import Path



# =========== Tokenizer ===========
@dataclass(frozen=True)
class TokenizerConfig:
    name: str
    model_path: Path
    vocab_size: int
    data_path: Path


TOKENIZERS = {
    "sp_32k": TokenizerConfig(
        name="sp_32k",
        model_path=Path(
            "data/tokenizer_32k.model"
        ),
        vocab_size=32_000,
        data_path=Path(
            "data/tokenized/sp_32k"
        ),
    ),

    "sp_50k": TokenizerConfig(
        name="sp_50k",
        model_path=Path(
            "data/tokenizer_50k.model"
        ),
        vocab_size=51_200,
        data_path=Path(
            "data/tokenized/sp_50k"
        ),
    ),
}



# =========== Model ===========
@dataclass
class ModelConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    # Set automatically from selected tokenizer.
    vocab_size: int = 0


# =========== Trainer ===========
@dataclass
class TrainConfig:

    # ------------------- Batching -------------------

    sequence_length: int
    
    batch_size: int = 8

    # Total tokens processed before optimizer step.
    total_batch_size: int = 20_480
    
    
    @property
    def grad_accumulation_steps(self) -> int:

        micro_batch_tokens = (
            self.batch_size * self.sequence_length
        )

        if self.total_batch_size % micro_batch_tokens != 0:
            raise ValueError(
                "total_batch_size must be divisible by "
                "batch_size * sequence_length"
            )

        return self.total_batch_size // micro_batch_tokens


    # ------------------- Optimization -------------------

    max_lr: float = 6e-4
    min_lr: float = 6e-5

    warmup_steps: int = 10
    max_steps: int = 1000

    weight_decay: float = 0.1
    grad_clip: float = 1.0


    # ------------------- Reproducibility -------------------

    seed: int = 1337


    # ------------------- Logging -------------------

    log_interval: int = 1
    eval_interval: int = 50
    
    checkpoint_interval = 500


# ============================================================
# Experiment
# ============================================================

@dataclass
class Config:

    # --------------------------------------------------------
    # Experiment selection
    # --------------------------------------------------------

    tokenizer_name: str = "sp_50k"

    # --------------------------------------------------------
    # Sequence length
    #
    # SINGLE SOURCE OF TRUTH
    # --------------------------------------------------------

    sequence_length: int = 256

    # --------------------------------------------------------
    # Sub-configurations
    # --------------------------------------------------------

    model: ModelConfig = None
    train: TrainConfig = None

    def __post_init__(self):

        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------

        if self.tokenizer_name not in TOKENIZERS:
            raise ValueError(
                f"Unknown tokenizer: {self.tokenizer_name}. "
                f"Available: {list(TOKENIZERS)}"
            )

        self.tokenizer = TOKENIZERS[
            self.tokenizer_name
        ]

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        if self.model is None:
            self.model = ModelConfig()

        self.model.vocab_size = (
            self.tokenizer.vocab_size
        )

        # ----------------------------------------------------
        # Trainer
        # ----------------------------------------------------

        if self.train is None:
            self.train = TrainConfig(
                sequence_length=self.sequence_length
            )

    # --------------------------------------------------------
    # Convenience
    # --------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size
