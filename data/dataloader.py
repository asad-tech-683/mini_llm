import torch

from data.dataset import TokenDataset


class DataLoaderLite:
    def __init__(
        self,
        B,
        T,
        process_rank,
        num_processes,
        data_dir,
        split="train",
        languages=("hindi", "urdu"),
    ):
        self.B = B
        self.T = T

        self.process_rank = process_rank
        self.num_processes = num_processes

        self.split = split
        self.data_dir = data_dir
        self.languages = tuple(languages)

        # ----------------------------------------------------------
        # Dataset
        # ----------------------------------------------------------

        self.dataset = TokenDataset(
            data_dir=self.data_dir,
            split=self.split,
            languages=self.languages,
        )

        # ----------------------------------------------------------
        # Batch statistics
        # ----------------------------------------------------------

        self.tokens_per_batch = (
            self.B * self.T
        )

        self.tokens_per_global_batch = (
            self.tokens_per_batch
            * self.num_processes
        )

        # ----------------------------------------------------------
        # Per-language statistics
        # ----------------------------------------------------------

        self.num_tokens = {}

        self.batches_per_epoch = {}

        for language in self.languages:
            self.dataset.set_language(language)

            self.num_tokens[language] = (
                self.dataset.num_tokens
            )

            self.batches_per_epoch[language] = (
                self.num_tokens[language]
                // self.tokens_per_global_batch
            )

        # ----------------------------------------------------------
        # Per-language, per-process position
        # ----------------------------------------------------------
        #
        # Every process gets a different starting position.
        #
        # rank 0:
        #   0
        #
        # rank 1:
        #   B*T
        #
        # rank 2:
        #   2*B*T
        #
        # etc.
        #
        # This state is maintained independently for every
        # language.
        # ----------------------------------------------------------

        self.positions = {}

        for language in self.languages:
            self.positions[language] = (
                self.tokens_per_batch
                * self.process_rank
            )

        # ----------------------------------------------------------
        # Active language
        # ----------------------------------------------------------

        self.lang = self.languages[0]

        self.dataset.set_language(
            self.lang
        )

    # --------------------------------------------------------------
    # Language
    # --------------------------------------------------------------

    def set_language(self, language):
        if language not in self.languages:
            raise ValueError(
                f"Unknown language: {language}. "
                f"Available languages: "
                f"{', '.join(self.languages)}"
            )

        self.lang = language

        self.dataset.set_language(
            language
        )

    # --------------------------------------------------------------
    # Current position
    # --------------------------------------------------------------

    @property
    def current_position(self):
        return self.positions[self.lang]

    @current_position.setter
    def current_position(self, value):
        self.positions[self.lang] = value

    # --------------------------------------------------------------
    # Current dataset statistics
    # --------------------------------------------------------------

    @property
    def current_num_tokens(self):
        return self.num_tokens[self.lang]

    @property
    def current_batches_per_epoch(self):
        return self.batches_per_epoch[self.lang]

    # --------------------------------------------------------------
    # Next batch
    # --------------------------------------------------------------

    def next_batch(self):
        position = self.current_position
        num_tokens = self.current_num_tokens

        # We need B*T + 1 tokens because:
        #
        # x = tokens[0 : B*T]
        # y = tokens[1 : B*T+1]
        #
        # So y is shifted by one token.

        end_position = (
            position
            + self.tokens_per_batch
            + 1
        )

        # ----------------------------------------------------------
        # Make sure this process has enough tokens.
        # ----------------------------------------------------------

        if end_position > num_tokens:
            self.reset()

            position = self.current_position

            end_position = (
                position
                + self.tokens_per_batch
                + 1
            )

        # ----------------------------------------------------------
        # Read current language's data.
        # ----------------------------------------------------------

        buf = self.dataset.get_slice(
            position,
            end_position,
        )

        # ----------------------------------------------------------
        # Convert only this batch to PyTorch.
        # ----------------------------------------------------------

        buf = torch.from_numpy(
            buf.astype("int64")
        )

        x = buf[:-1].view(
            self.B,
            self.T,
        )

        y = buf[1:].view(
            self.B,
            self.T,
        )

        # ----------------------------------------------------------
        # Advance this language's position.
        #
        # IMPORTANT:
        #
        # We advance by the GLOBAL batch size, not B*T.
        #
        # Therefore all distributed processes move together
        # through the language stream without overlapping.
        # ----------------------------------------------------------

        self.current_position = (
            position
            + self.tokens_per_global_batch
        )

        # ----------------------------------------------------------
        # Start a new epoch for this language.
        # ----------------------------------------------------------

        if (
            self.current_position
            + self.tokens_per_batch
            + 1
            > num_tokens
        ):
            self.reset()

        return x, y

    # --------------------------------------------------------------
    # Reset current language
    # --------------------------------------------------------------

    def reset(self):
        self.current_position = (
            self.tokens_per_batch
            * self.process_rank
        )

    # --------------------------------------------------------------
    # Reset all languages
    # --------------------------------------------------------------

    def reset_all(self):
        for language in self.languages:
            self.positions[language] = (
                self.tokens_per_batch
                * self.process_rank
            )

    # --------------------------------------------------------------
    # State inspection
    # --------------------------------------------------------------

    def get_positions(self):
        return dict(self.positions)