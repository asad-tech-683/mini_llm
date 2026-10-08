import torch

from data.dataset import TokenDataset


class DataLoaderLite:
    """Streaming dataloader that can switch between languages."""

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

        # Load the dataset; we'll point it at different languages below.
        self.dataset = TokenDataset(
            data_dir=self.data_dir,
            split=self.split,
            languages=self.languages,
        )

        # Local batch size and global batch size, measured in tokens.
        self.tokens_per_batch = self.B * self.T
        self.tokens_per_global_batch = (
            self.tokens_per_batch * self.num_processes
        )

        # Cache per-language token counts and epoch lengths.
        self.num_tokens = {}
        self.batches_per_epoch = {}

        for language in self.languages:
            self.dataset.set_language(language)

            self.num_tokens[language] = self.dataset.num_tokens
            self.batches_per_epoch[language] = (
                self.num_tokens[language] // self.tokens_per_global_batch
            )

        # Each process reads a different slice of the same stream. We keep
        # a separate position per language so switching languages doesn't
        # lose our place.
        self.positions = {}

        for language in self.languages:
            self.positions[language] = (
                self.tokens_per_batch * self.process_rank
            )

        # Start with the first language.
        self.lang = self.languages[0]
        self.dataset.set_language(self.lang)

    def set_language(self, language):
        if language not in self.languages:
            raise ValueError(
                f"Unknown language: {language}. "
                f"Available languages: {', '.join(self.languages)}"
            )

        self.lang = language
        self.dataset.set_language(language)

    @property
    def current_position(self):
        return self.positions[self.lang]

    @current_position.setter
    def current_position(self, value):
        self.positions[self.lang] = value

    @property
    def current_num_tokens(self):
        return self.num_tokens[self.lang]

    @property
    def current_batches_per_epoch(self):
        return self.batches_per_epoch[self.lang]

    def next_batch(self):
        position = self.current_position
        num_tokens = self.current_num_tokens

        # We need one extra token because y is x shifted by one.
        end_position = position + self.tokens_per_batch + 1

        # If this process would read past the end, wrap around first.
        if end_position > num_tokens:
            self.reset()

            position = self.current_position
            end_position = position + self.tokens_per_batch + 1

        buf = self.dataset.get_slice(position, end_position)
        buf = torch.from_numpy(buf.astype("int64"))

        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)

        # Advance by the global batch size, not just the local one, so all
        # ranks move through the language stream without overlapping.
        self.current_position = (
            position + self.tokens_per_global_batch
        )

        # If the next batch would start past the end, reset now so the next
        # call starts from the beginning of this language.
        if (
            self.current_position
            + self.tokens_per_batch
            + 1
            > num_tokens
        ):
            self.reset()

        return x, y

    def reset(self):
        self.current_position = (
            self.tokens_per_batch * self.process_rank
        )

    def reset_all(self):
        for language in self.languages:
            self.positions[language] = (
                self.tokens_per_batch * self.process_rank
            )

    def get_positions(self):
        return dict(self.positions)