from pathlib import Path

import numpy as np


class TokenDataset:
    def __init__(
        self,
        data_dir,
        split,
        languages=("hindi", "urdu"),
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.languages = tuple(languages)

        if not self.languages:
            raise ValueError(
                "At least one language must be provided."
            )

        # ----------------------------------------------------------
        # Load all language states
        # ----------------------------------------------------------

        self.states = {}

        for language in self.languages:
            self.states[language] = self._load_language(
                language
            )

        # ----------------------------------------------------------
        # Currently active language
        # ----------------------------------------------------------

        self.lang = self.languages[0]

    # ==============================================================
    # Language loading
    # ==============================================================

    def _load_language(self, language):
        split_dir = (
            self.data_dir
            / self.split
            / language
        )

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Dataset split not found: "
                f"{split_dir}"
            )

        # ----------------------------------------------------------
        # Find shards
        # ----------------------------------------------------------

        shard_paths = sorted(
            split_dir.glob("shard_*.npy")
        )

        if not shard_paths:
            raise FileNotFoundError(
                f"No token shards found in: "
                f"{split_dir}"
            )

        # ----------------------------------------------------------
        # Memory-map shards
        # ----------------------------------------------------------

        shards = []

        for path in shard_paths:
            tokens = np.load(
                path,
                mmap_mode="r",
            )

            if tokens.ndim != 1:
                raise ValueError(
                    f"Expected 1D token array in {path}, "
                    f"got shape {tokens.shape}"
                )

            shards.append(tokens)

        # ----------------------------------------------------------
        # Calculate shard boundaries
        # ----------------------------------------------------------

        shard_sizes = [
            len(shard)
            for shard in shards
        ]

        shard_offsets = []

        offset = 0

        for size in shard_sizes:
            shard_offsets.append(offset)
            offset += size

        # ----------------------------------------------------------
        # Return immutable dataset metadata/state
        # ----------------------------------------------------------

        return {
            "language": language,
            "split": self.split,
            "split_dir": split_dir,

            "shard_paths": shard_paths,
            "shards": shards,

            "shard_sizes": shard_sizes,
            "shard_offsets": shard_offsets,

            "num_tokens": offset,
            "num_shards": len(shards),
        }

    # ==============================================================
    # Language selection
    # ==============================================================

    def set_language(self, language):
        if language not in self.states:
            raise ValueError(
                f"Unknown language: {language}. "
                f"Available languages: "
                f"{', '.join(self.languages)}"
            )

        self.lang = language

    # ==============================================================
    # Current language state
    # ==============================================================

    @property
    def state(self):
        return self.states[self.lang]

    @property
    def num_tokens(self):
        return self.state["num_tokens"]

    @property
    def num_shards(self):
        return self.state["num_shards"]

    @property
    def shard_paths(self):
        return self.state["shard_paths"]

    @property
    def shards(self):
        return self.state["shards"]

    @property
    def shard_sizes(self):
        return self.state["shard_sizes"]

    @property
    def shard_offsets(self):
        return self.state["shard_offsets"]

    # ==============================================================
    # Dataset access
    # ==============================================================

    def get_slice(self, start, end):
        state = self.state

        if start < 0:
            raise ValueError(
                f"start must be >= 0, got {start}"
            )

        if end > state["num_tokens"]:
            raise ValueError(
                f"end ({end}) exceeds "
                f"{self.lang} dataset size "
                f"({state['num_tokens']})"
            )

        if start >= end:
            raise ValueError(
                f"Invalid slice: "
                f"start={start}, end={end}"
            )

        # ----------------------------------------------------------
        # Find shard containing start
        # ----------------------------------------------------------

        start_shard = self._find_shard(start)

        shard_start = state["shard_offsets"][
            start_shard
        ]

        start_offset = start - shard_start
        end_offset = end - shard_start

        shard = state["shards"][start_shard]

        # ----------------------------------------------------------
        # Fast path:
        # Entire slice is inside one shard.
        # ----------------------------------------------------------

        if end_offset <= len(shard):
            return shard[
                start_offset:end_offset
            ]

        # ----------------------------------------------------------
        # Cross-shard slice
        # ----------------------------------------------------------

        chunks = []

        position = start

        while position < end:
            shard_index = self._find_shard(position)

            shard_start = state["shard_offsets"][
                shard_index
            ]

            shard = state["shards"][
                shard_index
            ]

            local_start = (
                position - shard_start
            )

            remaining_in_shard = (
                len(shard) - local_start
            )

            remaining_to_read = (
                end - position
            )

            count = min(
                remaining_in_shard,
                remaining_to_read,
            )

            chunks.append(
                shard[
                    local_start:
                    local_start + count
                ]
            )

            position += count

        return np.concatenate(chunks)

    # ==============================================================
    # Helpers
    # ==============================================================

    def _find_shard(self, position):
        state = self.state

        for i in range(
            len(state["shard_offsets"]) - 1,
            -1,
            -1,
        ):
            if position >= state["shard_offsets"][i]:
                return i

        raise RuntimeError(
            f"Could not find shard for position "
            f"{position} in {self.lang}"
        )

    # ==============================================================
    # Dataset information
    # ==============================================================

    def get_language_info(self, language=None):
        if language is None:
            language = self.lang

        if language not in self.states:
            raise ValueError(
                f"Unknown language: {language}"
            )

        state = self.states[language]

        return {
            "language": state["language"],
            "split": state["split"],
            "num_shards": state["num_shards"],
            "num_tokens": state["num_tokens"],
        }

    def __len__(self):
        return self.num_tokens
