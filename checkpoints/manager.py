"""
Checkpoint utilities for resumable training.

Keeping this logic in one class keeps the training loop from having to know
which pieces need to be saved, restored, or unwrapped.
"""

from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, *, directory="checkpoints"):
        self.directory = Path(directory)

        # Make sure the directory exists, including any parent folders.
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _raw_model(model):
        # DDP/DataParallel wraps the real model under `.module`.
        # Unwrapping keeps the saved state_dict keys clean and portable.
        return getattr(model, "module", model)

    def save(
        self,
        *,
        step,
        model,
        optimizer,
        scheduler=None,
        config=None,
        train_loader=None,
        language_sampler=None,
        metrics=None,
        filename=None,
    ):
        """Save a complete resumable training checkpoint."""

        # Default to a zero-padded step number so checkpoints sort nicely.
        path = self.directory / (
            filename or f"checkpoint_step_{step:06d}.pt"
        )

        checkpoint = {
            "step": step,
            "model_state_dict": self._raw_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if train_loader is not None:
            # Preserve where we were in the data stream so training can resume.
            checkpoint["train_loader_state"] = {
                "language": train_loader.lang,
                "positions": train_loader.get_positions(),
            }

        torch.save(checkpoint, path)
        return path

    def load(
        self,
        *,
        path,
        model,
        optimizer=None,
        scheduler=None,
        train_loader=None,
        language_sampler=None,
        metrics=None,
        device=None,
    ):
        """Load a complete training checkpoint."""

        # weights_only=False is intentional here: checkpoints can contain
        # other Python objects, not just tensors.
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

        # The model is the one piece every checkpoint must have.
        self._raw_model(model).load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        loader_state = checkpoint.get("train_loader_state")

        if train_loader is not None and loader_state is not None:
            # Put the loader back in the same language and position.
            train_loader.set_language(
                loader_state["language"]
            )

            # The loader only exposes get_positions(), so restore the internal
            # positions mapping directly here.
            train_loader.positions = dict(
                loader_state["positions"]
            )

        return checkpoint

    def save_final_model(
        self,
        *,
        model,
        filename="final_model.pt",
    ):
        """Save model weights for inference."""

        path = self.directory / filename

        # Inference-only export: no optimizer, scheduler, or training metadata.
        torch.save(
            self._raw_model(model).state_dict(),
            path,
        )

        return path