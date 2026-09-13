from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, *, directory="checkpoints"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _raw_model(model):
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

        path = self.directory / (
            filename or f"checkpoint_step_{step:06d}.pt"
        )

        checkpoint = {
            "step": step,
            "model_state_dict": self._raw_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if scheduler is not None:
            checkpoint["scheduler_state"] = (
                scheduler.state_dict()
                if hasattr(scheduler, "state_dict")
                else None
            )

        if config is not None:
            checkpoint["config"] = vars(config)

        if train_loader is not None:
            checkpoint["train_loader_state"] = {
                "language": train_loader.lang,
                "positions": train_loader.get_positions(),
            }

        for key, obj in (
            ("language_sampler_state", language_sampler),
            ("metrics_state", metrics),
        ):
            if obj is not None and hasattr(obj, "state_dict"):
                checkpoint[key] = obj.state_dict()

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

        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

        self._raw_model(model).load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None
            and checkpoint.get("scheduler_state") is not None
            and hasattr(scheduler, "load_state_dict")
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler_state"]
            )

        loader_state = checkpoint.get("train_loader_state")

        if train_loader is not None and loader_state is not None:
            train_loader.set_language(
                loader_state["language"]
            )
            train_loader.positions = dict(
                loader_state["positions"]
            )

        for key, obj in (
            ("language_sampler_state", language_sampler),
            ("metrics_state", metrics),
        ):
            state = checkpoint.get(key)

            if (
                obj is not None
                and state is not None
                and hasattr(obj, "load_state_dict")
            ):
                obj.load_state_dict(state)

        return checkpoint

    def save_final_model(
        self,
        *,
        model,
        filename="final_model.pt",
    ):
        """Save model weights for inference."""

        path = self.directory / filename

        torch.save(
            self._raw_model(model).state_dict(),
            path,
        )

        return path
