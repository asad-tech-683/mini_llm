import torch

from model.gpt import GPT


class ModelLoader:
    def __init__(self, config, *, device=None):
        self.config = config
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def load_final(self, path):
        """Load a final model saved for inference."""

        state_dict = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        return self._build_model(state_dict)

    def load_checkpoint(self, path):
        """Load model weights from a training checkpoint."""

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        return self._build_model(
            checkpoint["model_state_dict"]
        )

    def _build_model(self, state_dict):
        model = GPT(self.config)

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model