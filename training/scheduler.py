import math


class CosineScheduler:
    def __init__(
        self,
        max_lr,
        min_lr,
        warmup_steps,
        max_steps,
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, step):
        # 1. Linear warmup
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps

        # 2. After training has finished
        if step > self.max_steps:
            return self.min_lr

        # 3. Progress through cosine decay
        decay_ratio = (
            (step - self.warmup_steps)
            / (self.max_steps - self.warmup_steps)
        )

        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )

        return self.min_lr + coeff * (
            self.max_lr - self.min_lr
        )