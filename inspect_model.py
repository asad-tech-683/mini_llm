from pathlib import Path

import torch


class ModelVisualizer:
    def __init__(
        self,
        model,
        *,
        output_dir="visualizations",
    ):
        self.model = model
        self.output_dir = Path(output_dir)

    def summary(self):
        total = 0
        trainable = 0
        parameter_bytes = 0

        print()
        print("=" * 96)
        print("MODEL")
        print("=" * 96)

        for name, parameter in self.model.named_parameters():
            count = parameter.numel()
            total += count

            if parameter.requires_grad:
                trainable += count

            parameter_bytes += count * parameter.element_size()

            print(
                f"  {name:<50}"
                f" {str(tuple(parameter.shape)):<18}"
                f" {count:>14,}"
            )

        non_trainable = total - trainable

        print("-" * 96)
        print(f"  {'Total parameters':<50} {total:>14,}")
        print(f"  {'Trainable parameters':<50} {trainable:>14,}")
        print(f"  {'Non-trainable parameters':<50} {non_trainable:>14,}")
        print(
            f"  {'Parameter memory':<50}"
            f" {parameter_bytes / 1024**2:>13.2f} MB"
        )
        print("=" * 96)
        print()

    def module_summary(self):
        totals = {}

        for name, parameter in self.model.named_parameters():
            parts = name.split(".")

            if len(parts) == 1:
                module = "root"
            else:
                module = ".".join(parts[:-1])

            totals[module] = totals.get(module, 0) + parameter.numel()

        print()
        print("=" * 72)
        print("MODULE PARAMETERS")
        print("=" * 72)

        for module, count in totals.items():
            print(f"  {module:<45} {count:>14,}")

        print("-" * 72)

        total = sum(totals.values())

        print(f"  {'TOTAL':<45} {total:>14,}")
        print("=" * 72)
        print()

    def run(self):
        self.summary()
        self.module_summary()