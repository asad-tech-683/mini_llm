import torch

from config import Config
from model.model import Model

from data.dataloader import DataLoaderLite

from training.distributed import DistributedContext
from training.optimizer import configure_optimizer
from training.scheduler import CosineScheduler
from training.metrics import TrainingMetrics
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from checkpoints.manager import CheckpointManager
from inspect_model import ModelVisualizer


def main():
    # Set up distributed training context and load experiment config.
    distributed = DistributedContext()
    config = Config()

    print(f"Device: {distributed.device}")
    print(f"Distributed: {distributed.is_distributed}")
    print(f"World size: {distributed.world_size}")

    # Build the model and move it to the right device. If we're running
    # distributed, wrap it in DDP and keep a handle to the raw model.
    model = Model(config)
    model.to(distributed.device)

    model = distributed.wrap_model(model)
    raw_model = distributed.unwrap_model(model)

    # Uncomment to print a model summary and exit.
    # model_visualizer = ModelVisualizer(model=raw_model)
    # model_visualizer.summary()
    # import sys; sys.exit(0)

    # Streaming dataloader that alternates between languages during training.
    train_loader = DataLoaderLite(
        B=config.train.batch_size,
        T=config.sequence_length,
        process_rank=distributed.rank,
        num_processes=distributed.world_size,
        split="train",
        data_dir=config.tokenizer.data_path,
    )

    print("DataLoader created successfully.")

    # Metrics tracker for loss, LR, grad norm, and tokens seen per language.
    metrics = TrainingMetrics(
        total_tokens_by_language=train_loader.num_tokens,
        batch_size=config.train.batch_size,
        sequence_length=config.sequence_length,
        world_size=distributed.world_size,
        grad_accumulation_steps=config.train.grad_accumulation_steps,
        max_steps=config.train.max_steps,
    )

    if distributed.master_process:
        metrics.summary()

    # AdamW with weight decay applied only to the right parameter groups.
    optimizer = configure_optimizer(
        model=model,
        weight_decay=config.train.weight_decay,
        learning_rate=config.train.max_lr,
        device_type=distributed.device.type,
    )

    print("Optimizer created successfully.")

    # Cosine schedule with linear warmup.
    scheduler = CosineScheduler(
        max_lr=config.train.max_lr,
        min_lr=config.train.min_lr,
        warmup_steps=config.train.warmup_steps,
        max_steps=config.train.max_steps,
    )

    print("Scheduler created successfully.")

    # Evaluator runs on a fixed number of batches per language.
    evaluator = Evaluator(
        model=model,
        batch_size=config.train.batch_size,
        sequence_length=config.sequence_length,
        distributed=distributed,
        languages=("hindi", "urdu"),
        max_batches=20,
        data_dir=config.tokenizer.data_path,
    )

    checkpoint_manager = CheckpointManager()

    # Resume from a checkpoint if one is provided. This restores model
    # weights, optimizer state, scheduler state, and dataloader position.
    start_step = 0
    # checkpoint_path = "D:\\AI-ML\\multi_lang_variants\\32m\\2.0\\checkpoint_step_009000.pt"

    # if checkpoint_path:
    #     checkpoint = checkpoint_manager.load(
    #         path=checkpoint_path,
    #         model=model,
    #         optimizer=optimizer,
    #         scheduler=scheduler,
    #         train_loader=train_loader,
    #         metrics=metrics,
    #         device=distributed.device,
    #     )

    #     start_step = checkpoint["step"]

    # if distributed.master_process:
    #     print(f"Resuming training from step {start_step}")

    # The trainer handles the main loop, evaluation, generation, and
    # periodic checkpointing.
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        distributed=distributed,
        config=config,
        evaluator=evaluator,
        checkpoint_manager=checkpoint_manager,
    )

    trainer.train(start_step=start_step)

    distributed.cleanup()


if __name__ == "__main__":
    main()