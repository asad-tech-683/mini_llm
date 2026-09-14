import torch

from config import Config
from model.gpt import GPT

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
    # --------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------

    distributed = DistributedContext()
    config = Config()

    print(f"Device: {distributed.device}")
    print(f"Distributed: {distributed.is_distributed}")
    print(f"World size: {distributed.world_size}")

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------


    model = GPT(config)
    model.to(distributed.device)
    
    model = distributed.wrap_model(model)
    raw_model = distributed.unwrap_model(model)
    
    # model_visualizer = ModelVisualizer(model=raw_model)
    # model_visualizer.summary()
    # import sys; sys.exit(0)
    # --------------------------------------------------------------
    # Data
    # --------------------------------------------------------------

    train_loader = DataLoaderLite(
        B=config.train.batch_size,
        T=config.sequence_length,
        process_rank=distributed.rank,
        num_processes=distributed.world_size,
        split="train",
        data_dir=config.tokenizer.data_path
    )

    print("DataLoader created successfully.")

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------
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
    # --------------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------------

    optimizer = configure_optimizer(
        model=model,
        weight_decay=config.train.weight_decay,
        learning_rate=config.train.max_lr,
        device_type=distributed.device.type,
    )

    print("Optimizer created successfully.")

    # --------------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------------

    scheduler = CosineScheduler(
        max_lr=config.train.max_lr,
        min_lr=config.train.min_lr,
        warmup_steps=config.train.warmup_steps,
        max_steps=config.train.max_steps,
    )

    print("Scheduler created successfully.")

    # --------------------------------------------------------------
    # Evaluator
    # --------------------------------------------------------------
    evaluator = Evaluator(
        model=model,
        batch_size=config.train.batch_size,
        sequence_length=config.sequence_length,
        distributed=distributed,
        languages=("hindi", "urdu"),
        max_batches=20,
        data_dir=config.tokenizer.data_path
    )
    
    # checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # # ----------------------------------------------------------
    # # Resume
    # # ----------------------------------------------------------

    # start_step = 0

    # checkpoint_path = "checkpoints/checkpoint_step_004000.pt"

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
    #     print(
    #         f"Resuming training from step {start_step}"
    #     )

    # --------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        distributed=distributed,
        config=config,
        evaluator=evaluator,
        checkpoint_manager=checkpoint_manager
    )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    trainer.train()

    # --------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------

    distributed.cleanup()
    
if __name__ == "__main__":
    main()