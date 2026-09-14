import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from evaluation.metrics import EvaluationMetrics
from generation.generator import Generator
from pathlib import Path


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        scheduler,
        config,
        distributed,
        metrics,
        evaluator,
        checkpoint_manager
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.scheduler = scheduler
        self.config = config
        self.distributed = distributed
        self.metrics = metrics
        self.evaluator = evaluator
        self.checkpoint_manager = checkpoint_manager

        self.generation_directory = Path("generation")
        self.generation_directory.mkdir(parents=True, exist_ok=True)
        
        # The model must already be on the correct device.
        if distributed.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[distributed.local_rank],
            )

        self.raw_model = (
            self.model.module
            if distributed.is_distributed
            else self.model
        )
        
        self.generator = generator = Generator(
            model=self.raw_model
        )

    def train(self, start_step = 0):
        self.model.train()
        eval_metrics = EvaluationMetrics()
        for step in range(start_step, self.config.train.max_steps):

            # ------------------------------------------------------
            # Select language for THIS optimizer step
            # ------------------------------------------------------
            language = 'urdu' if step % 2 == 0 else 'hindi'
            # language = 'urdu'

            self.train_loader.set_language(language)
            # ------------------------------------------------------
            # Start metrics timing
            # ------------------------------------------------------

            self.metrics.start_step()

            # ------------------------------------------------------
            # Learning rate
            # ------------------------------------------------------

            lr = self.scheduler.get_lr(step)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # ------------------------------------------------------
            # Gradient accumulation
            # ------------------------------------------------------

            self.optimizer.zero_grad(set_to_none=True)

            loss_accum = 0.0

            for micro_step in range(
                self.config.train.grad_accumulation_steps
            ):
                x, y = self.train_loader.next_batch()

                x = x.to(self.distributed.device)
                y = y.to(self.distributed.device)

                # During DDP gradient accumulation, only the final
                # micro-step needs gradient synchronization.
                if (
                    self.distributed.is_distributed
                    and micro_step
                    < self.config.train.grad_accumulation_steps - 1
                ):
                    context = self.model.no_sync()
                else:
                    context = torch.enable_grad()

                with context:
                    logits, loss = self.model(
                        x,
                        targets=y,
                    )

                    loss = (
                        loss
                        / self.config.train.grad_accumulation_steps
                    )

                    loss_accum += loss.detach()

                    loss.backward()

            # ------------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------------

            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.train.grad_clip,
            )

            # ------------------------------------------------------
            # Optimizer step
            # ------------------------------------------------------

            self.optimizer.step()

            # ------------------------------------------------------
            # Reduce loss across GPUs
            # ------------------------------------------------------

            if self.distributed.is_distributed:
                torch.distributed.all_reduce(
                    loss_accum,
                    op=torch.distributed.ReduceOp.AVG,
                )

            loss_value = loss_accum.item()

            # ------------------------------------------------------
            # Update metrics
            # ------------------------------------------------------

            self.metrics.end_step(
                language=language,
                loss=loss_value,
                learning_rate=lr,
                grad_norm=norm.item(),
            )

            # ------------------------------------------------------
            # Logging
            # ------------------------------------------------------

            if (
                self.distributed.master_process
                and step % self.config.train.log_interval == 0
            ):
                self.metrics.report_step()
                
            if (step + 1) % self.config.train.eval_interval == 0:
                # put in eval mode
                self.model.eval()
                
                with torch.inference_mode():
                    results = self.evaluator.evaluate()

                if self.distributed.master_process:
                    eval_metrics.results = results
                    eval_metrics.report(
                        step=step + 1,
                    )
                    self.generate_samples(step=step + 1)
                        
                # Wait for master to finish generation before
                # any rank starts the next training step.
                if self.distributed.is_distributed:
                    torch.distributed.barrier()
                    
                # back to train mode        
                self.model.train()
            
            # Checkpoint
            if (step + 1) % self.config.train.checkpoint_interval == 0 and self.distributed.master_process:
                self.checkpoint_manager.save(
                    step=step + 1,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    config=self.config,
                    train_loader=self.train_loader,
                    metrics=self.metrics,
                )
                
                # Make sure every rank waits until the checkpoint
                # has been completely written.
                if self.distributed.is_distributed:
                    torch.distributed.barrier()
                    
        # ==========================================================
        # TRAINING FINISHED
        # ==========================================================

        if self.distributed.master_process:
            path = self.checkpoint_manager.save_final_model(
                model=self.model,
                filename="final_model.pt",
            )

            print(f"Final model saved to: {path}")

        # Make sure all ranks wait for final model to finish saving
        if self.distributed.is_distributed:
            torch.distributed.barrier()
            
           
    def generate_samples(self, step):
        urdu_prompt = "ایک زمانے کی بات ہے"
        hindi_prompt = "एक ज़माने की बात है۔"

        temperature = 0.8
        top_k = 50
        max_new_tokens = 255
        num_generations = 5

        with torch.inference_mode():
            urdu_generations = self.generator.generate_many(
                prompt=urdu_prompt,
                num_generations=num_generations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            hindi_generations = self.generator.generate_many(
                prompt=hindi_prompt,
                num_generations=num_generations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        for i, (urdu, hindi) in enumerate(
            zip(urdu_generations, hindi_generations), 1
        ):
            print(f"\n--- Generation {i} ---")
            print(f"Urdu:  {urdu}")
            print(f"Hindi: {hindi}")

        self.save_generations(
            step=step,
            urdu_prompt=urdu_prompt,
            urdu_generations=urdu_generations,
            hindi_prompt=hindi_prompt,
            hindi_generations=hindi_generations,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )
    
     
    def save_generations(
        self,
        *,
        step,
        urdu_prompt,
        urdu_generations,
        hindi_prompt,
        hindi_generations,
        temperature,
        top_k,
        max_new_tokens,
    ):
        path = self.generation_directory / "generations.txt"

        with path.open("a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"STEP {step}\n")
            f.write("=" * 80 + "\n\n")

            f.write("Generation settings:\n")
            f.write(f"Temperature: {temperature}\n")
            f.write(f"Top-k: {top_k}\n")
            f.write(f"Max new tokens: {max_new_tokens}\n\n")

            f.write(f"URDU PROMPT:\n{urdu_prompt}\n\n")

            for i, generation in enumerate(urdu_generations, 1):
                f.write(f"--- Urdu Generation {i} ---\n")
                f.write(generation)
                f.write("\n\n")

            f.write(f"HINDI PROMPT:\n{hindi_prompt}\n\n")

            for i, generation in enumerate(hindi_generations, 1):
                f.write(f"--- Hindi Generation {i} ---\n")
                f.write(generation)
                f.write("\n\n")

            f.write("\n")