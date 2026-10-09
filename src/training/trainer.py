"""Complete Diffusion Trainer implementing full fine-tuning loop, validation, and analytics."""
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
import torch

from src.config.schema import AppConfig
from src.data.datamodule import DataModule
from src.evaluation.validator import DiffusionValidator
from src.models.component_loader import verify_model_compatibility
from src.models.pipeline import StableDiffusionFineTuningPipeline
from src.monitoring.experiment_tracker import ExperimentTracker
from src.monitoring.memory_monitor import MemoryMonitor
from src.training.checkpoint_manager import CheckpointManager
from src.training.device_manager import DeviceManager
from src.training.forward_diffusion import ForwardDiffusion
from src.training.loss import DiffusionLoss
from src.training.optimizer import create_optimizer_and_scheduler
from src.training.reproducibility import configure_reproducibility
from src.training.train_step import TrainStepExecutor

logger = logging.getLogger(__name__)


class DiffusionTrainer:
    """Orchestrates Stable Diffusion 1.5 fine-tuning, evaluation intervals, and monitoring."""

    def __init__(
        self,
        config: AppConfig,
        pipeline: StableDiffusionFineTuningPipeline,
        datamodule: DataModule,
        device_manager: DeviceManager,
        experiment_tracker: Optional[ExperimentTracker] = None,
    ):
        self.config = config
        self.pipeline = pipeline
        self.datamodule = datamodule
        self.device_manager = device_manager
        self.device = device_manager.device
        self.tracker = experiment_tracker or ExperimentTracker(config)

        # Move pipeline to device
        self.pipeline.to_device(self.device)

        # Setup loss and forward diffusion
        self.loss_fn = DiffusionLoss(loss_type="mse")
        self.forward_diffusion = ForwardDiffusion(self.pipeline.scheduler)

        # Setup Data
        self.datamodule.setup()
        self.train_loader = self.datamodule.train_dataloader()
        self.val_loader = self.datamodule.val_dataloader()

        # Calculate steps
        self.num_train_samples = len(self.datamodule.train_dataset)
        steps_per_epoch = max(1, len(self.train_loader) // config.training.gradient_accumulation_steps)
        self.total_training_steps = min(
            config.training.max_train_steps,
            config.training.epochs * max(1, len(self.train_loader)),
        )

        # Setup Optimizer and Scheduler
        self.optimizer, self.lr_scheduler = create_optimizer_and_scheduler(
            config=config,
            pipeline=self.pipeline,
            total_training_steps=self.total_training_steps,
        )

        # Setup Train Step Executor
        self.step_executor = TrainStepExecutor(
            config=config,
            pipeline=self.pipeline,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            loss_fn=self.loss_fn,
            forward_diffusion=self.forward_diffusion,
            device=self.device,
        )

        # Setup Validator & Checkpoint Manager
        self.validator = DiffusionValidator(
            config=config,
            pipeline=self.pipeline,
            forward_diffusion=self.forward_diffusion,
            loss_fn=self.loss_fn,
            device=self.device,
        )

        self.checkpoint_manager = CheckpointManager(
            config=config,
            checkpoint_dir=config.project.checkpoint_dir,
            max_to_keep=config.training.max_checkpoints_to_keep,
        )

        # Initial state variables
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self._interrupted = False

        # Register signal handlers for graceful exit
        signal.signal(signal.SIGINT, self._handle_interrupt)

        # Log initial environment and model summaries
        self._log_initial_metadata()

    def _handle_interrupt(self, signum, frame):
        logger.warning("\nInterrupt signal received! Finishing current batch and saving emergency checkpoint...")
        self._interrupted = True

    def _log_initial_metadata(self) -> None:
        """Log environment, model parameter counts, and dataset report."""
        env_info = configure_reproducibility(self.config)
        self.tracker.log_environment(env_info)

        model_report = verify_model_compatibility(
            self.pipeline.tokenizer,
            self.pipeline.text_encoder.text_encoder,
            self.pipeline.vae.vae,
            self.pipeline.unet.unet,
            self.pipeline.scheduler.scheduler,
        )
        self.tracker.log_model_summary(model_report)

        dataset_report = {
            "train_samples": len(self.datamodule.train_dataset),
            "val_samples": len(self.datamodule.val_dataset),
            "test_samples": len(self.datamodule.test_dataset),
            "target_resolution": self.config.dataset.target_resolution,
            "aspect_ratio_policy": self.config.dataset.aspect_ratio_policy,
        }
        self.tracker.log_dataset_report(dataset_report)

    def resume_if_configured(self) -> None:
        """Resume training from checkpoint if configured."""
        resume_target = self.config.training.resume_from_checkpoint
        if not resume_target and not self.config.project.resume:
            return

        target_path = Path(resume_target) if resume_target else Path(self.config.project.checkpoint_dir) / "latest"
        if not target_path.exists():
            # Check for latest step checkpoint in checkpoint_dir
            ckpt_dir = Path(self.config.project.checkpoint_dir)
            ckpts = sorted(ckpt_dir.glob("checkpoint_step_*"))
            if ckpts:
                target_path = ckpts[-1]

        if target_path.exists():
            meta = self.checkpoint_manager.load_checkpoint(
                checkpoint_path=target_path,
                pipeline=self.pipeline,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                scaler=self.step_executor.scaler,
            )
            self.global_step = meta.get("global_step", 0)
            self.current_epoch = meta.get("epoch", 0)
            self.best_val_loss = meta.get("best_val_loss", float("inf"))
            logger.info(f"Resumed training at step {self.global_step}, epoch {self.current_epoch}")

    def train(self) -> str:
        """Execute full training loop."""
        logger.info(
            f"Starting fine-tuning: {self.config.training.epochs} epochs, "
            f"{self.total_training_steps} max steps, batch size {self.config.training.train_batch_size} "
            f"(grad accum {self.config.training.gradient_accumulation_steps})"
        )

        self.resume_if_configured()
        train_cfg = self.config.training
        patience_counter = 0

        try:
            for epoch in range(self.current_epoch, train_cfg.epochs):
                self.current_epoch = epoch
                self.pipeline.train()

                for batch_idx, batch in enumerate(self.train_loader):
                    if self.global_step >= self.total_training_steps or self._interrupted:
                        break

                    collect_diag = (
                        self.global_step % train_cfg.tensor_stats_frequency_steps == 0
                        or self.config.logging.debug_mode
                    )

                    loss_val, metrics, diag_tensors = self.step_executor.execute_step(
                        batch=batch,
                        step_idx=self.global_step,
                        collect_diagnostics=collect_diag,
                    )

                    current_lr = self.optimizer.param_groups[0]["lr"]

                    # 1. Log metrics
                    self.tracker.log_train_step(
                        global_step=self.global_step,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        lr=current_lr,
                        metrics=metrics,
                    )

                    # 2. Log Memory Stats
                    if self.global_step % train_cfg.logging_frequency_steps == 0:
                        mem_stats = MemoryMonitor.get_memory_snapshot(self.device)
                        self.tracker.log_memory_stats(self.global_step, mem_stats)
                        self.device_manager.check_memory_threshold()

                    # 3. Log Tensor Diagnostics
                    if collect_diag and diag_tensors:
                        self.tracker.log_tensor_stats(self.global_step, diag_tensors)

                    # 4. Periodic Validation
                    if self.global_step > 0 and self.global_step % train_cfg.validation_frequency_steps == 0:
                        val_loss, count, elapsed = self.validator.evaluate_loss(self.val_loader)
                        self.tracker.log_validation_step(
                            global_step=self.global_step,
                            epoch=epoch,
                            val_loss=val_loss,
                            sample_count=count,
                            eval_time=elapsed,
                        )

                        # Check best model
                        if val_loss < self.best_val_loss - train_cfg.early_stopping.min_delta:
                            self.best_val_loss = val_loss
                            patience_counter = 0
                            if train_cfg.save_best_checkpoint:
                                self.checkpoint_manager.save_checkpoint(
                                    pipeline=self.pipeline,
                                    optimizer=self.optimizer,
                                    lr_scheduler=self.lr_scheduler,
                                    scaler=self.step_executor.scaler,
                                    global_step=self.global_step,
                                    epoch=epoch,
                                    val_loss=val_loss,
                                    is_best=True,
                                )
                        else:
                            patience_counter += 1

                        # Early stopping check
                        if train_cfg.early_stopping.enabled and patience_counter >= train_cfg.early_stopping.patience:
                            logger.info(f"Early stopping triggered after {patience_counter} validation intervals without improvement.")
                            self._interrupted = True
                            break

                        self.pipeline.train()

                    # 5. Periodic Preview Generation
                    if self.global_step > 0 and self.global_step % train_cfg.image_preview_frequency_steps == 0:
                        logger.info(f"Generating validation previews at step {self.global_step}...")
                        preview_imgs = self.validator.generate_previews(seed=self.config.project.seed + self.global_step)
                        self.tracker.log_previews(
                            images=preview_imgs,
                            prompts=self.config.validation.fixed_prompts[: len(preview_imgs)],
                            step=self.global_step,
                            epoch=epoch,
                        )
                        self.pipeline.train()

                    # 6. Periodic Checkpointing
                    if self.global_step > 0 and self.global_step % train_cfg.checkpoint_frequency_steps == 0:
                        self.checkpoint_manager.save_checkpoint(
                            pipeline=self.pipeline,
                            optimizer=self.optimizer,
                            lr_scheduler=self.lr_scheduler,
                            scaler=self.step_executor.scaler,
                            global_step=self.global_step,
                            epoch=epoch,
                            val_loss=None,
                            is_best=False,
                        )

                    self.global_step += 1

                if self._interrupted or self.global_step >= self.total_training_steps:
                    break

                # Save Epoch Checkpoint
                if (epoch + 1) % train_cfg.checkpoint_frequency_epochs == 0:
                    self.checkpoint_manager.save_checkpoint(
                        pipeline=self.pipeline,
                        optimizer=self.optimizer,
                        lr_scheduler=self.lr_scheduler,
                        scaler=self.step_executor.scaler,
                        global_step=self.global_step,
                        epoch=epoch + 1,
                        val_loss=None,
                        is_best=False,
                    )

        except Exception as e:
            logger.error(f"Training encountered an unhandled exception: {e}", exc_info=True)
            raise e
        finally:
            # Save final checkpoint
            logger.info("Saving final fine-tuned model checkpoint...")
            self.checkpoint_manager.save_checkpoint(
                pipeline=self.pipeline,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                scaler=self.step_executor.scaler,
                global_step=self.global_step,
                epoch=self.current_epoch,
                val_loss=self.best_val_loss,
                is_final=True,
            )

            status = "INTERRUPTED" if self._interrupted else "COMPLETED"
            report_path = self.tracker.generate_diagnostic_report(
                final_step=self.global_step,
                final_epoch=self.current_epoch,
                best_val_loss=self.best_val_loss,
                status=status,
            )

        return report_path
