"""Structured experiment tracking, multi-format metrics persistence, and summary report generation."""
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

from src.config.loader import save_config_snapshot
from src.config.schema import AppConfig
from src.monitoring.image_logger import ImageLogger
from src.monitoring.logger import setup_experiment_logger
from src.monitoring.tensor_stats import TensorStatsTracker
from src.utils.paths import get_timestamp_str

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Orchestrates structured logging, TensorBoard, CSV/JSONL records, and diagnostic summaries."""

    def __init__(self, config: AppConfig, timestamp: Optional[str] = None):
        self.config = config
        self.timestamp = timestamp or get_timestamp_str()
        self.exp_name = f"experiment_{config.project.experiment_name}_{self.timestamp}"

        # Initialize unique experiment directory under logs/
        self.exp_dir = Path(config.project.log_dir) / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Subfolders
        (self.exp_dir / "sample_predictions").mkdir(exist_ok=True)
        (self.exp_dir / "generated_previews").mkdir(exist_ok=True)

        # File paths
        self.log_file = self.exp_dir / "training.log"
        self.train_csv_file = self.exp_dir / "training_metrics.csv"
        self.train_jsonl_file = self.exp_dir / "training_metrics.jsonl"
        self.val_csv_file = self.exp_dir / "validation_metrics.csv"
        self.mem_csv_file = self.exp_dir / "memory_metrics.csv"
        self.tensor_stats_csv_file = self.exp_dir / "tensor_statistics.csv"
        self.diagnostic_report_file = self.exp_dir / "diagnostic_report.md"

        # Setup custom file logger
        self.logger = setup_experiment_logger(
            name="experiment",
            log_file=self.log_file,
            console=config.logging.console,
        )

        # Save configuration snapshot immediately
        save_config_snapshot(config, self.exp_dir / "config_snapshot.yaml")

        # Initialize ImageLogger
        self.image_logger = ImageLogger(self.exp_dir)

        # TensorBoard SummaryWriter
        self.tb_writer = None
        if config.logging.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.exp_dir / "tensorboard"))
            except Exception as e:
                self.logger.warning(f"Failed to initialize TensorBoard SummaryWriter: {e}")

        # In-memory tracking for moving average
        self.loss_history: List[float] = []
        self._init_csv_headers()

    def _init_csv_headers(self) -> None:
        """Initialize CSV files with header lines."""
        if self.config.logging.csv:
            with open(self.train_csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "global_step", "epoch", "batch_idx", "learning_rate",
                    "loss", "loss_ema", "grad_norm", "param_norm",
                    "fwd_time_sec", "bwd_time_sec", "opt_time_sec", "step_time_sec"
                ])

            with open(self.val_csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["global_step", "epoch", "val_loss", "sample_count", "eval_time_sec"])

            with open(self.mem_csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["global_step", "cuda_allocated_mb", "cuda_reserved_mb", "cuda_max_allocated_mb"])

            with open(self.tensor_stats_csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "global_step", "tensor_name", "shape", "dtype", "device",
                    "num_elements", "nan_count", "inf_count", "min", "max", "mean", "std", "q25", "median_q50", "q75"
                ])

    def log_environment(self, env_info: Dict[str, Any]) -> None:
        """Save environment info JSON."""
        with open(self.exp_dir / "environment.json", "w", encoding="utf-8") as f:
            json.dump(env_info, f, indent=2)

    def log_model_summary(self, model_summary: Dict[str, Any]) -> None:
        """Save model architecture and parameter count summary JSON."""
        with open(self.exp_dir / "model_summary.json", "w", encoding="utf-8") as f:
            json.dump(model_summary, f, indent=2)

    def log_dataset_report(self, dataset_report: Dict[str, Any]) -> None:
        """Save dataset split counts and preprocessing summary JSON."""
        with open(self.exp_dir / "dataset_report.json", "w", encoding="utf-8") as f:
            json.dump(dataset_report, f, indent=2)

    def log_train_step(
        self,
        global_step: int,
        epoch: int,
        batch_idx: int,
        lr: float,
        metrics: Dict[str, Any],
    ) -> None:
        """Log a training step across console, CSV, JSONL, and TensorBoard."""
        loss = metrics["loss"]
        self.loss_history.append(loss)
        # Exponential moving average over last 20 steps
        ema_window = self.loss_history[-20:]
        loss_ema = sum(ema_window) / len(ema_window)

        grad_norm = metrics.get("grad_norm", 0.0)
        param_norm = metrics.get("param_norm", 0.0)
        fwd_time = metrics.get("fwd_time", 0.0)
        bwd_time = metrics.get("bwd_time", 0.0)
        opt_time = metrics.get("opt_time", 0.0)
        step_time = metrics.get("step_time", 0.0)

        # CSV
        if self.config.logging.csv:
            with open(self.train_csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    global_step, epoch, batch_idx, f"{lr:.6e}",
                    f"{loss:.6f}", f"{loss_ema:.6f}", f"{grad_norm:.6f}", f"{param_norm:.6f}",
                    f"{fwd_time:.4f}", f"{bwd_time:.4f}", f"{opt_time:.4f}", f"{step_time:.4f}"
                ])

        # JSONL
        if self.config.logging.jsonl:
            record = {
                "global_step": global_step,
                "epoch": epoch,
                "batch_idx": batch_idx,
                "learning_rate": lr,
                "loss": loss,
                "loss_ema": loss_ema,
                "grad_norm": grad_norm,
                "param_norm": param_norm,
                "fwd_time": fwd_time,
                "bwd_time": bwd_time,
                "opt_time": opt_time,
                "step_time": step_time,
            }
            with open(self.train_jsonl_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        # TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Train/Loss", loss, global_step)
            self.tb_writer.add_scalar("Train/Loss_EMA", loss_ema, global_step)
            self.tb_writer.add_scalar("Train/LearningRate", lr, global_step)
            self.tb_writer.add_scalar("Train/GradNorm", grad_norm, global_step)
            self.tb_writer.add_scalar("Train/StepTimeSec", step_time, global_step)

        # Console
        if global_step % self.config.training.logging_frequency_steps == 0:
            self.logger.info(
                f"Epoch {epoch:02d} | Step {global_step:05d} | "
                f"Loss: {loss:.4f} (EMA: {loss_ema:.4f}) | "
                f"GradNorm: {grad_norm:.3f} | LR: {lr:.2e} | "
                f"StepTime: {step_time*1000:.1f}ms"
            )

    def log_validation_step(
        self,
        global_step: int,
        epoch: int,
        val_loss: float,
        sample_count: int,
        eval_time: float,
    ) -> None:
        """Log validation results."""
        if self.config.logging.csv:
            with open(self.val_csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([global_step, epoch, f"{val_loss:.6f}", sample_count, f"{eval_time:.4f}"])

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Val/DiffusionLoss", val_loss, global_step)

        self.logger.info(
            f"--- VALIDATION (Step {global_step:05d}, Epoch {epoch:02d}) --- "
            f"Val Loss: {val_loss:.4f} ({sample_count} samples evaluated in {eval_time:.2f}s)"
        )

    def log_memory_stats(self, global_step: int, mem_stats: Dict[str, Any]) -> None:
        """Log memory metrics."""
        if self.config.logging.csv:
            with open(self.mem_csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    global_step,
                    mem_stats.get("cuda_allocated_mb", 0.0),
                    mem_stats.get("cuda_reserved_mb", 0.0),
                    mem_stats.get("cuda_max_allocated_mb", 0.0),
                ])

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Memory/CUDA_Allocated_MB", mem_stats.get("cuda_allocated_mb", 0.0), global_step)
            self.tb_writer.add_scalar("Memory/CUDA_Reserved_MB", mem_stats.get("cuda_reserved_mb", 0.0), global_step)

    def log_tensor_stats(self, global_step: int, diag_tensors: Dict[str, Any]) -> None:
        """Log numerical statistics for all collected intermediate tensors."""
        if not self.config.logging.csv or not diag_tensors:
            return

        with open(self.tensor_stats_csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for name, tensor in diag_tensors.items():
                stats = TensorStatsTracker.compute_stats(name, tensor)
                writer.writerow([
                    global_step,
                    stats.get("tensor_name"),
                    stats.get("shape"),
                    stats.get("dtype"),
                    stats.get("device"),
                    stats.get("num_elements"),
                    stats.get("nan_count"),
                    stats.get("inf_count"),
                    stats.get("min"),
                    stats.get("max"),
                    stats.get("mean"),
                    stats.get("std"),
                    stats.get("q25"),
                    stats.get("median_q50"),
                    stats.get("q75"),
                ])

    def log_previews(self, images: List[Image.Image], prompts: List[str], step: int, epoch: int) -> None:
        """Save preview images and optionally log to TensorBoard."""
        saved_paths = self.image_logger.log_previews(images, prompts, step, epoch)
        if self.tb_writer is not None and images:
            import numpy as np
            import torch
            for idx, (img, prompt) in enumerate(zip(images, prompts)):
                img_arr = np.array(img).transpose(2, 0, 1) # (C, H, W)
                self.tb_writer.add_image(f"Previews/Sample_{idx:02d}", torch.from_numpy(img_arr), step)

    def generate_diagnostic_report(
        self,
        final_step: int,
        final_epoch: int,
        best_val_loss: float,
        status: str = "COMPLETED",
    ) -> str:
        """Generate a complete Markdown diagnostic report of the run."""
        avg_loss = sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0.0
        final_loss = self.loss_history[-1] if self.loss_history else 0.0

        report = f"""# Experiment Diagnostic & Training Summary Report

**Experiment Name**: `{self.config.project.experiment_name}`  
**Run Directory**: `{self.exp_dir.resolve()}`  
**Status**: `{status}`  
**Timestamp**: `{self.timestamp}`  

---

## 1. Executive Summary
- **Total Epochs**: {final_epoch}
- **Total Global Steps**: {final_step}
- **Final Training Loss**: {final_loss:.6f}
- **Average Training Loss**: {avg_loss:.6f}
- **Best Validation Loss**: {best_val_loss:.6f}

---

## 2. Configuration Highlights
- **Model**: `{self.config.model.model_identifier}`
- **Trainable Components**: `{self.config.fine_tuning.trainable_components}`
- **Mixed Precision**: `{self.config.training.mixed_precision}`
- **Batch Size (per step)**: `{self.config.training.train_batch_size}` (Gradient Accumulation: `{self.config.training.gradient_accumulation_steps}`)
- **Effective Batch Size**: `{self.config.training.train_batch_size * self.config.training.gradient_accumulation_steps}`
- **Learning Rate**: `{self.config.training.learning_rate}` ({self.config.training.lr_scheduler} scheduler)

---

## 3. Metric Artifacts Generated
- `training_metrics.csv` & `training_metrics.jsonl`: Step-by-step losses and timings.
- `validation_metrics.csv`: Periodic validation loss logs.
- `memory_metrics.csv`: VRAM and allocation tracking.
- `tensor_statistics.csv`: Intermediate tensor distribution summaries (mean, std, quantiles, min/max).
- `generated_previews/`: Image preview snapshots across checkpoints.
- `config_snapshot.yaml`: Complete exact reproduction config.
- `environment.json`: Hardware, PyTorch, and CUDA environment metadata.
"""
        with open(self.diagnostic_report_file, "w", encoding="utf-8") as f:
            f.write(report)

        if self.tb_writer is not None:
            self.tb_writer.close()

        self.logger.info(f"Generated diagnostic summary report at {self.diagnostic_report_file}")
        return str(self.diagnostic_report_file)
