"""Resumable checkpoint manager with state persistence, pruning, and deployable export."""
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import yaml

from src.config.schema import AppConfig
from src.models.pipeline import StableDiffusionFineTuningPipeline

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving, restoring, and pruning training checkpoints and deployable pipelines."""

    def __init__(
        self,
        config: AppConfig,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        max_to_keep: int = 3,
    ):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.best_val_loss = float("inf")
        self.saved_checkpoints: List[Path] = []

    def save_checkpoint(
        self,
        pipeline: StableDiffusionFineTuningPipeline,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        scaler: Optional[torch.amp.GradScaler],
        global_step: int,
        epoch: int,
        val_loss: Optional[float] = None,
        is_best: bool = False,
        is_final: bool = False,
    ) -> Path:
        """Save a complete, resumable checkpoint directory."""
        if is_final:
            ckpt_name = "final_checkpoint"
        elif is_best:
            ckpt_name = "best_checkpoint"
        else:
            ckpt_name = f"checkpoint_step_{global_step:06d}"

        save_path = self.checkpoint_dir / ckpt_name
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. Save trainable model weights
        # Always save U-Net
        unet_state = pipeline.unet.unet.state_dict()
        torch.save(unet_state, save_path / "unet.pt")

        if self.config.fine_tuning.train_text_encoder:
            te_state = pipeline.text_encoder.text_encoder.state_dict()
            torch.save(te_state, save_path / "text_encoder.pt")

        if self.config.fine_tuning.train_vae:
            vae_state = pipeline.vae.vae.state_dict()
            torch.save(vae_state, save_path / "vae.pt")

        # 2. Save Optimizer & Scheduler
        torch.save(optimizer.state_dict(), save_path / "optimizer.pt")
        if lr_scheduler is not None:
            torch.save(lr_scheduler.state_dict(), save_path / "scheduler.pt")

        # 3. Save GradScaler
        if scaler is not None:
            torch.save(scaler.state_dict(), save_path / "scaler.pt")

        # 4. Save RNG States for full reproducibility
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(rng_state, save_path / "rng_state.pt")

        # 5. Save Metadata
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        metadata = {
            "global_step": global_step,
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "experiment_name": self.config.project.experiment_name,
        }
        with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 6. Save Config Snapshot
        from src.config.loader import save_config_snapshot
        save_config_snapshot(self.config, save_path / "config_snapshot.yaml")

        logger.info(f"Saved checkpoint to {save_path} (step {global_step}, epoch {epoch})")

        # Also save deployable pipeline
        pipeline_export_dir = save_path / "pipeline_export"
        try:
            pipeline.save_pretrained(pipeline_export_dir)
        except Exception as e:
            logger.warning(f"Could not export full diffusers pipeline to {pipeline_export_dir}: {e}")

        # Handle pruning if it is a regular step checkpoint
        if not is_best and not is_final:
            self.saved_checkpoints.append(save_path)
            self._prune_old_checkpoints()

        return save_path

    def _prune_old_checkpoints(self) -> None:
        """Prune oldest checkpoints when count exceeds max_to_keep, protecting best & final."""
        if len(self.saved_checkpoints) > self.max_to_keep:
            to_remove = self.saved_checkpoints[: -self.max_to_keep]
            self.saved_checkpoints = self.saved_checkpoints[-self.max_to_keep :]
            for path in to_remove:
                if path.exists():
                    logger.info(f"Pruning older checkpoint: {path}")
                    shutil.rmtree(path, ignore_errors=True)

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        pipeline: StableDiffusionFineTuningPipeline,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> Dict[str, Any]:
        """Restore all training and model states from a checkpoint directory."""
        ckpt_dir = Path(checkpoint_path)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        logger.info(f"Loading checkpoint state from {ckpt_dir}...")

        # 1. Load U-Net weights
        unet_path = ckpt_dir / "unet.pt"
        if unet_path.exists():
            state_dict = torch.load(unet_path, map_location="cpu", weights_only=False)
            pipeline.unet.unet.load_state_dict(state_dict)
            logger.info("Restored U-Net weights from checkpoint.")

        # 2. Load Text Encoder weights if present
        te_path = ckpt_dir / "text_encoder.pt"
        if te_path.exists() and self.config.fine_tuning.train_text_encoder:
            state_dict = torch.load(te_path, map_location="cpu", weights_only=False)
            pipeline.text_encoder.text_encoder.load_state_dict(state_dict)
            logger.info("Restored Text Encoder weights from checkpoint.")

        # 3. Load VAE weights if present
        vae_path = ckpt_dir / "vae.pt"
        if vae_path.exists() and self.config.fine_tuning.train_vae:
            state_dict = torch.load(vae_path, map_location="cpu", weights_only=False)
            pipeline.vae.vae.load_state_dict(state_dict)
            logger.info("Restored VAE weights from checkpoint.")

        # 4. Restore Optimizer & Scheduler
        if optimizer is not None and (ckpt_dir / "optimizer.pt").exists():
            opt_state = torch.load(ckpt_dir / "optimizer.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(opt_state)
            logger.info("Restored Optimizer state.")

        if lr_scheduler is not None and (ckpt_dir / "scheduler.pt").exists():
            sched_state = torch.load(ckpt_dir / "scheduler.pt", map_location="cpu", weights_only=False)
            lr_scheduler.load_state_dict(sched_state)
            logger.info("Restored LR Scheduler state.")

        # 5. Restore GradScaler
        if scaler is not None and (ckpt_dir / "scaler.pt").exists():
            scaler_state = torch.load(ckpt_dir / "scaler.pt", map_location="cpu", weights_only=False)
            scaler.load_state_dict(scaler_state)
            logger.info("Restored GradScaler state.")

        # 6. Restore RNG
        if (ckpt_dir / "rng_state.pt").exists():
            rng_state = torch.load(ckpt_dir / "rng_state.pt", map_location="cpu", weights_only=False)
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                try:
                    torch.cuda.set_rng_state_all(rng_state["cuda"])
                except Exception:
                    pass
            logger.info("Restored Random Number Generator states.")

        # 7. Metadata
        meta_file = ckpt_dir / "metadata.json"
        metadata = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.best_val_loss = metadata.get("best_val_loss", float("inf"))

        return metadata
