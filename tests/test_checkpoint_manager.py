"""Unit tests for CheckpointManager saving, restoring, and pruning."""
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.config.schema import AppConfig
from src.training.checkpoint_manager import CheckpointManager


class RawUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = RawUNet()

    def forward(self, x, *args, **kwargs):
        return self.unet(x)


def test_checkpoint_save_and_restore(tmp_path):
    """Test full checkpoint save, restore, and weight verification."""
    config = AppConfig()
    ckpt_mgr = CheckpointManager(config, checkpoint_dir=tmp_path, max_to_keep=2)

    # Setup mock pipeline
    mock_pipeline = type(
        "MockPipeline",
        (),
        {
            "unet": TinyUNet(),
            "text_encoder": None,
            "vae": None,
            "save_pretrained": lambda self, path: None,
        },
    )()

    optimizer = torch.optim.Adam(mock_pipeline.unet.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Mutate a weight
    with torch.no_grad():
        mock_pipeline.unet.unet.conv.weight.fill_(3.14)

    save_path = ckpt_mgr.save_checkpoint(
        pipeline=mock_pipeline,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=None,
        global_step=50,
        epoch=1,
        val_loss=0.45,
    )

    assert save_path.exists()
    assert (save_path / "unet.pt").exists()
    assert (save_path / "optimizer.pt").exists()
    assert (save_path / "metadata.json").exists()

    # Reset weights
    with torch.no_grad():
        mock_pipeline.unet.unet.conv.weight.fill_(0.0)

    # Restore from checkpoint
    meta = ckpt_mgr.load_checkpoint(save_path, mock_pipeline, optimizer, lr_scheduler)
    assert meta["global_step"] == 50
    assert torch.allclose(mock_pipeline.unet.unet.conv.weight, torch.tensor(3.14))


def test_checkpoint_pruning(tmp_path):
    """Verify that oldest checkpoints are pruned when max_to_keep threshold is exceeded."""
    config = AppConfig()
    ckpt_mgr = CheckpointManager(config, checkpoint_dir=tmp_path, max_to_keep=2)

    mock_pipeline = type(
        "MockPipeline",
        (),
        {
            "unet": TinyUNet(),
            "save_pretrained": lambda self, path: None,
        },
    )()
    optimizer = torch.optim.Adam(mock_pipeline.unet.parameters(), lr=1e-3)

    # Save 4 sequential checkpoints
    ckpt_mgr.save_checkpoint(mock_pipeline, optimizer, None, None, global_step=10, epoch=1)
    ckpt_mgr.save_checkpoint(mock_pipeline, optimizer, None, None, global_step=20, epoch=1)
    ckpt_mgr.save_checkpoint(mock_pipeline, optimizer, None, None, global_step=30, epoch=1)
    ckpt_mgr.save_checkpoint(mock_pipeline, optimizer, None, None, global_step=40, epoch=1)

    # Checkpoints 10 and 20 should be pruned; 30 and 40 should remain
    assert not (tmp_path / "checkpoint_step_000010").exists()
    assert not (tmp_path / "checkpoint_step_000020").exists()
    assert (tmp_path / "checkpoint_step_000030").exists()
    assert (tmp_path / "checkpoint_step_000040").exists()
