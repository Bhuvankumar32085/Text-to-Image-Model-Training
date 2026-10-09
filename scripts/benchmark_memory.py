"""Hardware and memory benchmarking tool running forward/backward passes on synthetic batches."""
import argparse
import logging
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config
from src.models.model_factory import ModelFactory
from src.monitoring.memory_monitor import MemoryMonitor
from src.training.device_manager import DeviceManager
from src.training.forward_diffusion import ForwardDiffusion
from src.training.loss import DiffusionLoss
from src.training.optimizer import create_optimizer_and_scheduler
from src.training.train_step import TrainStepExecutor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def run_memory_benchmark(config_path: str = "configs/config.yaml") -> None:
    """Benchmark memory consumption and execution time for a single training step."""
    config = load_config(config_path)
    logger.info("=== HARDWARE & MEMORY BENCHMARK ===")

    dev_mgr = DeviceManager(config)
    device = dev_mgr.device
    logger.info(f"Target device: {device}")

    # Build model pipeline
    logger.info("Instantiating Stable Diffusion 1.5 pipeline...")
    pipeline = ModelFactory.build_pipeline(config)
    pipeline.to_device(device)

    # Build synthetic test batch
    b = config.training.train_batch_size
    h, w = config.dataset.target_resolution
    logger.info(f"Generating synthetic batch: BatchSize={b}, Resolution={h}x{w}")

    synthetic_batch = {
        "pixel_values": torch.randn(b, 3, h, w, device=device),
        "input_ids": torch.randint(0, 49408, (b, 77), device=device),
        "attention_mask": torch.ones(b, 77, device=device),
        "captions": ["A synthetic benchmark prompt"] * b,
        "image_paths": ["synthetic_path.png"] * b,
    }

    # Setup optimizer and step executor
    optimizer, lr_scheduler = create_optimizer_and_scheduler(config, pipeline, total_training_steps=10)
    loss_fn = DiffusionLoss(loss_type="mse")
    fwd_diff = ForwardDiffusion(pipeline.scheduler)

    step_executor = TrainStepExecutor(
        config=config,
        pipeline=pipeline,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        forward_diffusion=fwd_diff,
        device=device,
    )

    MemoryMonitor.reset_peak_stats(device)
    initial_mem = MemoryMonitor.get_memory_snapshot(device)

    logger.info(f"Initial VRAM Allocated: {initial_mem.get('cuda_allocated_mb', 0)} MB")

    # Run single training step
    logger.info("Executing benchmark forward & backward training step...")
    start_time = time.perf_counter()
    loss_val, metrics, _ = step_executor.execute_step(synthetic_batch, step_idx=0, collect_diagnostics=True)
    step_duration = time.perf_counter() - start_time

    final_mem = MemoryMonitor.get_memory_snapshot(device)

    logger.info("=== BENCHMARK RESULTS ===")
    logger.info(f"Step Loss: {loss_val:.6f}")
    logger.info(f"Step Execution Time: {step_duration*1000:.1f} ms")
    logger.info(f"Forward Pass Time:   {metrics['fwd_time']*1000:.1f} ms")
    logger.info(f"Backward Pass Time:  {metrics['bwd_time']*1000:.1f} ms")
    logger.info(f"Estimated Throughput: {b / step_duration:.2f} images/sec")

    if device.type == "cuda":
        peak_alloc = final_mem.get("cuda_max_allocated_mb", 0.0)
        peak_res = final_mem.get("cuda_max_reserved_mb", 0.0)
        logger.info(f"Peak VRAM Allocated: {peak_alloc:.2f} MB ({peak_alloc / 1024:.2f} GB)")
        logger.info(f"Peak VRAM Reserved:  {peak_res:.2f} MB ({peak_res / 1024:.2f} GB)")

        dev_info = dev_mgr.device_info
        total_vram_gb = dev_info.get("total_memory_gb", 0.0)
        if total_vram_gb:
            logger.info(f"Total GPU VRAM:      {total_vram_gb:.2f} GB")
            vram_util = (peak_res / 1024) / total_vram_gb * 100
            logger.info(f"VRAM Peak Utilization: {vram_util:.1f}%")

            if peak_res / 1024 > 5.5:
                logger.warning(
                    "High VRAM warning: Peak usage is close to 6 GB. "
                    "Ensure gradient_checkpointing is enabled and batch_size is 1."
                )

    # Clean up
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("Benchmark complete. Memory released successfully.")


def main():
    parser = argparse.ArgumentParser(description="Run hardware and memory benchmark for diffusion training.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run_memory_benchmark(args.config)


if __name__ == "__main__":
    main()
