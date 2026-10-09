"""Text-to-Image Diffusion Model Framework CLI Dispatcher."""
import sys


def main():
    print("=" * 70)
    print(" Config-Driven Text-to-Image Diffusion Model Fine-Tuning Framework")
    print(" Architecture: Stable Diffusion 1.5 (Latent Diffusion U-Net)")
    print("=" * 70)
    print("\nAvailable Entry Points:")
    print("  1. Prepare Dataset:      python scripts/prepare_dataset.py --config configs/config.yaml")
    print("  2. Benchmark Memory:     python scripts/benchmark_memory.py --config configs/config.yaml")
    print("  3. Fine-Tune Model:      python train.py --config configs/config.yaml")
    print("  4. Validate Checkpoint:  python validate.py --config configs/config.yaml --checkpoint <path>")
    print("  5. Test Evaluation:      python test.py --config configs/config.yaml --checkpoint <path>")
    print("  6. Generate Images:      python generate.py --config configs/config.yaml --prompt \"...\"")
    print("  7. Run Test Suite:       pytest tests/")
    print("\nSee README.md and configs/configs_README.md for comprehensive documentation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
