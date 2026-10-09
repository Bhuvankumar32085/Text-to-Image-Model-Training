"""Evaluation report generator compiling loss metrics, images, and experiment findings."""
import json
from pathlib import Path
from typing import Any, Dict, Optional


def generate_evaluation_report(
    test_results: Dict[str, Any],
    output_path: str = "outputs/reports/evaluation_report.md",
) -> str:
    """Render a comprehensive Markdown evaluation report."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    exp_name = test_results.get("experiment_name", "Unknown")
    test_loss = test_results.get("test_loss")
    loss_str = f"{test_loss:.6f}" if test_loss is not None else "N/A (Loss evaluation skipped)"
    sample_count = test_results.get("test_samples_count", 0)
    eval_time = test_results.get("evaluation_time_sec", 0.0)

    gen_images = test_results.get("generated_images", [])
    metrics_info = test_results.get("metrics", {}).get("metrics", {})

    report = f"""# Test Evaluation & Benchmark Report

**Experiment**: `{exp_name}`  
**Report Output**: `{p.resolve()}`  

---

## 1. Held-Out Test Set Metrics
- **Test Loss**: `{loss_str}`
- **Test Set Samples**: {sample_count}
- **Evaluation Time**: {eval_time:.2f} seconds
"""

    if metrics_info:
        report += "\n## 2. Quantitative Image Generation Metrics\n"
        for k, v in metrics_info.items():
            report += f"- **{k}**: `{v}`\n"

    if gen_images:
        report += "\n## 3. Benchmark Generated Samples\n"
        for item in gen_images:
            report += f"- **Prompt**: \"{item.get('prompt')}\"\n"
            report += f"  - Path: `{item.get('file_path')}`\n"
            report += f"  - Steps: {item.get('steps')}, Guidance: {item.get('guidance_scale')}, Seed: {item.get('seed')}\n"

    with open(p, "w", encoding="utf-8") as f:
        f.write(report)

    return str(p)
