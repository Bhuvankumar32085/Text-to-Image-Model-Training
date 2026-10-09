"""Path utilities and directory management."""
import os
from datetime import datetime
from pathlib import Path
from typing import Union


def resolve_path(path: Union[str, Path], base_dir: Union[str, Path] = None) -> Path:
    """Resolve a relative or absolute path against an optional base directory."""
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    if base_dir is not None:
        return (Path(base_dir) / p).resolve()
    return p.resolve()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure that the parent directory (or directory itself) exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_timestamp_str() -> str:
    """Return a formatted timestamp string for run directories."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
