"""Deterministic dataset splitting with group-aware leakage prevention and manifest persistence."""
import csv
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


def compute_manifest_fingerprint(records: List[Dict[str, Any]]) -> str:
    """Compute an MD5 checksum fingerprint over all records in a manifest."""
    hasher = hashlib.md5()
    for rec in sorted(records, key=lambda x: str(x.get("image_path", ""))):
        serialized = json.dumps(rec, sort_keys=True)
        hasher.update(serialized.encode("utf-8"))
    return hasher.hexdigest()


def create_deterministic_splits(
    records: List[Dict[str, Any]],
    train_fraction: float = 0.90,
    test_fraction: float = 0.10,
    validation_fraction_of_train: float = 0.10,
    seed: int = 42,
    group_column: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records deterministically into train, validation, and test disjoint sets.
    
    Returns:
        (train_records, val_records, test_records)
    """
    if len(records) == 0:
        raise ValueError("Cannot split empty record list.")

    rng = np.random.RandomState(seed)

    if group_column is not None and any(group_column in r for r in records):
        # Group-aware splitting to prevent data leakage across subjects/videos
        group_to_records = defaultdict(list)
        for r in records:
            gid = r.get(group_column, f"ungrouped_{id(r)}")
            group_to_records[gid].append(r)

        unique_groups = sorted(list(group_to_records.keys()))
        rng.shuffle(unique_groups)

        num_groups = len(unique_groups)
        num_test_groups = max(1, int(round(num_groups * test_fraction))) if test_fraction > 0 else 0
        test_groups = set(unique_groups[:num_test_groups])
        train_pool_groups = unique_groups[num_test_groups:]

        num_val_groups = max(1, int(round(len(train_pool_groups) * validation_fraction_of_train))) if validation_fraction_of_train > 0 and len(train_pool_groups) > 1 else 0
        val_groups = set(train_pool_groups[:num_val_groups])
        train_groups = set(train_pool_groups[num_val_groups:])

        # Fallback if too few groups
        if not train_groups and val_groups:
            train_groups = val_groups
            val_groups = set()

        train_records = [r for g in train_groups for r in group_to_records[g]]
        val_records = [r for g in val_groups for r in group_to_records[g]]
        test_records = [r for g in test_groups for r in group_to_records[g]]
    else:
        # Standard deterministic item-level shuffle
        indices = np.arange(len(records))
        rng.shuffle(indices)

        num_test = max(1, int(round(len(records) * test_fraction))) if test_fraction > 0 else 0
        test_idx = indices[:num_test]
        train_pool_idx = indices[num_test:]

        num_val = max(1, int(round(len(train_pool_idx) * validation_fraction_of_train))) if validation_fraction_of_train > 0 and len(train_pool_idx) > 1 else 0
        val_idx = train_pool_idx[:num_val]
        train_idx = train_pool_idx[num_val:]

        if len(train_idx) == 0 and len(val_idx) > 0:
            train_idx = val_idx
            val_idx = np.array([], dtype=int)

        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        test_records = [records[i] for i in test_idx]

    # Verify disjointness
    def get_identities(recs):
        return {r.get("image_path") for r in recs if "image_path" in r}

    train_ids = get_identities(train_records)
    val_ids = get_identities(val_records)
    test_ids = get_identities(test_records)

    overlap_train_val = train_ids.intersection(val_ids)
    overlap_train_test = train_ids.intersection(test_ids)
    overlap_val_test = val_ids.intersection(test_ids)

    if overlap_train_val:
        raise ValueError(f"Data leakage detected! Overlap between train and val: {len(overlap_train_val)} samples.")
    if overlap_train_test:
        raise ValueError(f"Data leakage detected! Overlap between train and test: {len(overlap_train_test)} samples.")
    if overlap_val_test:
        raise ValueError(f"Data leakage detected! Overlap between val and test: {len(overlap_val_test)} samples.")

    return train_records, val_records, test_records


def save_split_manifests(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    output_dir: Union[str, Path] = "data/splits",
) -> Dict[str, str]:
    """Save train, val, and test splits into CSV manifests and return file paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = {}
    for name, recs in [("train", train_records), ("val", val_records), ("test", test_records)]:
        file_path = out_dir / f"{name}_manifest.csv"
        manifest_paths[name] = str(file_path)
        if recs:
            keys = list(recs[0].keys())
            # Exclude complex objects if present
            fieldnames = [k for k in keys if k != "original_record"]
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in recs:
                    row = {k: r.get(k, "") for k in fieldnames}
                    writer.writerow(row)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("image_path,caption\n")

    return manifest_paths


def load_split_manifests(manifest_dir: Union[str, Path] = "data/splits") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load pre-generated train, val, test CSV manifests."""
    d = Path(manifest_dir)
    results = []
    for split in ["train", "val", "test"]:
        path = d / f"{split}_manifest.csv"
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        recs = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                recs.append(row)
        results.append(recs)
    return results[0], results[1], results[2]
