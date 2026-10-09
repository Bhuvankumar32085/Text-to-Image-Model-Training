"""Unit tests for deterministic and group-aware dataset splitting."""
import pytest
from src.data.splits import create_deterministic_splits, load_split_manifests, save_split_manifests


def test_split_disjointness_and_counts():
    """Verify that train, validation, and test splits are completely disjoint."""
    records = [{"image_path": f"path_{i}.png", "caption": f"caption {i}", "group_id": f"g_{i % 5}"} for i in range(100)]

    train, val, test = create_deterministic_splits(
        records=records,
        train_fraction=0.9,
        test_fraction=0.1,
        validation_fraction_of_train=0.1,
        seed=42,
    )

    # 10% test = 10 samples
    # 90 samples train pool: 10% val = 9 samples -> 81 train, 9 val, 10 test
    assert len(test) == 10
    assert len(val) == 9
    assert len(train) == 81

    train_paths = {r["image_path"] for r in train}
    val_paths = {r["image_path"] for r in val}
    test_paths = {r["image_path"] for r in test}

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)


def test_group_aware_splitting():
    """Verify that samples with identical group_id are never split across train and test."""
    records = []
    # 10 distinct groups, each containing 5 images
    for g in range(10):
        for i in range(5):
            records.append({
                "image_path": f"img_g{g}_{i}.png",
                "caption": f"caption g{g}_{i}",
                "group_id": f"group_{g}",
            })

    train, val, test = create_deterministic_splits(
        records=records,
        train_fraction=0.8,
        test_fraction=0.2,
        validation_fraction_of_train=0.25,
        seed=123,
        group_column="group_id",
    )

    train_groups = {r["group_id"] for r in train}
    val_groups = {r["group_id"] for r in val}
    test_groups = {r["group_id"] for r in test}

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_split_persistence_and_reload(tmp_path):
    """Test saving split manifests to CSV and reloading."""
    records = [{"image_path": f"path_{i}.png", "caption": f"caption {i}"} for i in range(20)]
    train, val, test = create_deterministic_splits(records, seed=42)

    manifest_paths = save_split_manifests(train, val, test, output_dir=tmp_path)
    assert (tmp_path / "train_manifest.csv").exists()

    reloaded_train, reloaded_val, reloaded_test = load_split_manifests(tmp_path)
    assert len(reloaded_train) == len(train)
    assert len(reloaded_val) == len(val)
    assert len(reloaded_test) == len(test)
