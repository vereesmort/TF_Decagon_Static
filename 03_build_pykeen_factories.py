#!/usr/bin/env python3
"""
03_build_pykeen_factories.py
================================================================================
Builds PyKEEN TriplesFactory objects from the raw string TSV files produced
by 02_build_selfloops_dataset.py.

This script REPLACES the LibKGE preprocess_default.py step.
PyKEEN does NOT use .del files — it reads string TSVs directly.

Key differences from LibKGE preprocessing:
  - LibKGE: str TSV → preprocess_default.py → int .del files → training
  - PyKEEN:  str TSV → TriplesFactory (in memory) → training
              (no intermediate files needed)

This script demonstrates how to build and optionally save the factories,
and validates them before training begins.

Usage:
    python 03_build_pykeen_factories.py \\
        --data_dir raw_selfloops/ \\
        --save_dir pykeen_factories/   # optional: save factories for reuse

Output (optional):
    pykeen_factories/training_factory.pkl
    pykeen_factories/validation_factory.pkl
    pykeen_factories/testing_factory.pkl
================================================================================
"""

import argparse
import pickle
from pathlib import Path


def build_factories(data_dir: Path, save_dir: Path | None = None):
    """
    Build PyKEEN TriplesFactory objects from train/valid/test TSV files.

    The shared entity_to_id and relation_to_id maps are constructed from
    the training set and then passed explicitly to valid and test, exactly
    as LibKGE's preprocess_default.py builds a global vocabulary.

    Parameters
    ----------
    data_dir:
        Directory containing train.txt, valid.txt, test.txt
    save_dir:
        Optional directory to pickle the factories for reuse

    Returns
    -------
    (training, validation, testing) TriplesFactory tuple
    """
    from pykeen.triples import TriplesFactory

    train_path = data_dir / "train.txt"
    valid_path = data_dir / "valid.txt"
    test_path  = data_dir / "test.txt"

    for p in [train_path, valid_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing file: {p}\n"
                f"Run 02_build_selfloops_dataset.py first."
            )

    print("Building TriplesFactory objects...")

    # Training factory — builds the vocabulary from scratch
    print("  Loading train.txt...")
    training = TriplesFactory.from_path(
        str(train_path),
        create_inverse_triples=False,   # do not add inverse relations
    )

    # Validation and test MUST share the same vocabulary as training
    # This is equivalent to LibKGE using a global entity_ids.del / relation_ids.del
    print("  Loading valid.txt (shared vocab)...")
    validation = TriplesFactory.from_path(
        str(valid_path),
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
        create_inverse_triples=False,
    )

    print("  Loading test.txt  (shared vocab)...")
    testing = TriplesFactory.from_path(
        str(test_path),
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
        create_inverse_triples=False,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\nFactory statistics:")
    print(f"  Entities  : {training.num_entities:,}  "
          f"(from training vocab)")
    print(f"  Relations : {training.num_relations:,}  "
          f"(expected: 963 PSE types)")
    print(f"  train     : {training.num_triples:>12,} triples")
    print(f"  valid     : {validation.num_triples:>12,} triples")
    print(f"  test      : {testing.num_triples:>12,} triples")
    print(f"  train+valid: {training.num_triples + validation.num_triples:>11,} triples  "
          f"(paper: 5,761,807)")

    # ── Self-loop check ───────────────────────────────────────────────────────
    import torch
    mapped = training.mapped_triples       # [N, 3] int tensor
    self_loops = (mapped[:, 0] == mapped[:, 2]).sum().item()
    print(f"  self-loops in train: {self_loops:,}  (confirms Selfloops variant)")

    # ── Leakage check ─────────────────────────────────────────────────────────
    print("\nChecking for leakage...")
    train_set = set(map(tuple, training.mapped_triples.tolist()))
    valid_leak = sum(
        1 for t in validation.mapped_triples.tolist() if tuple(t) in train_set
    )
    test_leak  = sum(
        1 for t in testing.mapped_triples.tolist()    if tuple(t) in train_set
    )
    if valid_leak > 0:
        print(f"  WARNING: {valid_leak} valid triples appear in train!")
    elif test_leak > 0:
        print(f"  WARNING: {test_leak} test triples appear in train!")
    else:
        print("  Leakage check: PASSED")

    # ── Optional save ─────────────────────────────────────────────────────────
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, factory in [
            ("training_factory.pkl",   training),
            ("validation_factory.pkl", validation),
            ("testing_factory.pkl",    testing),
        ]:
            path = save_dir / name
            with open(path, "wb") as f:
                pickle.dump(factory, f)
            print(f"  Saved: {path}")

    return training, validation, testing


def load_factories(save_dir: Path):
    """
    Load previously saved TriplesFactory objects.
    Faster than rebuilding from TSV on subsequent runs.
    """
    import pickle
    factories = []
    for name in ["training_factory.pkl", "validation_factory.pkl", "testing_factory.pkl"]:
        with open(save_dir / name, "rb") as f:
            factories.append(pickle.load(f))
    return tuple(factories)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build PyKEEN TriplesFactory objects from selfloops TSV files"
    )
    parser.add_argument(
        "--data_dir", type=Path, required=True,
        help="Directory containing train.txt, valid.txt, test.txt"
    )
    parser.add_argument(
        "--save_dir", type=Path, default=None,
        help="Optional: directory to save pickled factories for reuse"
    )
    args = parser.parse_args()

    training, validation, testing = build_factories(args.data_dir, args.save_dir)
    print("\nFactories ready for training.")
    print("Pass these directly to pykeen.pipeline.pipeline() or hpo_pipeline().")
