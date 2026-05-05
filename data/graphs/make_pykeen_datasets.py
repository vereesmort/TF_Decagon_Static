"""
make_pykeen_datasets.py
-----------------------
Replaces:  data/graphs/make_libkge_datasets.py
Purpose:   Converts the Selfloops edgelist TSV into PyKEEN TriplesFactory
           objects and saves them to disk as .pt files for reuse.

LibKGE required a preprocessing step (preprocess_default.py) that converted
the edgelist into integer-ID files (entity_ids.del, relation_ids.del,
train.del etc.).  PyKEEN handles this internally via TriplesFactory —
no separate preprocessing step is needed.

What this script does:
  1. Reads edgelist_selfloops.tsv (same file the original pipeline built)
  2. Creates a 80/10/10 train/valid/test split (same proportions as original)
  3. Saves three TriplesFactory objects to disk as .pt files
  4. Also saves entity_to_id and relation_to_id dicts as JSON for use
     in assessment.py later

Outputs (written to --out_dir):
  train_tf.pt        TriplesFactory for training
  valid_tf.pt        TriplesFactory for validation
  test_tf.pt         TriplesFactory for test
  entity_to_id.json  {entity_name: int_id}
  relation_to_id.json {relation_name: int_id}

Usage:
    python data/graphs/make_pykeen_datasets.py \\
        --edgelist data/graphs/selfloops/edgelist_selfloops.tsv \\
        --out_dir  data/pykeen/selfloops \\
        --seed     0
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from pykeen.triples import TriplesFactory

parser = argparse.ArgumentParser()
parser.add_argument("--edgelist", required=True,
                    help="edgelist_selfloops.tsv or edgelist_non-naive.tsv (h, r, t — no header)")
parser.add_argument("--out_dir",  required=True)
parser.add_argument("--seed",     type=int, default=0)
parser.add_argument("--shared_vocab_dir", default=None,
                    help="If set, load entity_to_id and relation_to_id from this "
                         "directory instead of building from scratch. Use when "
                         "preparing the non-naive dataset so it shares the same "
                         "integer IDs as the selfloops dataset — required for the "
                         "pretrained PCA/SVD vectors to align correctly.")
args = parser.parse_args()

np.random.seed(args.seed)
os.makedirs(args.out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load edgelist
# ---------------------------------------------------------------------------
print("Loading edgelist...")
edges = pd.read_csv(args.edgelist, sep="\t", header=None,
                    names=["h", "r", "t"], dtype=str)
print(f"  {len(edges):,} total edges")

# ---------------------------------------------------------------------------
# 2. 80/10/10 split  (same as original make_libkge_datasets.py)
# ---------------------------------------------------------------------------
split_size = 0.1
train_valid, test   = train_test_split(edges, test_size=split_size,
                                        random_state=args.seed)
train, valid        = train_test_split(train_valid,
                                        test_size=split_size / (1 - split_size),
                                        random_state=args.seed)
assert abs(len(valid) - len(test)) <= 1, "valid/test size mismatch"
print(f"  train: {len(train):,}  valid: {len(valid):,}  test: {len(test):,}")

# ---------------------------------------------------------------------------
# 3. Build TriplesFactory for training set.
#    The entity and relation vocabularies are built from the FULL edgelist
#    so that valid/test can reuse the same integer IDs.
# ---------------------------------------------------------------------------
print("Building TriplesFactory from full vocabulary...")

if args.shared_vocab_dir:
    # Non-naive: reuse the selfloops vocabulary so that pretrained .npy row
    # indices match the entity integer IDs used during training.
    with open(os.path.join(args.shared_vocab_dir, "entity_to_id.json")) as f:
        entity_to_id = json.load(f)
    with open(os.path.join(args.shared_vocab_dir, "relation_to_id.json")) as f:
        relation_to_id = json.load(f)
    print(f"  Shared vocab loaded from {args.shared_vocab_dir}")
    # Add any relations present in non-naive but absent from selfloops
    new_rels = sorted(set(edges["r"]) - set(relation_to_id))
    if new_rels:
        next_id = max(relation_to_id.values()) + 1
        for rel in new_rels:
            relation_to_id[rel] = next_id
            next_id += 1
        print(f"  Added {len(new_rels)} new relations not in selfloops vocab")
else:
    # Selfloops: build vocab from scratch
    all_entities  = sorted(set(edges["h"]) | set(edges["t"]))
    all_relations = sorted(set(edges["r"]))
    entity_to_id   = {e: i for i, e in enumerate(all_entities)}
    relation_to_id = {r: i for i, r in enumerate(all_relations)}

print(f"  {len(entity_to_id):,} entities  |  {len(relation_to_id):,} relations")

def make_tf(df, entity_to_id, relation_to_id):
    """Build a TriplesFactory from a DataFrame with the shared vocabulary."""
    triples = df[["h", "r", "t"]].values.astype(str)
    return TriplesFactory.from_labeled_triples(
        triples=triples,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

train_tf = make_tf(train, entity_to_id, relation_to_id)
valid_tf = make_tf(valid, entity_to_id, relation_to_id)
test_tf  = make_tf(test,  entity_to_id, relation_to_id)

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
for name, tf in [("train", train_tf), ("valid", valid_tf), ("test", test_tf)]:
    path = os.path.join(args.out_dir, f"{name}_tf.pt")
    torch.save(tf, path)
    print(f"  Saved {path}")

with open(os.path.join(args.out_dir, "entity_to_id.json"), "w") as f:
    json.dump(entity_to_id, f, indent=2)
with open(os.path.join(args.out_dir, "relation_to_id.json"), "w") as f:
    json.dump(relation_to_id, f, indent=2)
print(f"  Saved entity_to_id.json and relation_to_id.json")

print("\nDone. Use --dataset_dir with train_simplE_pykeen.py.")
