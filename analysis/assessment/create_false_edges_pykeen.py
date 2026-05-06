"""
create_false_edges_pykeen.py
-----------------------------
Replaces:  analysis/assessment/create_false_edges.py
Purpose:   Generate random negative drug-pair edges for each SE type.

The logic is identical to the original — for each SE type, sample random
drug pairs not in the positive set.  The only change is that it reads
drug CIDs from entity_to_id.json (produced by make_pykeen_datasets.py)
instead of from LibKGE's entity_ids.del file.

The output format is unchanged: one TSV per SE type in false_edges/,
with string entity names (not integer IDs) — assessment_pykeen.py
converts them to IDs during scoring.

Run from:  TF_Decagon_Static/analysis/assessment/

Usage:
    python create_false_edges_pykeen.py \\
        --dataset_dir data/pykeen/selfloops \\
        --n_cores     8      # default: all CPU cores
"""

import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", required=True,
                    help="Directory with entity_to_id.json / relation_to_id.json")
parser.add_argument("--n_cores", type=int, default=None,
                    help="Number of CPU cores (default: all)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load entity/relation mappings
# ---------------------------------------------------------------------------
with open(os.path.join(args.dataset_dir, "entity_to_id.json")) as f:
    entity_to_id = json.load(f)
with open(os.path.join(args.dataset_dir, "relation_to_id.json")) as f:
    relation_to_id = json.load(f)

# Drug nodes start with "CID" — identical filter to original create_false_edges.py
compound_ids = [name for name in entity_to_id if name.startswith("CID")]
print(f"  {len(compound_ids)} drug nodes available for negative sampling")

# ---------------------------------------------------------------------------
# Load holdout (same relative path as original)
# ---------------------------------------------------------------------------
holdout = pd.read_csv(
    "../../data/processed/polypharmacy/holdout_polypharmacy.tsv",
    header=None, sep="\t", names=["h", "r", "t"], dtype=str
)
print(f"  {len(holdout):,} holdout edges across {holdout['r'].nunique()} SE types")

# Also load train triples from the dataset dir to exclude from negatives
# (mirrors original which excluded train+holdout positives)
try:
    train_tf = __import__("torch").load(
        os.path.join(args.dataset_dir, "train_tf.pt"), weights_only=False)
    # Reconstruct string triples from mapped integers
    id_to_entity   = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    mt = train_tf.mapped_triples.numpy()
    train_str = pd.DataFrame({
        "h": [id_to_entity[x] for x in mt[:, 0]],
        "r": [id_to_relation[x] for x in mt[:, 1]],
        "t": [id_to_entity[x] for x in mt[:, 2]],
    })
    all_edges = pd.concat([holdout, train_str], ignore_index=True)
    print(f"  {len(train_str):,} train edges loaded for exclusion")
except Exception as e:
    print(f"  Warning: could not load train edges ({e}). "
          f"Using holdout only for exclusion.")
    all_edges = holdout

os.makedirs("false_edges_pykeen", exist_ok=True)

# ---------------------------------------------------------------------------
# Negative edge generation (identical logic to original)
# ---------------------------------------------------------------------------
def create_negative_edges(n_fake, pos_edges_set, se_name):
    """Sample n_fake random drug pairs not in pos_edges_set."""
    neg_edges = []
    while len(neg_edges) < n_fake:
        head = np.random.choice(compound_ids)
        tail = np.random.choice(compound_ids)
        if (head, se_name, tail) not in pos_edges_set:
            neg_edges.append([head, se_name, tail])
    return neg_edges


parallel_args = []
for se_name, group in holdout.groupby("r"):
    false_edge_file = f"false_edges_pykeen/{se_name}.tsv"
    if os.path.exists(false_edge_file):
        print(f"  {se_name}: existing file found, skipping.")
        continue

    # Build positive set from ALL edges (train + holdout) for this SE type
    se_all = all_edges[all_edges["r"] == se_name]
    pos_set = set(zip(se_all["h"], se_all["r"], se_all["t"]))
    n_needed = len(group)
    parallel_args.append((n_needed, pos_set, se_name))
    print(f"  {se_name}: need {n_needed} negatives")

n_cores = args.n_cores or mp.cpu_count()
print(f"\nGenerating negatives for {len(parallel_args)} SE types on {n_cores} cores...")

with mp.Pool(n_cores) as pool:
    results = pool.starmap(create_negative_edges, parallel_args)

print("Saving...")
for neg_edges in results:
    if neg_edges:
        se_name = neg_edges[0][1]
        pd.DataFrame(neg_edges, columns=["h", "r", "t"]).to_csv(
            f"false_edges_pykeen/{se_name}.tsv", header=False, index=False, sep="\t"
        )

print(f"Done. {len(os.listdir('false_edges_pykeen'))} files in false_edges_pykeen/")
