"""
assessment_pykeen.py
--------------------
Replaces:  analysis/assessment/assessment.py
Purpose:   Computes AUROC, AUPRC, AP@50 per SE type from a trained PyKEEN
           SimplE model, using the same holdout + false-edges protocol as
           the original paper.

KEY DIFFERENCE FROM assessment.py
-----------------------------------
Original:  uses kge.util.io.load_checkpoint + model.score_spo(s, p, o)
This file: uses torch.load(checkpoint) + model.score_hrt(hrt_batch)

score_hrt() takes a LongTensor of shape (n, 3) where columns are
[head_id, relation_id, tail_id] — the PyKEEN convention (h, r, t).
It returns a FloatTensor of shape (n, 1).

Everything else — false edges, holdout loading, AUROC/AUPRC/AP@50
calculation — is identical to the original assessment.py.

Run from:  TF_Decagon_Static/analysis/assessment/

Usage:
    cd TF_Decagon_Static/analysis/assessment/

    # All 963 SE types
    python assessment_pykeen.py \\
        --checkpoint  results/simplE_selfloops/checkpoints/checkpoint_best.pt \\
        --dataset_dir data/pykeen/selfloops \\
        --out_dir     results/simplE_selfloops/assessment/

    # Subset of SE types (fast check during development)
    python assessment_pykeen.py \\
        --checkpoint  results/simplE_selfloops/checkpoints/checkpoint_best.pt \\
        --dataset_dir data/pykeen/selfloops \\
        --out_dir     results/simplE_selfloops/assessment_subset/ \\
        --ses C0000039 C0002871 C0003126

    # Resume after interruption
    python assessment_pykeen.py \\
        --checkpoint  results/simplE_selfloops/checkpoints/checkpoint_best.pt \\
        --dataset_dir data/pykeen/selfloops \\
        --out_dir     results/simplE_selfloops/assessment/ \\
        --partial_results results/simplE_selfloops/assessment/results_temp.csv

Prerequisites:
    - false_edges/ directory must exist and be populated.
      Run create_false_edges_pykeen.py first if not done.
    - holdout_polypharmacy.tsv must exist at:
      ../../data/processed/polypharmacy/holdout_polypharmacy.tsv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, os.path.dirname(__file__))
import decagon_rank_metrics   # same file as in original repo

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",      required=True,
                    help="checkpoint_best.pt from train_simplE_pykeen.py")
parser.add_argument("--dataset_dir",     required=True,
                    help="Directory with entity_to_id.json / relation_to_id.json")
parser.add_argument("--out_dir",         required=True)
parser.add_argument("--partial_results", default=None)
parser.add_argument("--ses", nargs="+",  default=None,
                    help="Subset of SE names to assess (default: all 963)")
parser.add_argument("--batch_size",      type=int, default=4096,
                    help="Scoring batch size (increase if GPU has memory)")
parser.add_argument("--holdout",         default=None,
                    help="Path to holdout_polypharmacy.tsv. "
                         "Defaults to ../../data/processed/polypharmacy/holdout_polypharmacy.tsv "
                         "relative to this script (works when run from analysis/assessment/).")
args = parser.parse_args()

np.random.seed(0)
os.makedirs(args.out_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Load model from PyKEEN checkpoint
#
#    Two possible formats:
#      a) Full model object  — if the checkpoint was saved with torch.save(model, ...)
#      b) State dict only    — EarlyStopper saves only model.state_dict(), so
#                              we must rebuild the architecture and load weights.
# ---------------------------------------------------------------------------
print("Loading model...")
raw = torch.load(args.checkpoint, map_location=device, weights_only=False)

if isinstance(raw, dict):
    # State-dict only: reconstruct SimplE from the TriplesFactory + infer dim.
    from pykeen.models import SimplE
    from pykeen.losses import CrossEntropyLoss

    train_tf = torch.load(
        os.path.join(args.dataset_dir, "train_tf.pt"),
        weights_only=False, map_location="cpu",
    )
    # Find the entity embedding weight — shape is (n_entities, embedding_dim)
    emb_key = next(
        (k for k in raw if "entity_representations" in k and "weight" in k), None
    )
    if emb_key is None or len(raw[emb_key].shape) < 2:
        # Fallback: find any 2-D tensor whose first dim matches entity count
        n_ent = train_tf.num_entities
        emb_key = next(
            k for k, v in raw.items()
            if hasattr(v, "shape") and len(v.shape) == 2 and v.shape[0] == n_ent
        )
    embedding_dim = raw[emb_key].shape[1]
    print(f"  State-dict checkpoint — rebuilding SimplE  "
          f"(embedding_dim={embedding_dim} inferred from '{emb_key}')")

    model = SimplE(
        triples_factory=train_tf,
        embedding_dim=embedding_dim,
        loss=CrossEntropyLoss(),
    ).to(device)
    # strict=False: ignore regularizer state keys (regularizer.weight,
    # regularizer.regularization_term) — these are training-only scalars
    # that have no effect on score_hrt inference.
    missing, unexpected = model.load_state_dict(raw, strict=False)
    if unexpected:
        print(f"  Ignored unexpected keys (regularizer state): {unexpected}")
else:
    model = raw

model.eval()
print(f"  Model type: {type(model).__name__}")

# ---------------------------------------------------------------------------
# 2. Load entity/relation ID mappings saved by make_pykeen_datasets.py
# ---------------------------------------------------------------------------
with open(os.path.join(args.dataset_dir, "entity_to_id.json")) as f:
    entity_to_id = json.load(f)
with open(os.path.join(args.dataset_dir, "relation_to_id.json")) as f:
    relation_to_id = json.load(f)

id_to_entity   = {v: k for k, v in entity_to_id.items()}
id_to_relation = {v: k for k, v in relation_to_id.items()}
print(f"  {len(entity_to_id):,} entities  |  {len(relation_to_id):,} relations")

# ---------------------------------------------------------------------------
# 3. Load holdout polypharmacy edges
# ---------------------------------------------------------------------------
print("Loading holdout edges...")
_holdout_path = args.holdout or os.path.join(
    os.path.dirname(__file__),
    "../../data/processed/polypharmacy/holdout_polypharmacy.tsv",
)
holdout = pd.read_csv(
    _holdout_path, header=None, sep="\t", names=["h", "r", "t"], dtype=str
)

# Convert string names to integer IDs
holdout["h_id"] = holdout["h"].map(entity_to_id)
holdout["r_id"] = holdout["r"].map(relation_to_id)
holdout["t_id"] = holdout["t"].map(entity_to_id)

missing = holdout[["h_id", "r_id", "t_id"]].isna().any(axis=1).sum()
if missing > 0:
    print(f"  WARNING: {missing} holdout edges have unmapped entities/relations — dropping.")
    holdout = holdout.dropna(subset=["h_id", "r_id", "t_id"])

holdout[["h_id", "r_id", "t_id"]] = holdout[["h_id", "r_id", "t_id"]].astype(int)
print(f"  {len(holdout):,} holdout edges across {holdout['r'].nunique()} SE types")

# ---------------------------------------------------------------------------
# 4. Optional SE subset filter
# ---------------------------------------------------------------------------
if args.ses is not None:
    holdout = holdout[holdout["r"].isin(args.ses)]
    print(f"  Filtered to {holdout['r'].nunique()} SE types ({len(holdout):,} edges)")

# ---------------------------------------------------------------------------
# 5. Scoring helper
#    PyKEEN's score_hrt takes LongTensor shape (n, 3) → returns (n, 1)
#    We batch large edge lists to avoid OOM on GPU.
# ---------------------------------------------------------------------------
def score_edges(h_ids, r_ids, t_ids):
    """Score a list of (h, r, t) integer triples. Returns a flat numpy array."""
    all_scores = []
    hrt = torch.stack([
        torch.tensor(h_ids, dtype=torch.long),
        torch.tensor(r_ids, dtype=torch.long),
        torch.tensor(t_ids, dtype=torch.long),
    ], dim=1).to(device)

    for batch_start in range(0, len(hrt), args.batch_size):
        batch = hrt[batch_start: batch_start + args.batch_size]
        with torch.no_grad():
            scores = model.score_hrt(batch)   # (batch, 1)
        all_scores.append(scores.squeeze(1).cpu().numpy())

    return np.concatenate(all_scores)

# ---------------------------------------------------------------------------
# 6. Load or initialise results
# ---------------------------------------------------------------------------
if args.partial_results and os.path.exists(args.partial_results):
    results = pd.read_csv(args.partial_results)
    already_done = set(results["Relation"])
    print(f"  Resuming: {len(already_done)} SE types already assessed.")
else:
    results = pd.DataFrame(columns=["Relation", "AUROC", "AUPRC", "AP@50"])
    already_done = set()

# ---------------------------------------------------------------------------
# 7. Assessment loop — per SE type, identical logic to original assessment.py
# ---------------------------------------------------------------------------
se_types = holdout["r"].unique()
total    = len(se_types)
print(f"\nAssessing {total} SE types...")

for i, se_name in enumerate(se_types):

    if se_name in already_done:
        print(f"  [{i+1}/{total}] {se_name}: already done, skipping.")
        continue

    # Positive edges for this SE type
    pos_group = holdout[holdout["r"] == se_name]
    pos_h = pos_group["h_id"].tolist()
    pos_r = pos_group["r_id"].tolist()
    pos_t = pos_group["t_id"].tolist()

    # Load false (negative) edges from false_edges_pykeen/ (string-name format)
    false_edge_file = os.path.join("false_edges_pykeen", f"{se_name}.tsv")
    if not os.path.exists(false_edge_file):
        print(f"  [{i+1}/{total}] {se_name}: no false edges file, skipping.")
        print(f"    Run create_false_edges_pykeen.py to generate it.")
        continue

    neg_df = pd.read_csv(false_edge_file, header=None, sep="\t",
                         names=["h", "r", "t"], dtype=str)

    # Map string names → PyKEEN integer IDs
    neg_h = [entity_to_id.get(h, -1)   for h in neg_df["h"]]
    neg_r = [relation_to_id.get(r, -1) for r in neg_df["r"]]
    neg_t = [entity_to_id.get(t, -1)   for t in neg_df["t"]]
    valid_neg = [(h, r, t) for h, r, t in zip(neg_h, neg_r, neg_t)
                 if h != -1 and r != -1 and t != -1]
    if not valid_neg:
        print(f"  [{i+1}/{total}] {se_name}: no valid negatives, skipping.")
        continue
    neg_h, neg_r, neg_t = zip(*valid_neg)

    # Score all positives + negatives together
    all_h = list(pos_h) + list(neg_h)
    all_r = list(pos_r) + list(neg_r)
    all_t = list(pos_t) + list(neg_t)
    labels = [1] * len(pos_h) + [0] * len(neg_h)

    preds = score_edges(all_h, all_r, all_t)

    # AUROC and AUPRC (sklearn)
    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)

    # AP@50 using Decagon's function (identical to original)
    pos_edges_list = list(zip(pos_h, pos_r, pos_t))
    all_edges_list = list(zip(all_h, all_r, all_t))
    ranked_idx     = np.argsort(preds)[::-1]
    ranked_edges   = [all_edges_list[j] for j in ranked_idx]
    ap50 = decagon_rank_metrics.apk(pos_edges_list, ranked_edges, k=50)

    # Store
    results.loc[len(results)] = [se_name, auroc, auprc, ap50]
    results.to_csv(os.path.join(args.out_dir, "results_temp.csv"), index=False)
    print(f"  [{i+1}/{total}] {se_name}: AUROC={auroc:.4f}  "
          f"AUPRC={auprc:.4f}  AP@50={ap50:.4f}")

# ---------------------------------------------------------------------------
# 8. Save and summarise
# ---------------------------------------------------------------------------
final_path = os.path.join(args.out_dir, "results_full.csv")
results.to_csv(final_path, index=False)

temp_path = os.path.join(args.out_dir, "results_temp.csv")
if os.path.exists(temp_path) and len(results) == total:
    os.remove(temp_path)

print(f"\nResults saved to {final_path}")
print(f"\nMedian over {len(results)} SE types:")
print(results[["AUROC", "AUPRC", "AP@50"]].median().round(4).to_string())
