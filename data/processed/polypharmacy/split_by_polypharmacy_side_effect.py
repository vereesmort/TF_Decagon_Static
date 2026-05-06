import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.random import seed


seed(0)

_script_dir = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Split polypharmacy edges into train/holdout with SE stratification.",
)
parser.add_argument(
    "--edgelist",
    type=str,
    default=str(_script_dir / "polypharmacy_edges.tsv"),
    help=(
        "Polypharmacy TSV edgelist (no header, tab-separated): "
        "head_entity, relation (side-effect ID), tail_entity — "
        "e.g. output from process_raw_data.py under polypharmacy/."
    ),
)
parser.add_argument(
    "--combo_csv",
    type=str,
    default=str(_script_dir.parent.parent / "raw" / "bio-decagon-combo.csv"),
    help=(
        "Decagon bio-decagon-combo.csv; column 'Polypharmacy Side Effect' "
        "defines which relation values are split by SE."
    ),
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=".",
    help="Directory for train_polypharmacy.tsv and holdout_polypharmacy.tsv.",
)
args = parser.parse_args()
edgelist_path = Path(args.edgelist).expanduser().resolve()
combo_csv_path = Path(args.combo_csv).expanduser().resolve()
out_dir = Path(args.out_dir).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

if not edgelist_path.is_file():
    raise FileNotFoundError(f"--edgelist not found: {edgelist_path}")
if not combo_csv_path.is_file():
    raise FileNotFoundError(f"--combo_csv not found: {combo_csv_path}")

# Load target edgelist (h, r, t)
edges = pd.read_csv(
    edgelist_path,
    header=None,
    sep="\t",
    dtype={0: str, 1: str, 2: str},
)

# Side-effect relation IDs present in the raw combo file (used to filter/group edges)
poly_edges = pd.read_csv(combo_csv_path)["Polypharmacy Side Effect"].unique()

# Create holdout data that has 10% of each polypharmacy side effect
done = False
while not done:
    train_chunks = []
    holdout_chunks = []
    for edge_type, subdf in edges.groupby(1):
        if edge_type in poly_edges:
            train_edges, test_edges = train_test_split(subdf, test_size=0.1)
            train_chunks.append(train_edges)
            holdout_chunks.append(test_edges)
    train_df = pd.concat(train_chunks, ignore_index=True)
    holdout_df = pd.concat(holdout_chunks, ignore_index=True)

    holdout_nodes = set()
    train_nodes = set()
    for col in [0, 2]:
        for node in holdout_df[col].unique():
            holdout_nodes.add(node)
        for node in train_df[col].unique():
            train_nodes.add(node)

    intersect = train_nodes.intersection(holdout_nodes)
    if len(intersect) == 0:
        raise ValueError('Something went wrong - no overlap between train and holdout nodes.')
    elif len(intersect) == len(holdout_nodes):
        print('Found total overlap of holdout nodes with train nodes. Saving..')
        done = True
    else: 
        print('Holdout set contains nodes unseen to train data. Trying again..')

# Save
train_df.to_csv(
    out_dir / "train_polypharmacy.tsv", header=None, index=False, sep="\t"
)
holdout_df.to_csv(
    out_dir / "holdout_polypharmacy.tsv", header=None, index=False, sep="\t"
)
print(f"Wrote train/holdout under {out_dir}")
