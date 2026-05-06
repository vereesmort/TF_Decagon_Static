import argparse
from pathlib import Path

import pandas as pd
from os import listdir

_script_dir = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Process bio-decagon raw CSVs into LibKGE-style TSV edgelists."
)
parser.add_argument(
    "--raw_dir",
    type=str,
    default=str(_script_dir.parent / "raw"),
    help=(
        "Directory containing bio-decagon-*.csv (Decagon raw downloads). "
        "Default: data/raw relative to this script."
    ),
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=".",
    help="Directory to write output TSV files (default: current working directory).",
)
args = parser.parse_args()
raw_dir = Path(args.raw_dir).expanduser().resolve()
if not raw_dir.is_dir():
    raise NotADirectoryError(f"--raw_dir is not a directory: {raw_dir}")
out_dir = Path(args.out_dir).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
poly_out_dir = out_dir / "polypharmacy"
poly_out_dir.mkdir(parents=True, exist_ok=True)

# Load in data
name_converter = {
    'combo': 'polypharmacy',
    'mono': 'monopharmacy',
    'ppi': 'ppi',
    'targets': 'drug-target'
}
dfs = {}
for f in listdir(raw_dir):
    if f.startswith('bio-decagon'):
        df = pd.read_csv(raw_dir / f)
        f = f.split('-')[-1][:-4]
        new_name = name_converter[f]
        dfs[new_name] = df

# Drop unnecessary columns
dfs['monopharmacy'].drop(columns=['Side Effect Name'], inplace=True)
dfs['polypharmacy'].drop(columns=['Side Effect Name'], inplace=True)

# Process all to (h, r, t) format
dfs['ppi']['r'] = 'ProteinProteinInteraction'  # Set relation column values
dfs['ppi'].columns = ['h', 't', 'r']  # Set column names
dfs['ppi'] = dfs['ppi'][['h', 'r', 't']]  # Order the columns correctly

dfs['drug-target']['r'] = 'DrugTarget'
dfs['drug-target'].columns = ['h', 't', 'r']
dfs['drug-target'] = dfs['drug-target'][['h', 'r', 't']]

dfs['monopharmacy']['r'] = 'MonopharmacySideEffect'
dfs['monopharmacy'].columns = ['h', 't', 'r']
dfs['monopharmacy'] = dfs['monopharmacy'][['h', 'r', 't']]

dfs['polypharmacy'].columns = ['h', 't', 'r']  # Don't need to add 'r' column (as above), just have to rename 'Polypharmacy Side Effect' column
dfs['polypharmacy'] = dfs['polypharmacy'][['h', 'r', 't']]

# Filter polypharmacy side effects by frequency
poly_SE_counts = dict(dfs['polypharmacy']['r'].value_counts())
poly_SE_to_keep = [SE for SE in poly_SE_counts if poly_SE_counts[SE] >= 500]  # 500 is threshold used in Decagon paper
dfs['polypharmacy'] = dfs['polypharmacy'].loc[dfs['polypharmacy']['r'].isin(poly_SE_to_keep)]

# Save core graph to disk in LibKGE format
core_network = pd.concat([dfs['ppi'], dfs['drug-target']], ignore_index=True)
core_network.to_csv(
    out_dir / "core_network_ppi_drugtarget.tsv",
    index=False, header=None, sep="\t",
)

# Save Mono-/Polypharmacy side effect data to disk in LibKGE format
dfs['monopharmacy'].to_csv(
    out_dir / "monopharmacy_edges.tsv",
    index=False, header=None, sep="\t",
)
dfs['polypharmacy'].to_csv(
    poly_out_dir / "polypharmacy_edges.tsv",
    index=False, header=None, sep="\t",
)

print(f"Wrote outputs under {out_dir}")
