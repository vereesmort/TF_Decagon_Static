import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.random import seed


seed(0)

# Load target edgelist
edges = pd.read_csv(
    'polypharmacy_edges.tsv', 
    header=None, sep='\t', 
    dtype={0:str, 1:str, 2:str}
)

# Get list of polypharmacy side effects
poly_edges = pd.read_csv('../../raw/bio-decagon-combo.csv')['Polypharmacy Side Effect'].unique()

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
train_df.to_csv('train_polypharmacy.tsv', header=None, index=False, sep='\t')
holdout_df.to_csv('holdout_polypharmacy.tsv', header=None, index=False, sep='\t')
