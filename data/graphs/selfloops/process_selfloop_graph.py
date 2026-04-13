import pandas as pd

# Load core graph (drug-target and PPI edges)
core = pd.read_csv('../../processed/core_network_ppi_drugtarget.tsv', header=None, sep='\t', dtype={0:str, 1:str, 2:str})
core.columns = ['head', 'relation', 'tail']
core.drop_duplicates(inplace=True)

# Add monopharmic side effect edges as drug self-loops
monoSE = pd.read_csv('../../processed/monopharmacy_edges.tsv', header=None, sep='\t')
monoSE.columns = ['head', 'relation', 'tail']
monoSE['relation'] = monoSE['tail']  # Side effect type becomes the relation
monoSE['tail'] = monoSE['head']  # Tail becomes same node as head
monoSE.drop_duplicates(inplace=True)

# Add Polypharmic side effect edges
polySE = pd.read_csv('../../processed/polypharmacy/train_polypharmacy.tsv', header=None, sep='\t')
polySE.columns = ['head', 'relation', 'tail']
polySE.drop_duplicates(inplace=True)

out_edges = pd.concat([core, monoSE, polySE], ignore_index=True)

# Save edges
out_edges.to_csv('edgelist_selfloops.tsv', sep='\t', header=None, index=False)