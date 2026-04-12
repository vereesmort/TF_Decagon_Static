import pandas as pd
from os import listdir

# Load in data
name_converter = {
    'combo': 'polypharmacy',
    'mono': 'monopharmacy',
    'ppi': 'ppi',
    'targets': 'drug-target'
}
dfs = {}
for f in listdir('../raw'):
    if f.startswith('bio-decagon'):
        df = pd.read_csv(f'../raw/{f}')
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
core_network.to_csv('core_network_ppi_drugtarget.tsv', index=False, header=None, sep='\t')

# Save Mono-/Polypharmacy side effect data to disk in LibKGE format
dfs['monopharmacy'].to_csv('monopharmacy_edges.tsv', index=False, header=None, sep='\t')
dfs['polypharmacy'].to_csv('polypharmacy/polypharmacy_edges.tsv', index=False, header=None, sep='\t')