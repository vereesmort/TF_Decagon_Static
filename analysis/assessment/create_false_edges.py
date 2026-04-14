import pandas as pd
import numpy as np
import multiprocessing as mp
from argparse import ArgumentParser
from os import listdir


def create_negative_edges(n_fake, pos_edges, entity_list):
    rel = pos_edges[0][1]
    rel_check = all([edge[1] == rel for edge in pos_edges])
    if not rel_check:
        raise ValueError('Positive edges contains multiple relation types.')
    neg_edges = []
    while len(neg_edges) < n_fake:
        head = np.random.choice(entity_list)
        tail = np.random.choice(entity_list)
        edge = [head, rel, tail]
        if edge not in pos_edges and edge not in neg_edges:
            neg_edges.append(edge)
    return neg_edges


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('libkge_data_dir', type=str)
    args = parser.parse_args()

    entity_ids = pd.read_csv(
        f'{args.libkge_data_dir}/entity_ids.del',
        sep='\t',
        header=None,
        index_col=0
    ).to_dict()[1]
    compound_IDs = [
        key for key in entity_ids if entity_ids[key].startswith('CID')
    ]
    relation_ids = pd.read_csv(
        f'{args.libkge_data_dir}/relation_ids.del',
        sep='\t',
        header=None,
        index_col=0
    ).to_dict()[1]

    full_edgelist = pd.concat([
        pd.read_csv(
            f'{args.libkge_data_dir}/{split}.del',
            sep='\t',
            header=None
        )
        for split in ['train', 'test', 'valid']
    ], ignore_index=True)

    holdout = pd.read_csv(
        '../../data/processed/polypharmacy/holdout_polypharmacy.tsv',
        header=None, sep='\t'
    )
    # Convert holdout edges to IDs if necessary
    if all(holdout.dtypes == object):
        entity_name_to_id = {entity_ids[key]: key for key in entity_ids}
        holdout[0] = [entity_name_to_id[name] for name in holdout[0]]
        holdout[2] = [entity_name_to_id[name] for name in holdout[2]]

        relation_name_to_id = {relation_ids[key]: key for key in relation_ids}
        holdout[1] = [relation_name_to_id[name] for name in holdout[1]]
    elif any(holdout.dtypes == object):
        raise ValueError(
            'Appears that there is a mix of IDs and strings in holdout data.'
        )

    # Make arguments to create false edges in parallel
    parallel_args = []
    for rel_id, holdout_subdf in holdout.groupby(1):
        relation = relation_ids[rel_id]
        false_edge_file = f'{relation}.tsv'
        if false_edge_file not in listdir('false_edges'):
            positive_edges = holdout_subdf.to_numpy().tolist()
            train_subdf = full_edgelist.loc[full_edgelist[1] == rel_id]
            positive_edges = positive_edges + train_subdf.to_numpy().tolist()
            parallel_args.append([
                len(holdout_subdf),
                positive_edges,
                compound_IDs
            ])
            print(f'Preparing to make false edges for relation: {relation}')
        else:
            print(f'Found existing fakes for relation: {relation}. Skipping..')

    # Create false edges
    n_cores = mp.cpu_count()
    n_tasks = len(parallel_args)
    print(f'Making false edges for {n_tasks} relations on {n_cores} cores.')
    with mp.Pool(mp.cpu_count()) as pool:
        parallel_results = pool.starmap(create_negative_edges, parallel_args)

    # Save false edges by relation
    print('Saving..')
    for negative_edges in parallel_results:
        rel_id = negative_edges[0][1]
        relation = relation_ids[rel_id]
        false_edge_file = f'{relation}.tsv'
        pd.DataFrame(negative_edges).to_csv(
            f'false_edges/{false_edge_file}',
            header=None, sep='\t', index=False
        )
