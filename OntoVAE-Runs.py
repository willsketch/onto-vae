#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import pickle
import argparse
import torch
import torch.nn.functional as F
from onto_vae.ontobj import Ontobj
from onto_vae.vae_model import OntoVAE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from dataset_configs import DATASETS


# --------------------------
# pass arguments
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run OntoVAE experiments")
    parser.add_argument("--obo_path", type=str, default="data/go-basic.obo")
    parser.add_argument("--gene_annot_path", type=str, default="data/gene_annot_ontovae.txt")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2,3,4,5,6])
    parser.add_argument("--top_thresh_ontobj", type=int, default=100)
    parser.add_argument("--bottom_thresh_ontobj", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_coeff", type=float, default=1e-4)
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Dataset names to run e.g. --datasets TCGA SEAAD. Defaults to all in dataset_configs.py.")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Randomly subsample N cells per split. For local testing of large datasets.")
    return parser.parse_args()

# --------------------------
# Utility functions
# --------------------------

def load_split(expr_path, split_ids_path, nan_cols, drop_genes, id_col='patient_id',
               label_col='cancer_type', donor_col=None, subsample=None):
    """
    Load a subset of expression data given IDs and drop metadata columns.

    Parameters
    ----------
    id_col
        column used as row index (e.g. 'patient_id' for TCGA, 'sample_id' for SEAAD)
    label_col
        column used as cluster labels (e.g. 'cancer_type' for TCGA, 'Subclass' for SEAAD)
    donor_col
        if provided, split_ids are donor IDs and rows are filtered by this column.
        Used for datasets where multiple rows (cells) belong to the same donor.
    """
    df = pd.read_csv(expr_path)
    ids = np.load(split_ids_path, allow_pickle=True)

    filter_col = donor_col if donor_col else id_col
    df = df[df[filter_col].isin(ids)]
    df = df.dropna(subset=label_col)

    if subsample is not None and len(df) > subsample:
        orig_len = len(df)
        df = df.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(n=max(1, int(round(len(x) * subsample / orig_len))), random_state=42)
        )
        print(f"  Subsampled to {len(df)} cells (from {orig_len} total, stratified by {label_col})")
        print(df[label_col].value_counts().to_string())

    labels = df[label_col].copy()
    df = df.drop(columns=nan_cols)

    if drop_genes is not None:
        df = df.drop(columns=drop_genes, errors='ignore')

    df = df.set_index(id_col, drop=True)
    df.index.name = None
    df = df.T
    df.columns.name = None
    return df, labels


def setup_ontology(obo_path, gene_annot_path, top_thresh=1000, bottom_thresh=30, description='GO-untrimmed'):
    """
    Initialize, trim, and create masks for Ontobj.
    """
    ont = Ontobj(description=description)
    ont.initialize_dag(obo=obo_path, gene_annot=gene_annot_path)
    ont.trim_dag(top_thresh=top_thresh, bottom_thresh=bottom_thresh)
    ont.create_masks(top_thresh=top_thresh, bottom_thresh=bottom_thresh)
    # ont.randomize_masks(top_thresh=top_thresh, bottom_thresh=bottom_thresh, Q=100, seed=4)
    return ont


def match_dataset(ontobj, expr_df, name, top_thresh=1000, bottom_thresh=30):
    """
    Match the dataframe to the ontology within Ontobj.
    """
    ontobj.match_dataset(expr_data=expr_df, name=name, top_thresh=top_thresh, bottom_thresh=bottom_thresh)
    return ontobj


def train_ontovae(ontobj, dataset_name, top_thresh, bottom_thresh,
                  save_path='models/best_model.pt', lr=1e-4, kl_coeff=1e-4, batch_size=32, epochs=20, log=None,
                  mask_override=None):
    """
    Initialize OntoVAE and train the model.
    """
    model = OntoVAE(ontobj=ontobj, dataset=dataset_name, top_thresh=top_thresh, bottom_thresh=bottom_thresh,
                    mask_override=mask_override)
    model.to(model.device)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.train_model(save_path, lr=lr, kl_coeff=kl_coeff, batch_size=batch_size, epochs=epochs, run=log)
    return model


def compute_latent_embeddings(model, ontobj, dataset_name, top_thresh, bottom_thresh, batch_size=512):
    """
    Return latent embeddings for a given dataset, processed in batches to avoid GPU OOM.
    """
    model.eval()
    device = model.device
    trim_key = f'{top_thresh}_{bottom_thresh}'

    if trim_key not in ontobj.data:
        raise KeyError(f'Trim key {trim_key} not found in {ontobj.dataset}. '
                       f'First set up ontology with [{top_thresh}, {bottom_thresh}]')
    elif dataset_name not in ontobj.data[trim_key]:
        raise KeyError(f'Dataset {dataset_name} not found in {ontobj.dataset}.'
                       f'Match dataset: {dataset_name} with onology object')
    elif ontobj.data[trim_key][dataset_name] is None:
        raise ValueError(f'Dataset in ontology object is None')

    x = torch.tensor(ontobj.data[trim_key][dataset_name], dtype=torch.float32)
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size].to(device)
            chunks.append(model.get_embedding(x_batch).cpu().numpy())

    return np.vstack(chunks)


def get_non_zero_cols(x):

    epsilon = 1e-6
    return np.where(np.abs(x).sum(axis=0) > epsilon)[0]


def mse_loss(model, ontObj, dataset_name, top_thresh, bottom_thresh, batch_size=512):
    """
    Calculate mse of reconstruction for given dataset, processed in batches to avoid GPU OOM.
    """
    trim_key = f'{top_thresh}_{bottom_thresh}'

    x = torch.tensor(ontObj.data[trim_key][dataset_name], dtype=torch.float32)
    valid_genes_key = f"{dataset_name}_GONNECT_GENE_MAP"
    valid_genes_idx = ontObj.data[trim_key][valid_genes_key]
    valid_genes_mask = valid_genes_idx != -1

    total_loss = 0.0
    total_elements = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size].to(model.device)
            rec_batch, _, _ = model.forward(x_batch)
            x_sub = x_batch[:, valid_genes_idx[valid_genes_mask]]
            rec_sub = rec_batch[:, valid_genes_idx[valid_genes_mask]]
            total_loss += F.mse_loss(rec_sub, x_sub, reduction='sum').item()
            total_elements += x_sub.numel()

    return total_loss / total_elements

def save_pathway_activities(model, ontobj, dataset_name, top_thresh, bottom_thresh, save_path, raw=False):
    """
    Save per-term pathway activations to disk.

    Parameters
    ----------
    save_path
        base path without extension — .parquet is appended when raw=True, .csv when raw=False
    raw
        False: save averaged activations as CSV with GO IDs as column names
        True: save raw neuron activations as parquet with MultiIndex columns (go_id, neuron)
              reload with pd.read_parquet() to restore the MultiIndex
    """
    act = model.get_pathway_activities(ontobj, dataset_name, raw=raw)
    if raw:
        annot = ontobj.extract_annot(top_thresh=top_thresh, bottom_thresh=bottom_thresh)
        go_id_to_depth = annot.set_index('ID')['depth'].to_dict()
        go_ids = act.columns.get_level_values('go_id')
        neurons = act.columns.get_level_values('neuron')
        depths = [go_id_to_depth[gid] for gid in go_ids]
        act.columns = pd.MultiIndex.from_arrays(
            [go_ids, depths, neurons],
            names=['go_id', 'depth', 'neuron']
        )
        act.to_parquet(save_path + '.parquet')
    else:
        annot = ontobj.extract_annot(top_thresh=top_thresh, bottom_thresh=bottom_thresh)
        pd.DataFrame(act, columns=annot['ID'].tolist()).to_csv(save_path + '.csv', index=False)


def clustering_metrics(latent, labels, seed=42):
    """
    Compute clustering metrics: NMI, ARI, Silhouette Score.
    """
    n_clusters = labels.nunique()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    cluster_assignments = kmeans.fit_predict(latent)

    unique, counts = np.unique(cluster_assignments, return_counts=True)
    print(f"  KMeans: requested {n_clusters} clusters, got {len(unique)} non-empty — sizes: {dict(zip(unique.tolist(), counts.tolist()))}")

    nmi = normalized_mutual_info_score(labels, cluster_assignments)
    ari = adjusted_rand_score(labels, cluster_assignments)

    # Silhouette requires >1 unique cluster in the actual assignments
    n_unique = len(unique)
    sil_score = silhouette_score(latent, cluster_assignments) if n_unique > 1 else np.nan

    return {'NMI': nmi, 'ARI': ari, 'Silhouette': sil_score}


# --------------------------
# Main pipeline per run
# --------------------------

def run_experiment(run_seed, expr_data_path, split_dir, nan_cols, drop_genes, base_ontObj, model_dir='models',
                   top_thresh_ontobj=1000, bottom_thresh_ontobj=30, batch_size=32, epochs=20,
                   lr=1e-4, kl_coeff=1e-4, id_col='patient_id', label_col='cancer_type', donor_col=None,
                   subsample=None):
    """
    Run a full OntoVAE training + evaluation for a single seed.
    Saves the trained model and metrics to disk.
    """

    print(f'Running OntoVAE for split with seed: {run_seed}\n')

    run_folder = os.path.join(split_dir, f'run_seed-{run_seed}')

    # --------------------------
    # Load data
    # --------------------------
    train_path = os.path.join(run_folder, 'train_ids.npy')
    val_path = os.path.join(run_folder, 'val_ids.npy')
    test_path = os.path.join(run_folder, 'test_ids.npy')

    train_df, train_labels = load_split(expr_data_path, train_path, nan_cols, drop_genes=None,
                                        id_col=id_col, label_col=label_col, donor_col=donor_col,
                                        subsample=subsample)
    val_df, val_labels = load_split(expr_data_path, val_path, nan_cols, drop_genes=None,
                                    id_col=id_col, label_col=label_col, donor_col=donor_col,
                                    subsample=subsample)
    test_df, test_labels = load_split(expr_data_path, test_path, nan_cols, drop_genes=drop_genes,
                                      id_col=id_col, label_col=label_col, donor_col=donor_col,
                                      subsample=subsample)

    combined_train_df = pd.concat([train_df, val_df], axis=1)
    combined_train_labels = pd.concat([train_labels, val_labels], axis=0)

    #TODO also check whether labels still correspond to the patient id
    assert len(combined_train_df.columns) == len(combined_train_labels.index)

    print('Data loaded \n')
    # --------------------------
    # Match dataset
    # --------------------------
    dataset_name_train = f'TCGA_run_seed-{run_seed}-train'
    dataset_name_test = f'TCGA_run_seed-{run_seed}-test'
    ont_train = match_dataset(base_ontObj, combined_train_df, name=dataset_name_train, top_thresh=top_thresh_ontobj,
                              bottom_thresh=bottom_thresh_ontobj)

    ont_test = match_dataset(base_ontObj, test_df, name=dataset_name_test, top_thresh=top_thresh_ontobj,
                             bottom_thresh=bottom_thresh_ontobj)

    print('Datasets matched with Ontology \n')

    # --------------------------
    # Generate randomized masks
    # --------------------------
    masks_random = base_ontObj.randomize_masks(top_thresh=top_thresh_ontobj, bottom_thresh=bottom_thresh_ontobj,
                                               method='random', seed=run_seed)
    masks_dp = base_ontObj.randomize_masks(top_thresh=top_thresh_ontobj, bottom_thresh=bottom_thresh_ontobj,
                                           method='degree_preserving', Q=100, seed=run_seed)

    conditions = {
        'true': None,
        'random': masks_random,
        'degree_preserving': masks_dp,
    }

    # --------------------------
    # Train, evaluate and save per condition
    # --------------------------
    model_dir_path = os.path.join(run_folder, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)

    all_metrics = {}

    for condition, mask_override in conditions.items():
        print(f'Starting training [{condition}] \n')

        model_save_path = os.path.join(model_dir_path, f'best_model_{condition}.pt')
        model = train_ontovae(ont_train, dataset_name_train, top_thresh=top_thresh_ontobj,
                              bottom_thresh=bottom_thresh_ontobj, save_path=model_save_path,
                              lr=lr, kl_coeff=kl_coeff, batch_size=batch_size, epochs=epochs,
                              mask_override=mask_override)

        print(f'Computing metrics [{condition}] \n')

        mse_train = mse_loss(model, ont_train, dataset_name_train, top_thresh=top_thresh_ontobj,
                             bottom_thresh=bottom_thresh_ontobj)
        mse_test = mse_loss(model, ont_test, dataset_name_test, top_thresh=top_thresh_ontobj,
                            bottom_thresh=bottom_thresh_ontobj)

        latent_train = compute_latent_embeddings(model, ont_train, dataset_name_train,
                                                 top_thresh=top_thresh_ontobj, bottom_thresh=bottom_thresh_ontobj)
        metrics_train = clustering_metrics(latent_train, combined_train_labels)
        metrics_train['mse'] = mse_train

        latent_test = compute_latent_embeddings(model, ont_test, dataset_name_test,
                                                top_thresh=top_thresh_ontobj, bottom_thresh=bottom_thresh_ontobj)
        print(f"  [{condition}] latent_test NaN: {np.isnan(latent_test).any()}, unique rows: {len(np.unique(latent_test, axis=0))}/{len(latent_test)}")
        metrics_test = clustering_metrics(latent_test, test_labels)
        metrics_test['mse'] = mse_test

        save_pathway_activities(model, ont_train, dataset_name_train,
                                top_thresh_ontobj, bottom_thresh_ontobj,
                                os.path.join(run_folder, f'pathway_activities_train_{condition}'), raw=True)
        save_pathway_activities(model, ont_test, dataset_name_test,
                                top_thresh_ontobj, bottom_thresh_ontobj,
                                os.path.join(run_folder, f'pathway_activities_test_{condition}'), raw=True)

        all_metrics[condition] = {'train': metrics_train, 'test': metrics_test}

        del model
        torch.cuda.empty_cache()

    # --------------------------
    # Save metrics
    # --------------------------
    metrics_path = os.path.join(split_dir, 'metrics.txt')
    with open(metrics_path, 'a') as f:
        f.write(f"run-{run_seed}: {all_metrics}\n")

    print(f"[Run_seed {run_seed}] Metrics saved to {metrics_path}")
    return all_metrics



if __name__ == "__main__":
    args = parse_args()
    obo_path = args.obo_path
    gene_annot_path = args.gene_annot_path
    ontology_description = 'GO-basic'
    model_dir = args.model_dir
    top_thresh_ontobj = args.top_thresh_ontobj
    bottom_thresh_ontobj = args.bottom_thresh_ontobj
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    kl_coeff = args.kl_coeff
    seeds = args.seeds
    trim_key = f"{top_thresh_ontobj}_{bottom_thresh_ontobj}"

    # --------------------------
    # Filter datasets
    # --------------------------
    active_datasets = DATASETS
    if args.datasets is not None:
        active_datasets = [d for d in DATASETS if d['name'] in args.datasets]
        if not active_datasets:
            raise ValueError(f"No matching datasets found. Available: {[d['name'] for d in DATASETS]}")

    # --------------------------
    # Loop over datasets
    # --------------------------
    for dataset_cfg in active_datasets:
        name        = dataset_cfg['name']
        split_dir   = dataset_cfg['split_dir']
        expr_path   = dataset_cfg['expr_data_path']
        id_col      = dataset_cfg['id_col']
        label_col   = dataset_cfg['label_col']
        donor_col   = dataset_cfg.get('donor_col', None)
        nan_cols    = dataset_cfg['nan_cols']
        drop_genes  = dataset_cfg.get('drop_genes', None)

        print(f'\n{"="*50}')
        print(f'Dataset: {name}')
        print(f'{"="*50}\n')

        os.makedirs(split_dir, exist_ok=True)

        # --------------------------
        # Initialize ontology (cached per split_dir)
        # --------------------------
        cached_ontObj_path = os.path.join(split_dir, 'cached_ontology.pkl')
        rebuild_ontology = False

        if os.path.exists(cached_ontObj_path):
            with open(cached_ontObj_path, "rb") as f:
                BaseOntObj = pickle.load(f)
            if trim_key in BaseOntObj.masks:
                print(f"[{name}] Loaded cached ontology")
            else:
                print(f"[{name}] Cached ontology found but trim key missing, rebuilding")
                rebuild_ontology = True
        else:
            print(f"[{name}] Initializing ontology")
            rebuild_ontology = True

        if rebuild_ontology:
            BaseOntObj = setup_ontology(obo_path, gene_annot_path, top_thresh=top_thresh_ontobj,
                                        bottom_thresh=bottom_thresh_ontobj,
                                        description=ontology_description)
            with open(cached_ontObj_path, "wb") as f:
                pickle.dump(BaseOntObj, f)
            print(f"[{name}] Built and cached ontology")

        # --------------------------
        # Compute missing genes once per dataset
        # --------------------------
        all_cols = pd.read_csv(expr_path, nrows=0).columns.tolist()
        meta = set(nan_cols + [id_col] + ([donor_col] if donor_col else []))
        gene_cols = [c for c in all_cols if c not in meta]
        if drop_genes:
            gene_cols = [c for c in gene_cols if c not in drop_genes]
        ontology_genes = set(BaseOntObj.genes[trim_key])
        missing = sorted(set(gene_cols) - ontology_genes)
        if missing:
            missing_path = os.path.join(split_dir, 'missing_genes.txt')
            with open(missing_path, 'w') as f:
                for g in missing:
                    f.write(g + '\n')
            print(f"[{name}] {len(missing)} genes missing from ontology, saved to {missing_path}")

        # --------------------------
        # Run all seeds
        # --------------------------
        for s in seeds:
            run_experiment(
                run_seed=s,
                expr_data_path=expr_path,
                split_dir=split_dir,
                nan_cols=nan_cols,
                drop_genes=drop_genes,
                base_ontObj=BaseOntObj,
                model_dir=model_dir,
                top_thresh_ontobj=top_thresh_ontobj,
                bottom_thresh_ontobj=bottom_thresh_ontobj,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                kl_coeff=kl_coeff,
                id_col=id_col,
                label_col=label_col,
                donor_col=donor_col,
                subsample=args.subsample,
            )
