#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_donors(donors, split=0.7, seed=1):
    """
    Split unique donors into train/val/test.
    Returns arrays of donor IDs — all cells from a donor go to the same split.
    """
    donor_df = pd.DataFrame({'Donor ID': donors})

    train_donors, remaining = train_test_split(donor_df, train_size=split, random_state=seed)
    val_donors, test_donors = train_test_split(remaining, train_size=0.5, random_state=seed)

    return train_donors['Donor ID'].values, val_donors['Donor ID'].values, test_donors['Donor ID'].values


if __name__ == "__main__":

    DATA_PATH = os.path.join(os.getcwd(), 'data/SEAAD_slim.csv')
    SEEDS = [2, 3, 4, 5, 6]
    OUTPUT_DIR = os.path.join(os.getcwd(), 'splits_SEAAD')

    # only load the columns needed for splitting
    df = pd.read_csv(DATA_PATH, usecols=['sample_id', 'Donor ID', 'Subclass'])
    df = df.dropna(subset=['Subclass'])

    unique_donors = df['Donor ID'].unique()
    print(f"Total donors: {len(unique_donors)}, Total cells: {len(df)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for seed in SEEDS:
        run_dir = os.path.join(OUTPUT_DIR, f"run_seed-{seed}")
        os.makedirs(run_dir, exist_ok=True)

        train_donors, val_donors, test_donors = split_donors(unique_donors, seed=seed)

        # verify no donor leakage between splits
        assert len(set(train_donors) & set(test_donors)) == 0, "Donor leakage: train/test overlap"
        assert len(set(val_donors) & set(test_donors)) == 0, "Donor leakage: val/test overlap"
        assert len(set(train_donors) & set(val_donors)) == 0, "Donor leakage: train/val overlap"

        np.save(os.path.join(run_dir, 'train_ids.npy'), train_donors)
        np.save(os.path.join(run_dir, 'val_ids.npy'), val_donors)
        np.save(os.path.join(run_dir, 'test_ids.npy'), test_donors)

        # report donor and cell counts
        train_cells = df[df['Donor ID'].isin(train_donors)]
        val_cells = df[df['Donor ID'].isin(val_donors)]
        test_cells = df[df['Donor ID'].isin(test_donors)]

        print(f"Seed {seed}: "
              f"train={len(train_donors)} donors ({len(train_cells)} cells), "
              f"val={len(val_donors)} donors ({len(val_cells)} cells), "
              f"test={len(test_donors)} donors ({len(test_cells)} cells)")