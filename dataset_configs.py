TCGA = {
    'name': 'TCGA',
    'expr_data_path': 'data/TCGA_complete_bp_top1k.csv',
    'split_dir': 'splits_TCGA',
    'id_col': 'patient_id',
    'label_col': 'cancer_type',
    'donor_col': None,
    'nan_cols': ['sample_type', 'cancer_type', 'tumor_tissue_site', 'stage_pathologic_stage'],
    'drop_genes': [
        "A6NC42", "O60635", "O95857", "P00450", "P02814", "P05976", "P11686", "P12882",
        "P18283", "P22792", "P30408", "P48230", "Q4ZG55", "Q53G44", "Q5T7N2", "Q5TH69",
        "Q685J3", "Q7Z7J9", "Q86XP6", "Q8IWL1", "Q8IWL2", "Q8IZW8", "Q8WXD2", "Q969L2",
        "Q96KN4", "Q9BYZ8", "Q9NR99", "Q9NRC9", "Q9UKX2"
    ],
}

SEAAD = {
    'name': 'SEAAD',
    'expr_data_path': 'data/SEAAD_slim.csv',
    'split_dir': 'splits_SEAAD',
    'id_col': 'sample_id',
    'label_col': 'Subclass',
    'donor_col': 'Donor ID',
    'nan_cols': ['Donor ID', 'Subclass'],
    'drop_genes': None,
}

DATASETS = [TCGA, SEAAD]