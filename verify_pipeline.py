import pandas as pd
import numpy as np
from data_loader import load_data, create_stratified_folds
from features import merge_features, LeakageFreePreprocessor

def verify_pipeline():
    print("=== START PIPELINE VERIFICATION ===")
    
    # 1. Load data
    train_raw, test_raw = load_data('data')
    
    # Check DAYS_EMPLOYED anomaly replacement
    assert train_raw['DAYS_EMPLOYED'].max() < 365243 or train_raw['DAYS_EMPLOYED'].isna().sum() > 0, "Anomaly 365243 not cleaned."
    assert 'DAYS_EMPLOYED_ANOM' in train_raw.columns, "Anomaly flag not created."
    print("[OK] Basic anomaly cleanup verified.")
    
    # 2. Stratified folds split
    train_folds = create_stratified_folds(train_raw, n_splits=5, seed=42)
    assert 'fold' in train_folds.columns, "Fold column missing."
    print("[OK] Stratified folds created.")
    
    # 3. Load bureau and merge features
    bureau_df = pd.read_csv('data/bureau.csv')
    train_featured = merge_features(train_folds, bureau_df)
    test_featured = merge_features(test_raw, bureau_df)
    
    assert 'debt_to_income_ratio' in train_featured.columns, "Debt-to-income ratio missing."
    assert 'credit_to_income_ratio' in train_featured.columns, "Credit-to-income ratio missing."
    assert 'prior_default_counts' in train_featured.columns, "Prior default counts missing."
    print("[OK] Feature engineering ratios and bureau defaults verified.")
    
    # 4. Leakage-free preprocessing on Fold 0
    fold_idx = 0
    train_split = train_featured[train_featured['fold'] != fold_idx]
    val_split = train_featured[train_featured['fold'] == fold_idx]
    
    # Features to preprocess
    numerical_cols = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'debt_to_income_ratio', 'credit_to_income_ratio', 'prior_default_counts'
    ]
    categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    
    # X and y
    X_train = train_split[numerical_cols + categorical_cols]
    y_train = train_split['TARGET']
    
    X_val = val_split[numerical_cols + categorical_cols]
    y_val = val_split['TARGET']
    
    print(f"\nProcessing Fold {fold_idx}:")
    print(f"  Train size: {X_train.shape}")
    print(f"  Val size:   {X_val.shape}")
    
    # Initialize preprocessor
    preprocessor = LeakageFreePreprocessor(numerical_cols, categorical_cols)
    
    # Fit & transform train
    X_train_clean = preprocessor.fit_transform(X_train)
    # Transform validation
    X_val_clean = preprocessor.transform(X_val)
    
    # Check shape compatibility
    assert X_train_clean.shape[1] == X_val_clean.shape[1], "Feature dimensions do not match between splits."
    assert not X_train_clean.isnull().any().any(), "Missing values remain in preprocessed training set."
    assert not X_val_clean.isnull().any().any(), "Missing values remain in preprocessed validation set."
    
    print("[OK] Leakage-free preprocessing (imputation & scaling) verified.")
    print("Preprocessed Training Features sample:")
    print(X_train_clean.head(2))
    
    print("\n=== PIPELINE VERIFICATION SUCCESSFUL ===")

if __name__ == '__main__':
    verify_pipeline()
