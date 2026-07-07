import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def clean_application_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the application dataframe by flagging and handling anomalies.
    Does NOT impute missing values (to prevent data leakage before splitting).
    """
    df = df.copy()
    
    # 1. Flag and handle DAYS_EMPLOYED anomaly
    # 365243 represents ~1000 years, likely a placeholder code for unemployed
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
        
    # 2. Convert DAYS_BIRTH to positive age in years for readability/interpretability
    if 'DAYS_BIRTH' in df.columns:
        df['DAYS_BIRTH_YEARS'] = -df['DAYS_BIRTH'] / 365.25
        
    return df

def load_data(data_dir: str = 'data') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train and test application data, applying basic cleaning and anomaly flagging.
    """
    train_path = os.path.join(data_dir, 'application_train.csv')
    test_path = os.path.join(data_dir, 'application_test.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at: {test_path}")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded train data: {train_df.shape}")
    print(f"Loaded test data: {test_df.shape}")
    
    # Apply cleaning
    train_cleaned = clean_application_data(train_df)
    test_cleaned = clean_application_data(test_df)
    
    return train_cleaned, test_cleaned

def create_stratified_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Adds a 'fold' column to the training dataframe using StratifiedKFold.
    Splits are stratified based on the TARGET column.
    """
    df = df.copy()
    if 'TARGET' not in df.columns:
        raise ValueError("DataFrame must contain a 'TARGET' column to perform stratified splitting.")
        
    df['fold'] = -1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Split using target
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df['TARGET'])):
        df.iloc[val_idx, df.columns.get_loc('fold')] = fold_idx
        
    print(f"Created {n_splits} stratified folds.")
    for f in range(n_splits):
        fold_size = (df['fold'] == f).sum()
        fold_defaults = df[df['fold'] == f]['TARGET'].sum()
        print(f"  Fold {f}: Size = {fold_size}, Defaults = {fold_defaults} ({fold_defaults/fold_size*100:.2f}%)")
        
    return df

if __name__ == '__main__':
    # Simple test run when executing module directly
    try:
        train, test = load_data()
        train_folds = create_stratified_folds(train)
    except Exception as e:
        print(f"Test run failed: {e}. Generate mock data first.")
