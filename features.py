import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def engineer_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates ratio features: debt-to-income and credit-to-income.
    """
    df = df.copy()
    
    # 1. Debt-to-income ratio (Annuity relative to total income)
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        # Avoid division by zero
        income = df['AMT_INCOME_TOTAL'].replace(0, np.nan)
        df['debt_to_income_ratio'] = df['AMT_ANNUITY'] / income
        
    # 2. Credit-to-income ratio (Total credit amount relative to total income)
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        income = df['AMT_INCOME_TOTAL'].replace(0, np.nan)
        df['credit_to_income_ratio'] = df['AMT_CREDIT'] / income
        
    # 3. Payment rate (Annuity relative to total credit)
    if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
        credit = df['AMT_CREDIT'].replace(0, np.nan)
        df['payment_rate'] = df['AMT_ANNUITY'] / credit
        
    return df

def aggregate_bureau_defaults(bureau_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates prior default counts per client (SK_ID_CURR) from bureau.csv.
    A default is defined as having:
      - CREDIT_DAY_OVERDUE > 0, OR
      - AMT_CREDIT_SUM_OVERDUE > 0, OR
      - CREDIT_ACTIVE == 'Bad debt'
    """
    bureau_df = bureau_df.copy()
    
    # Flag each credit record if it meets default criteria
    is_overdue = bureau_df['CREDIT_DAY_OVERDUE'] > 0
    is_sum_overdue = bureau_df['AMT_CREDIT_SUM_OVERDUE'] > 0
    is_bad_debt = bureau_df['CREDIT_ACTIVE'] == 'Bad debt'
    
    bureau_df['is_default'] = (is_overdue | is_sum_overdue | is_bad_debt).astype(int)
    
    # Aggregate defaults count and total active credits count per client
    bureau_agg = bureau_df.groupby('SK_ID_CURR').agg(
        prior_default_counts=('is_default', 'sum'),
        total_bureau_credits=('SK_ID_BUREAU', 'count')
    ).reset_index()
    
    return bureau_agg

def merge_features(app_df: pd.DataFrame, bureau_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers application-level features and merges prior credit defaults from bureau data.
    """
    # 1. Engineer ratios
    app_featured = engineer_ratios(app_df)
    
    # 2. Aggregate and merge bureau features
    if bureau_df is not None and not bureau_df.empty:
        bureau_features = aggregate_bureau_defaults(bureau_df)
        app_merged = app_featured.merge(bureau_features, on='SK_ID_CURR', how='left')
        
        # For clients with no credit bureau history, fill default count with 0
        app_merged['prior_default_counts'] = app_merged['prior_default_counts'].fillna(0).astype(int)
        app_merged['total_bureau_credits'] = app_merged['total_bureau_credits'].fillna(0).astype(int)
    else:
        # Fallback if no bureau data is available
        app_merged = app_featured.copy()
        app_merged['prior_default_counts'] = 0
        app_merged['total_bureau_credits'] = 0
        
    return app_merged

class LeakageFreePreprocessor:
    """
    A class that manages preprocessing (imputation, encoding, scaling) fold-by-fold.
    Ensures that imputers, encoders, and scalers are fitted ONLY on the training split,
    avoiding data leakage from the validation or test splits.
    """
    def __init__(self, numerical_cols, categorical_cols):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        # Define numerical pipeline
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Define categorical pipeline
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, self.numerical_cols),
                ('cat', cat_pipeline, self.categorical_cols)
            ],
            remainder='drop'
        )
        
    def fit(self, X_train):
        """
        Fits the preprocessor on the training data.
        """
        self.preprocessor.fit(X_train)
        return self
        
    def transform(self, X):
        """
        Transforms the input data (can be train, validation, or test).
        """
        transformed_arr = self.preprocessor.transform(X)
        
        # Get feature names for transparency
        num_features = self.numerical_cols
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
        # Handle case when categorical features might be empty
        if len(self.categorical_cols) > 0:
            cat_features = list(cat_encoder.get_feature_names_out(self.categorical_cols))
        else:
            cat_features = []
            
        feature_names = num_features + cat_features
        
        # Recreate DataFrame
        transformed_df = pd.DataFrame(transformed_arr, columns=feature_names, index=X.index)
        return transformed_df
        
    def fit_transform(self, X_train):
        """
        Fits and transforms the training data.
        """
        return self.fit(X_train).transform(X_train)
