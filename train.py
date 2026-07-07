import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import lightgbm as lgb
import optuna

# Silence Optuna logs to keep terminal output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Ensure src/ is in the python path for importing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data, create_stratified_folds
from features import merge_features, LeakageFreePreprocessor

def compute_pr_auc(y_true, y_prob):
    """
    Computes the Area Under the Precision-Recall Curve (PR-AUC).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)

def compute_cost(y_true, y_prob, threshold):
    """
    Computes the total cost based on the threshold.
    $10,000 loss per False Negative (FN).
    $2,000 lost profit per False Positive (FP).
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    cost = 10000 * fn + 2000 * fp
    return cost, fp, fn

def find_optimal_threshold(y_true, oof_probs, label="Model"):
    """
    Performs grid search to find the cost-optimal decision threshold.
    """
    best_threshold = 0.5
    min_cost = float('inf')
    best_fp = 0
    best_fn = 0
    
    thresholds = np.linspace(0.0, 1.0, 1001)
    for t in thresholds:
        cost, fp, fn = compute_cost(y_true, oof_probs, t)
        if cost < min_cost:
            min_cost = cost
            best_threshold = t
            best_fp = fp
            best_fn = fn
            
    print(f"\n{label} Threshold Optimization:")
    default_cost, default_fp, default_fn = compute_cost(y_true, oof_probs, 0.5)
    print(f"  Default Threshold (0.500) Cost: ${default_cost:,} (FPs: {default_fp}, FNs: {default_fn})")
    print(f"  Optimal Threshold ({best_threshold:.3f}) Cost: ${min_cost:,} (FPs: {best_fp}, FNs: {best_fn})")
    print(f"  Savings: ${default_cost - min_cost:,}")
    
    return best_threshold, min_cost

def train_logistic_regression(train_df, numerical_cols, categorical_cols):
    """
    Trains a baseline Logistic Regression model using stratified 5-fold CV.
    """
    print("\n--- Training Baseline Logistic Regression (Stratified 5-Fold CV) ---")
    oof_probs = np.zeros(len(train_df))
    
    for fold_idx in range(5):
        train_split = train_df[train_df['fold'] != fold_idx]
        val_split = train_df[train_df['fold'] == fold_idx]
        
        X_train = train_split[numerical_cols + categorical_cols]
        y_train = train_split['TARGET']
        X_val = val_split[numerical_cols + categorical_cols]
        y_val = val_split['TARGET']
        
        # Leakage-free preprocessing
        preprocessor = LeakageFreePreprocessor(numerical_cols, categorical_cols)
        X_train_clean = preprocessor.fit_transform(X_train)
        X_val_clean = preprocessor.transform(X_val)
        
        # Logistic Regression with class weight balancing for class imbalance
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(X_train_clean, y_train)
        
        val_idx_in_df = val_split.index
        oof_probs[val_idx_in_df] = model.predict_proba(X_val_clean)[:, 1]
        
    y_true = train_df['TARGET'].values
    roc_auc = roc_auc_score(y_true, oof_probs)
    pr_auc = compute_pr_auc(y_true, oof_probs)
    
    print(f"Logistic Regression OOF ROC-AUC: {roc_auc:.4f}")
    print(f"Logistic Regression OOF PR-AUC: {pr_auc:.4f}")
    
    return oof_probs, roc_auc, pr_auc

def train_lightgbm_cv(train_df, numerical_cols, categorical_cols, params):
    """
    Helper function to run cross-validation for a specific set of LightGBM parameters.
    """
    oof_probs = np.zeros(len(train_df))
    
    for fold_idx in range(5):
        train_split = train_df[train_df['fold'] != fold_idx]
        val_split = train_df[train_df['fold'] == fold_idx]
        
        X_train = train_split[numerical_cols + categorical_cols]
        y_train = train_split['TARGET']
        X_val = val_split[numerical_cols + categorical_cols]
        y_val = val_split['TARGET']
        
        # Leakage-free preprocessing
        preprocessor = LeakageFreePreprocessor(numerical_cols, categorical_cols)
        X_train_clean = preprocessor.fit_transform(X_train)
        X_val_clean = preprocessor.transform(X_val)
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_clean, 
            y_train,
            eval_set=[(X_val_clean, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )
        
        val_idx_in_df = val_split.index
        oof_probs[val_idx_in_df] = model.predict_proba(X_val_clean)[:, 1]
        
    return oof_probs

def tune_lightgbm(train_df, numerical_cols, categorical_cols, n_trials=50):
    """
    Tunes LightGBM hyperparameters using Optuna over 5-fold CV to maximize ROC-AUC.
    """
    print(f"\n--- Tuning LightGBM with Optuna ({n_trials} Trials) ---")
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'random_state': 42,
            'verbose': -1,
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        
        # Hyperparameter search space for handling class imbalance
        imbalance_handling = trial.suggest_categorical('imbalance_handling', ['none', 'is_unbalance', 'scale_pos_weight'])
        if imbalance_handling == 'is_unbalance':
            params['is_unbalance'] = True
        elif imbalance_handling == 'scale_pos_weight':
            params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 5.0, 15.0)
            
        oof_probs = train_lightgbm_cv(train_df, numerical_cols, categorical_cols, params)
        y_true = train_df['TARGET'].values
        return roc_auc_score(y_true, oof_probs)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("\nOptuna Search Complete.")
    print(f"Best Trial ROC-AUC: {study.best_trial.value:.4f}")
    
    # Map trial params back to LightGBM parameters
    best_params = study.best_trial.params
    best_lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 42,
        'verbose': -1,
    }
    for k, v in best_params.items():
        if k == 'imbalance_handling':
            continue
        best_lgb_params[k] = v
        
    if best_params['imbalance_handling'] == 'is_unbalance':
        best_lgb_params['is_unbalance'] = True
    elif best_params['imbalance_handling'] == 'none':
        # Ensure no scale_pos_weight is passed if imbalance handling was 'none'
        best_lgb_params.pop('scale_pos_weight', None)
        
    return best_lgb_params

def main():
    # 1. Load Data
    print("Loading data...")
    train_raw, test_raw = load_data('data')
    
    # 2. Stratified folds split
    train_folds = create_stratified_folds(train_raw, n_splits=5, seed=42)
    
    # 3. Load bureau and merge features
    print("Loading bureau features...")
    bureau_df = pd.read_csv('data/bureau.csv')
    train_featured = merge_features(train_folds, bureau_df)
    test_featured = merge_features(test_raw, bureau_df)
    
    # Define features to use
    numerical_cols = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
        'DAYS_BIRTH_YEARS', 'DAYS_EMPLOYED', 'DAYS_EMPLOYED_ANOM',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'debt_to_income_ratio', 'credit_to_income_ratio', 'payment_rate',
        'prior_default_counts', 'total_bureau_credits'
    ]
    categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    
    # Ensure target labels are binary 0 and 1
    y_true = train_featured['TARGET'].values
    
    # 4. Train Logistic Regression baseline
    lr_oof, lr_roc_auc, lr_pr_auc = train_logistic_regression(train_featured, numerical_cols, categorical_cols)
    lr_opt_threshold, lr_min_cost = find_optimal_threshold(y_true, lr_oof, "Logistic Regression Baseline")
    
    # 5. Tune LightGBM with Optuna
    best_lgb_params = tune_lightgbm(train_featured, numerical_cols, categorical_cols, n_trials=50)
    
    print("\nBest LightGBM Parameters:")
    for k, v in best_lgb_params.items():
        print(f"  {k}: {v}")
        
    # 6. Evaluate LightGBM via cross-validation using the best parameters
    print("\nEvaluating Best LightGBM Model with 5-Fold CV...")
    lgb_oof = train_lightgbm_cv(train_featured, numerical_cols, categorical_cols, best_lgb_params)
    lgb_roc_auc = roc_auc_score(y_true, lgb_oof)
    lgb_pr_auc = compute_pr_auc(y_true, lgb_oof)
    print(f"LightGBM OOF ROC-AUC: {lgb_roc_auc:.4f}")
    print(f"LightGBM OOF PR-AUC: {lgb_pr_auc:.4f}")
    
    # Find LightGBM cost-optimal threshold
    lgb_opt_threshold, lgb_min_cost = find_optimal_threshold(y_true, lgb_oof, "LightGBM")
    
    # 7. Print side-by-side performance summary table
    print("\n" + "="*50)
    print("               MODEL COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Metric':<25} | {'Logistic Regression':<20} | {'LightGBM':<20}")
    print("-"*72)
    print(f"{'ROC-AUC':<25} | {lr_roc_auc:<20.4f} | {lgb_roc_auc:<20.4f}")
    print(f"{'PR-AUC':<25} | {lr_pr_auc:<20.4f} | {lgb_pr_auc:<20.4f}")
    
    lr_def_cost, _, _ = compute_cost(y_true, lr_oof, 0.5)
    lgb_def_cost, _, _ = compute_cost(y_true, lgb_oof, 0.5)
    print(f"{'Cost at Default Thresh 0.5':<25} | ${lr_def_cost:<19,} | ${lgb_def_cost:<19,}")
    print(f"{'Optimal Decision Thresh':<25} | {lr_opt_threshold:<20.3f} | {lgb_opt_threshold:<20.3f}")
    print(f"{'Cost at Optimal Thresh':<25} | ${lr_min_cost:<19,} | ${lgb_min_cost:<19,}")
    print("="*50)
    
    # 8. Train Final LightGBM model on all training data
    print("\n--- Training Final LightGBM Model on Full Training Set ---")
    X_train_full = train_featured[numerical_cols + categorical_cols]
    y_train_full = train_featured['TARGET']
    
    preprocessor = LeakageFreePreprocessor(numerical_cols, categorical_cols)
    X_train_full_clean = preprocessor.fit_transform(X_train_full)
    
    final_model = lgb.LGBMClassifier(**best_lgb_params)
    final_model.fit(X_train_full_clean, y_train_full)
    
    # 9. Save the full model package
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'loan_default_model.pkl')
    
    model_data = {
        'preprocessor': preprocessor.preprocessor,
        'model': final_model,
        'optimal_threshold': lgb_opt_threshold,
        'best_params': best_lgb_params,
        'metrics': {
            'lr_oof_roc_auc': lr_roc_auc,
            'lr_oof_pr_auc': lr_pr_auc,
            'lgb_oof_roc_auc': lgb_roc_auc,
            'lgb_oof_pr_auc': lgb_pr_auc,
            'lgb_optimal_threshold': lgb_opt_threshold,
            'lgb_optimal_cost': lgb_min_cost,
            'lr_optimal_threshold': lr_opt_threshold,
            'lr_optimal_cost': lr_min_cost,
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"\nFinal model pipeline saved to: {model_path}")
    print("Execution complete.")

if __name__ == '__main__':
    main()
