import os
import pandas as pd
import numpy as np

def generate_mock_data(seed=42):
    np.random.seed(seed)
    os.makedirs('data', exist_ok=True)
    
    # 1. Generate application_train
    n_train = 1000
    train_ids = np.arange(100001, 100001 + n_train)
    
    # Target is imbalanced (approx 8% default rate)
    target = np.random.choice([0, 1], size=n_train, p=[0.92, 0.08])
    
    # Days birth (e.g. 20 to 68 years old in days)
    days_birth = np.random.randint(-25000, -7000, size=n_train)
    
    # Days employed - containing the anomaly 365243 for some unemployed
    days_employed = []
    for db in days_birth:
        is_anom = np.random.rand() < 0.15  # 15% anomaly rate
        if is_anom:
            days_employed.append(365243)
        else:
            # Must be less than days_birth in magnitude
            max_work = int(-db * 0.8)
            days_employed.append(np.random.randint(-max_work, -100))
    days_employed = np.array(days_employed)
    
    # Ratios values
    amt_income = np.random.exponential(scale=150000, size=n_train) + 50000
    amt_credit = amt_income * np.random.uniform(1.5, 6.0, size=n_train)
    amt_annuity = amt_credit * np.random.uniform(0.04, 0.12, size=n_train)
    
    # Randomly inject missing values
    ext_1 = np.random.uniform(0, 1, size=n_train)
    ext_2 = np.random.uniform(0, 1, size=n_train)
    ext_3 = np.random.uniform(0, 1, size=n_train)
    ext_1[np.random.rand(n_train) < 0.5] = np.nan
    ext_2[np.random.rand(n_train) < 0.1] = np.nan
    ext_3[np.random.rand(n_train) < 0.2] = np.nan
    
    gender = np.random.choice(['F', 'M', 'XNA'], size=n_train, p=[0.65, 0.349, 0.001])
    own_car = np.random.choice(['Y', 'N'], size=n_train)
    own_realty = np.random.choice(['Y', 'N'], size=n_train)
    
    df_train = pd.DataFrame({
        'SK_ID_CURR': train_ids,
        'TARGET': target,
        'CODE_GENDER': gender,
        'FLAG_OWN_CAR': own_car,
        'FLAG_OWN_REALTY': own_realty,
        'AMT_INCOME_TOTAL': amt_income,
        'AMT_CREDIT': amt_credit,
        'AMT_ANNUITY': amt_annuity,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'EXT_SOURCE_1': ext_1,
        'EXT_SOURCE_2': ext_2,
        'EXT_SOURCE_3': ext_3
    })
    
    # 2. Generate application_test
    n_test = 200
    test_ids = np.arange(200001, 200001 + n_test)
    
    days_birth_test = np.random.randint(-25000, -7000, size=n_test)
    days_employed_test = []
    for db in days_birth_test:
        is_anom = np.random.rand() < 0.15
        if is_anom:
            days_employed_test.append(365243)
        else:
            max_work = int(-db * 0.8)
            days_employed_test.append(np.random.randint(-max_work, -100))
    days_employed_test = np.array(days_employed_test)
    
    amt_income_test = np.random.exponential(scale=150000, size=n_test) + 50000
    amt_credit_test = amt_income_test * np.random.uniform(1.5, 6.0, size=n_test)
    amt_annuity_test = amt_credit_test * np.random.uniform(0.04, 0.12, size=n_test)
    
    ext_1_test = np.random.uniform(0, 1, size=n_test)
    ext_2_test = np.random.uniform(0, 1, size=n_test)
    ext_3_test = np.random.uniform(0, 1, size=n_test)
    ext_1_test[np.random.rand(n_test) < 0.5] = np.nan
    ext_2_test[np.random.rand(n_test) < 0.1] = np.nan
    ext_3_test[np.random.rand(n_test) < 0.2] = np.nan
    
    df_test = pd.DataFrame({
        'SK_ID_CURR': test_ids,
        'CODE_GENDER': np.random.choice(['F', 'M', 'XNA'], size=n_test, p=[0.65, 0.35, 0.0]),
        'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], size=n_test),
        'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], size=n_test),
        'AMT_INCOME_TOTAL': amt_income_test,
        'AMT_CREDIT': amt_credit_test,
        'AMT_ANNUITY': amt_annuity_test,
        'DAYS_BIRTH': days_birth_test,
        'DAYS_EMPLOYED': days_employed_test,
        'EXT_SOURCE_1': ext_1_test,
        'EXT_SOURCE_2': ext_2_test,
        'EXT_SOURCE_3': ext_3_test
    })
    
    # 3. Generate bureau data
    # Each client in train/test has a random number of prior credits (e.g. 0 to 5)
    bureau_rows = []
    bureau_id = 5000001
    
    all_ids = np.concatenate([train_ids, test_ids])
    for curr_id in all_ids:
        n_credits = np.random.randint(0, 6)
        for _ in range(n_credits):
            # Prior defaults indicators
            # We want some active credits, some closed, and some with overdue days or overdue amounts
            active = np.random.choice(['Closed', 'Active', 'Sold', 'Bad debt'], p=[0.6, 0.37, 0.02, 0.01])
            
            # Days credit
            days_credit = np.random.randint(-3000, -10)
            
            # Defaults indicator logic: Day overdue or sum overdue
            has_overdue = np.random.rand() < 0.05  # 5% default rate on credit records
            if has_overdue:
                credit_day_overdue = np.random.randint(1, 120)
                amt_credit_sum_overdue = np.random.exponential(scale=5000)
                if active == 'Closed':
                    active = 'Active' # overdue means active usually
            else:
                credit_day_overdue = 0
                amt_credit_sum_overdue = 0.0
                
            bureau_rows.append({
                'SK_ID_CURR': curr_id,
                'SK_ID_BUREAU': bureau_id,
                'CREDIT_ACTIVE': active,
                'DAYS_CREDIT': days_credit,
                'CREDIT_DAY_OVERDUE': credit_day_overdue,
                'AMT_CREDIT_SUM_OVERDUE': amt_credit_sum_overdue,
                'AMT_CREDIT_SUM': np.random.exponential(scale=100000) + 10000
            })
            bureau_id += 1
            
    df_bureau = pd.DataFrame(bureau_rows)
    
    # Save files
    df_train.to_csv('data/application_train.csv', index=False)
    df_test.to_csv('data/application_test.csv', index=False)
    df_bureau.to_csv('data/bureau.csv', index=False)
    
    print("Mock datasets generated successfully in 'data/' directory:")
    print(f"- application_train.csv: {df_train.shape}")
    print(f"- application_test.csv: {df_test.shape}")
    print(f"- bureau.csv: {df_bureau.shape}")

if __name__ == '__main__':
    generate_mock_data()
