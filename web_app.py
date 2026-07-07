import os
import sys
import pickle
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify, render_template

# Ensure src/ is in the python path to load custom preprocessing classes from model package
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

app = Flask(__name__, template_folder='templates')

# Global variables for model artifacts
preprocessor = None
model = None
threshold = None
explainer = None

def init_model():
    global preprocessor, model, threshold, explainer
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'loan_default_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at: {model_path}. Please run 'python src/train.py' first.")
        
    print(f"Loading model artifacts from {model_path}...")
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
        
    preprocessor = model_package['preprocessor']
    model = model_package['model']
    threshold = model_package['optimal_threshold']
    
    # Initialize TreeExplainer for SHAP explanations
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    print("Model and explainer ready.")

# Serve main webpage
@app.route('/')
def index():
    return render_template('index.html')

# JSON API endpoint for evaluations
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Demographics & Basic Assets
        gender = data['CODE_GENDER']
        own_car = data['FLAG_OWN_CAR']
        own_realty = data['FLAG_OWN_REALTY']
        age_years = float(data['DAYS_BIRTH_YEARS'])
        
        # Employment Status Conversion
        emp_status = data['DAYS_EMPLOYED_STATUS']
        if emp_status == 'Retired':
            days_employed = np.nan
            days_employed_anom = 1
        else:
            days_employed = -float(data['DAYS_EMPLOYED_YEARS']) * 365.25
            days_employed_anom = 0
            
        # Bureau Records
        prior_defaults = int(data['prior_default_counts'])
        total_bureau = int(data['total_bureau_credits'])
        
        # Finances
        amt_income = float(data['AMT_INCOME_TOTAL'])
        amt_credit = float(data['AMT_CREDIT'])
        amt_annuity = float(data['AMT_ANNUITY'])
        
        # External scores with missing support
        ext_source_1 = np.nan if data['ext1_nan'] else float(data['ext_source_1'])
        ext_source_2 = np.nan if data['ext2_nan'] else float(data['ext_source_2'])
        ext_source_3 = np.nan if data['ext3_nan'] else float(data['ext_source_3'])
        
        # Real-time Feature Engineering (Ratios)
        debt_to_income = amt_annuity / max(amt_income, 1.0)
        credit_to_income = amt_credit / max(amt_income, 1.0)
        payment_rate = amt_annuity / max(amt_credit, 1.0)
        
        # Build input record DataFrame
        input_dict = {
            'AMT_INCOME_TOTAL': [amt_income],
            'AMT_CREDIT': [amt_credit],
            'AMT_ANNUITY': [amt_annuity],
            'DAYS_BIRTH_YEARS': [age_years],
            'DAYS_EMPLOYED': [days_employed],
            'DAYS_EMPLOYED_ANOM': [days_employed_anom],
            'EXT_SOURCE_1': [ext_source_1],
            'EXT_SOURCE_2': [ext_source_2],
            'EXT_SOURCE_3': [ext_source_3],
            'debt_to_income_ratio': [debt_to_income],
            'credit_to_income_ratio': [credit_to_income],
            'payment_rate': [payment_rate],
            'prior_default_counts': [prior_defaults],
            'total_bureau_credits': [total_bureau],
            'CODE_GENDER': [gender],
            'FLAG_OWN_CAR': [own_car],
            'FLAG_OWN_REALTY': [own_realty],
        }
        input_df = pd.DataFrame(input_dict)
        
        # Transform inputs using model preprocessor
        feature_names = list(preprocessor.get_feature_names_out())
        input_clean_arr = preprocessor.transform(input_df)
        input_clean_df = pd.DataFrame(input_clean_arr, columns=feature_names)
        
        # Evaluate model prediction
        prob = float(model.predict_proba(input_clean_df)[0, 1])
        decision = "Approved" if prob < threshold else "Denied"
        
        # Calculate local SHAP impact
        shap_values = explainer(input_clean_df)
        
        # Format SHAP values for front-end ApexCharts drawing
        shap_list = []
        for i, name in enumerate(shap_values.feature_names):
            clean_name = name.replace("num__", "").replace("cat__", "")
            clean_name = clean_name.replace("_", " ").title()
            val = float(shap_values.values[0, i])
            shap_list.append({
                'feature': clean_name,
                'value': val
            })
            
        # Sort features by absolute impact and return the top 8
        shap_list = sorted(shap_list, key=lambda x: abs(x['value']), reverse=True)[:8]
        
        return jsonify({
            'probability': prob,
            'threshold': threshold,
            'decision': decision,
            'shap_values': shap_list
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize model artifacts on startup
    init_model()
    # Run locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
