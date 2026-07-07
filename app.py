import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Loan Default Detector",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Theme State Management
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# 3. Inject CSS Design System
bg = "#09090b" if IS_DARK else "#ffffff"
bg_subtle = "#0c0c0f" if IS_DARK else "#f9fafb"
card = "#0c0c0f" if IS_DARK else "#ffffff"
card_hover = "#131316" if IS_DARK else "#f4f4f5"
border = "#1e1e24" if IS_DARK else "#e4e4e7"
border_subtle = "#16161a" if IS_DARK else "#f0f0f2"
text = "#fafafa" if IS_DARK else "#09090b"
text_muted = "#71717a"
text_dim = "#52525b" if IS_DARK else "#a1a1aa"
shadow = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"
green = "#22c55e" if IS_DARK else "#16a34a"
green_muted = "rgba(34,197,94,0.12)" if IS_DARK else "rgba(22,163,74,0.08)"
red = "#ef4444" if IS_DARK else "#dc2626"
red_muted = "rgba(239,68,68,0.12)" if IS_DARK else "rgba(220,38,38,0.08)"

css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

:root {{
    --bg: {bg};
    --bg-subtle: {bg_subtle};
    --card: {card};
    --card-hover: {card_hover};
    --border: {border};
    --border-subtle: {border_subtle};
    --text: {text};
    --text-muted: {text_muted};
    --text-dim: {text_dim};
    --accent: #2563eb;
    --accent-muted: #1d4ed8;
    --green: {green};
    --green-muted: {green_muted};
    --red: {red};
    --red-muted: {red_muted};
    --radius: 10px;
    --shadow: {shadow};
}}

/* Hide Streamlit chrome */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
div[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}

/* Global app styling */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}
.block-container {{
    padding: 1.5rem 2rem 2rem !important;
    max-width: 1320px !important;
}}

/* ===== LIGHT MODE TEXT FIX FOR ALL STREAMLIT WIDGETS ===== */
/* Labels */
label, .stSelectbox label, .stSlider label, .stNumberInput label,
.stCheckbox label, .stTextInput label, .stRadio label,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
.stMarkdown p, .stMarkdown span, .stCaption, p, span {{
    color: var(--text) !important;
}}
/* Selectbox / dropdown text */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stSelectbox"] div[data-baseweb="select"] span {{
    color: var(--text) !important;
}}
/* Number input text */
[data-testid="stNumberInput"] input {{
    color: var(--text) !important;
    background-color: var(--card) !important;
}}
/* Slider value labels */
[data-testid="stSlider"] div[data-testid="stTickBarMin"],
[data-testid="stSlider"] div[data-testid="stTickBarMax"],
[data-testid="stSlider"] [data-testid="stThumbValue"] {{
    color: var(--text) !important;
}}
/* Checkbox text */
.stCheckbox span {{
    color: var(--text) !important;
}}

/* Brand header */
.brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.5rem;
}}
.brand-symbol {{
    font-size: 1.6rem;
    color: var(--accent);
    font-weight: 700;
}}
.brand-name {{
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
}}

/* Custom container card */
.card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
}}
.card-title {{
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.5rem;
}}

/* Decision cards */
.decision-card {{
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    border: 1px solid transparent;
}}
.decision-approved {{
    background: var(--green-muted);
    border-color: var(--green);
    color: var(--green);
}}
.decision-denied {{
    background: var(--red-muted);
    border-color: var(--red);
    color: var(--red);
}}
.decision-title {{
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}}
.decision-text {{
    font-size: 0.85rem;
    color: var(--text);
    opacity: 0.9;
}}

/* Metric Card */
.metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.4rem;
    box-shadow: var(--shadow);
    text-align: center;
}}
.metric-label {{
    font-size: 0.78rem;
    color: var(--text-muted);
    font-weight: 500;
    margin-bottom: 0.25rem;
}}
.metric-value {{
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.03em;
}}

/* Check button styling */
.check-btn button {{
    background: var(--accent) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em;
    transition: background 0.2s ease;
}}
.check-btn button:hover {{
    background: var(--accent-muted) !important;
}}

/* Waiting state card */
.waiting-card {{
    background: var(--bg-subtle);
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    padding: 3rem 2rem;
    text-align: center;
}}
.waiting-icon {{
    font-size: 3rem;
    margin-bottom: 1rem;
}}
.waiting-title {{
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.5rem;
}}
.waiting-sub {{
    font-size: 0.82rem;
    color: var(--text-muted);
}}

/* Layout horizontal spacing */
[data-testid="stHorizontalBlock"] {{
    gap: 1.25rem !important;
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# 4. Model Loading with Cache
@st.cache_resource
def load_model_package():
    model_path = 'models/loan_default_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Trained model package not found at: {model_path}. Please run training first.")
        st.stop()
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model_package = load_model_package()
preprocessor = model_package['preprocessor']
model = model_package['model']
threshold = model_package['optimal_threshold']
metrics = model_package['metrics']

# 5. Header Bar
head_left, head_right = st.columns([8, 1])
with head_left:
    st.markdown("""
    <div class="brand">
        <span class="brand-symbol">◆</span>
        <span class="brand-name">Credit Default Risk Analyzer</span>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Light" if IS_DARK else "🌙 Dark"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

# 6. Page Layout
col_inputs, col_results = st.columns([5, 7])

# --- LEFT PANEL: INPUT FORM ---
with col_inputs:
    st.markdown('<div class="card"><div class="card-title">👤 Demographics & Employment</div>', unsafe_allow_html=True)
    c_demo1, c_demo2 = st.columns(2)
    with c_demo1:
        gender = st.selectbox("Gender", ["F", "M", "XNA"], index=0)
        own_car = st.selectbox("Owns Car", ["N", "Y"], index=0)
    with c_demo2:
        age_years = st.slider("Applicant Age (Years)", 18, 80, 38)
        own_realty = st.selectbox("Owns Real Estate", ["Y", "N"], index=0)
        
    emp_status = st.selectbox("Employment Status", ["Employed", "Retired / Unemployed"], index=0)
    if emp_status == "Employed":
        emp_years = st.slider("Employment Length (Years)", 0.0, 45.0, 4.5, step=0.5)
        days_employed = -emp_years * 365.25
        days_employed_anom = 0
    else:
        st.caption("Active Status: Unemployed/Retired. Days Employed placeholder auto-selected.")
        days_employed = np.nan
        days_employed_anom = 1
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">💵 Financial Assets & Requested Loan</div>', unsafe_allow_html=True)
    amt_income = st.number_input("Annual Income ($)", min_value=5000.0, max_value=2000000.0, value=65000.0, step=5000.0)
    amt_credit = st.number_input("Requested Credit Amount ($)", min_value=10000.0, max_value=5000000.0, value=250000.0, step=10000.0)
    amt_annuity = st.number_input("Expected Annual Annuity Payment ($)", min_value=1000.0, max_value=500000.0, value=12000.0, step=500.0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🛡️ Bureau History & Internal Risk Scores</div>', unsafe_allow_html=True)
    c_hist1, c_hist2 = st.columns(2)
    with c_hist1:
        prior_defaults = st.number_input("Credit Bureau Defaults Count", min_value=0, max_value=20, value=0, step=1)
    with c_hist2:
        total_bureau = st.number_input("Total Prior Credit Accounts", min_value=0, max_value=100, value=4, step=1)
        
    st.markdown("<p style='font-size: 0.8rem; font-weight: 500; margin-bottom: 0.5rem;'>External Source Risk Scores</p>", unsafe_allow_html=True)
    
    c_ext1, c_ext1_chk = st.columns([3, 2])
    with c_ext1_chk:
        ext1_nan = st.checkbox("Not Provided", key="ext1_nan", value=False)
    with c_ext1:
        ext_source_1 = np.nan if ext1_nan else st.slider("Ext Score 1", 0.0, 1.0, 0.52, step=0.01)
        
    c_ext2, c_ext2_chk = st.columns([3, 2])
    with c_ext2_chk:
        ext2_nan = st.checkbox("Not Provided", key="ext2_nan", value=False)
    with c_ext2:
        ext_source_2 = np.nan if ext2_nan else st.slider("Ext Score 2", 0.0, 1.0, 0.61, step=0.01)
        
    c_ext3, c_ext3_chk = st.columns([3, 2])
    with c_ext3_chk:
        ext3_nan = st.checkbox("Not Provided", key="ext3_nan", value=True)
    with c_ext3:
        ext_source_3 = np.nan if ext3_nan else st.slider("Ext Score 3", 0.0, 1.0, 0.44, step=0.01)
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== CHECK RISK BUTTON =====
    st.markdown('<div class="check-btn">', unsafe_allow_html=True)
    check_clicked = st.button("🔍  Check Risk", key="check_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- RIGHT PANEL: PREDICTION RESULTS ---
with col_results:
    if check_clicked:
        # A. Feature Engineering on inputs
        debt_to_income = amt_annuity / max(amt_income, 1.0)
        credit_to_income = amt_credit / max(amt_income, 1.0)
        payment_rate = amt_annuity / max(amt_credit, 1.0)
        
        input_data = {
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
        input_df = pd.DataFrame(input_data)
        
        # B. Preprocessing
        feature_names = list(preprocessor.get_feature_names_out())
        input_clean_arr = preprocessor.transform(input_df)
        input_clean_df = pd.DataFrame(input_clean_arr, columns=feature_names)
        
        # C. Prediction
        prob = model.predict_proba(input_clean_df)[0, 1]
        is_approved = prob < threshold
        
        # D. Decision Card Display
        if is_approved:
            st.markdown(f"""
            <div class="decision-card decision-approved">
                <div class="decision-title">✔ APPLICATION APPROVED</div>
                <div class="decision-text">
                    The applicant's credit risk profile is within acceptable parameters. 
                    Default probability of <b>{prob*100:.2f}%</b> is below the cost-optimal decision threshold of <b>{threshold*100:.1f}%</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="decision-card decision-denied">
                <div class="decision-title">✘ APPLICATION DENIED</div>
                <div class="decision-text">
                    The applicant's credit risk profile exceeds acceptable parameters.
                    Default probability of <b>{prob*100:.2f}%</b> is above or equal to the cost-optimal decision threshold of <b>{threshold*100:.1f}%</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # E. KPI Metric Cards row
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Default Risk Prob</div>
                <div class="metric-value">{prob*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Decision Threshold</div>
                <div class="metric-value">{threshold*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Business Savings</div>
                <div class="metric-value" style="color: var(--green);">$470K</div>
            </div>
            """, unsafe_allow_html=True)

        # F. SHAP Visual Explanations
        st.markdown('<div class="card" style="margin-top: 1.25rem;"><div class="card-title">🔍 Feature Impact Explanation (SHAP)</div>', unsafe_allow_html=True)
        
        # Configure Matplotlib styles based on theme
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.default'] = 'regular'
        
        if IS_DARK:
            plt.style.use('dark_background')
            text_color = '#fafafa'
        else:
            plt.style.use('default')
            text_color = '#09090b'
            
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['DM Sans', 'Arial'],
            'text.color': text_color,
            'axes.labelcolor': text_color,
            'xtick.color': text_color,
            'ytick.color': text_color
        })
        
        # Calculate SHAP Tree values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_clean_df)
        
        # Clean feature names for visualization
        cleaned_names = []
        for name in shap_values.feature_names:
            clean = name.replace("num__", "").replace("cat__", "")
            clean = clean.replace("_", " ").title()
            cleaned_names.append(clean)
        shap_values.feature_names = cleaned_names
        
        # Draw waterfall plot
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        shap.plots.waterfall(shap_values[0], max_display=7, show=False)
        
        # Customize plot
        plt.title("How features shifted prediction from base rate", fontsize=10, pad=15, color=text_color, fontweight='semibold')
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown("""
        <p style="font-size: 0.72rem; color: var(--text-dim); text-align: center; margin-top: 0.5rem;">
            Blue bars represent features decreasing default probability (lower risk). 
            Red bars represent features increasing default probability (higher risk).
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Show waiting state when no prediction has been made yet
        st.markdown("""
        <div class="waiting-card">
            <div class="waiting-icon">📋</div>
            <div class="waiting-title">Ready to Analyze</div>
            <div class="waiting-sub">Fill in the applicant details on the left, then click <b>Check Risk</b> to see the prediction results, risk metrics, and SHAP feature explanations.</div>
        </div>
        """, unsafe_allow_html=True)
