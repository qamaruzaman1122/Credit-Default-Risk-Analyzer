# ◆ Credit Default Risk Analyzer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.59-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-4.6-2980B9?style=for-the-badge" />
  <img src="https://img.shields.io/badge/SHAP-0.52-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Optuna-4.9-3F51B5?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p align="center">
  <b>An end-to-end ML pipeline & interactive web app for credit default risk assessment,<br/>
  featuring cost-optimized decision thresholds and real-time SHAP explainability.</b>
</p>

---

## 🎯 Overview

This project tackles the **Home Credit Default Risk** problem — predicting whether a loan applicant is likely to default. It goes beyond a simple classifier by incorporating:

- **Business cost optimization** — a custom cost function (`$10K × FN + $2K × FP`) to find the threshold that minimizes real-world financial loss
- **Explainable AI** — SHAP waterfall plots show _exactly why_ each application was approved or denied
- **Interactive web interface** — a polished Streamlit dashboard with dark/light themes and one-click risk assessment

---

## 🏗️ Architecture & Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                             │
├──────────────┬───────────────┬───────────────────────────────┤
│  Raw Data    │  Feature Eng  │  Leakage-Free Preprocessing   │
│  (CSV)       │  • Ratios     │  • Median Imputation          │
│              │  • Bureau     │  • Standard Scaling            │
│              │    Defaults   │  • One-Hot Encoding            │
├──────────────┴───────────────┴───────────────────────────────┤
│                    MODEL TRAINING                            │
├──────────────┬───────────────┬───────────────────────────────┤
│  Baseline    │  Optuna HPO   │  Cost-Optimal Threshold       │
│  Logistic    │  (50 trials)  │  Grid Search                  │
│  Regression  │  LightGBM     │  over [0, 1]                  │
├──────────────┴───────────────┴───────────────────────────────┤
│                    SERVING                                   │
├──────────────────────────────────────────────────────────────┤
│  Streamlit App → Preprocessor → Model → SHAP Explainer      │
│  Dark/Light Theme  •  Check Risk Button  •  Waterfall Plot   │
└──────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
Loan-Default-Risk-Analyzer/
│
├── app.py                    # Streamlit web application
├── web_app.py                # Flask-based alternative web app
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── .gitignore                # Git ignore rules
│
├── src/                      # Source modules
│   ├── data_loader.py        # Data loading & anomaly cleaning
│   ├── features.py           # Feature engineering & preprocessing
│   ├── train.py              # Full training pipeline (LR + LightGBM + Optuna)
│   ├── verify_pipeline.py    # Pipeline verification script
│   ├── generate_mock_data.py # Mock data generator for testing
│   └── create_notebook.py    # EDA notebook generator
│
├── models/                   # Saved model artifacts
│   └── loan_default_model.pkl
│
├── data/                     # Dataset files (not committed to Git)
│   ├── application_train.csv
│   ├── application_test.csv
│   └── bureau.csv
│
├── templates/                # HTML templates (Flask app)
│   └── index.html
│
└── eda_notebook.ipynb        # Exploratory Data Analysis notebook
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Loan-Default-Risk-Analyzer.git
cd Loan-Default-Risk-Analyzer
```

### 2. Create a virtual environment & install dependencies
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Prepare the data
Place your dataset CSV files (`application_train.csv`, `application_test.csv`, `bureau.csv`) inside the `data/` folder.

> **Note:** Sample mock data can be generated using:
> ```bash
> python src/generate_mock_data.py
> ```

### 4. Train the model
```bash
python src/train.py
```

This runs the full pipeline:
- Loads & cleans data
- Engineers features (ratios, bureau defaults)
- Trains Logistic Regression baseline (5-fold CV)
- Tunes LightGBM via Optuna (50 trials, 5-fold CV)
- Finds cost-optimal decision threshold
- Saves final model to `models/loan_default_model.pkl`

### 5. Launch the web app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Model Performance

| Metric | Logistic Regression | LightGBM (Tuned) |
|:-------|:-------------------:|:-----------------:|
| **ROC-AUC** | 0.5005 | **0.6685** |
| **PR-AUC** | 0.0787 | **0.1295** |
| **Cost @ Default (0.5)** | $1,222,000 | **$790,000** |
| **Optimal Threshold** | 0.794 | **0.142** |
| **Cost @ Optimal** | $790,000 | **$790,000** |

### 💰 Business Impact
The cost-optimized LightGBM model delivers up to **$470K in savings** compared to the naïve Logistic Regression baseline, by intelligently balancing false negatives ($10K each) against false positives ($2K each).

---

## 🔍 Key Features

### 🎨 Interactive Dashboard
- **Dark & Light themes** with full widget text color adaptation
- **One-click "Check Risk" button** — results appear only after user submits
- **Clean waiting state** when no prediction has been made

### 🤖 ML Pipeline
- **Leakage-free preprocessing** — imputers/scalers fitted only on training folds
- **Optuna hyperparameter optimization** — 50-trial Bayesian search
- **Stratified 5-fold cross-validation** — robust OOF evaluation
- **Class imbalance handling** — `is_unbalance` / `scale_pos_weight` search

### 📈 Explainability
- **SHAP TreeExplainer** for instant feature attribution
- **Waterfall plots** showing how each feature shifts the base prediction
- Clean, readable feature labels with theme-aware colors

---

## 🛠️ Tech Stack

| Category | Technologies |
|:---------|:-------------|
| **ML Framework** | scikit-learn, LightGBM, Optuna |
| **Explainability** | SHAP |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Web App** | Streamlit, Flask |
| **Language** | Python 3.10+ |

---

## 🌐 Deployment (Streamlit Cloud)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** → select your repo, branch `main`, and set main file to `app.py`
4. Click **Deploy!** — auto-syncs on every push

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ using Python, LightGBM, SHAP, and Streamlit</sub>
</p>

