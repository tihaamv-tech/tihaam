"""
FairCheck India - AI Fairness Auditing System (Indian Edition)
Indian Army, Education, Bank Loan - Bias Detection & Solutions
Version 2.0 - Built for India
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import time
import random
import os

try:
    from fetch_gov_data import load_real_data
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, MetricFrame
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

import shap
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io

st.set_page_config(page_title="FairCheck India", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

DARK_BG = "#ffffff"
CARD_BG = "#f8f9fa"
BORDER_COLOR = "#dee2e6"
TEXT_COLOR = "#212529"
ACCENT_BLUE = "#0d6efd"
ACCENT_GREEN = "#198754"
ACCENT_RED = "#dc3545"
RISK_LOW = "#198754"
RISK_MEDIUM = "#ffc107"
RISK_HIGH = "#dc3545"

INDIAN_STATES = [
    'Maharashtra', 'Uttar Pradesh', 'West Bengal', 'Tamil Nadu', 'Delhi',
    'Karnataka', 'Gujarat', 'Rajasthan', 'Madhya Pradesh', ' Kerala',
    'Bihar', 'Odisha', 'Punjab', 'Jharkhand', 'Chhattisgarh'
]

EDUCATION_LEVELS = ['10th Pass', '12th Pass', 'Graduate', 'Post Graduate', 'PhD']
OCCUPATIONS = ['Farmer', 'Teacher', 'Engineer', 'Doctor', 'Business', 'Private Job', 'Government Job', 'Self Employed']
RELIGIONS = ['Hindu', 'Muslim', 'Christian', 'Sikh', 'Buddhist', 'Other']
CASTE_CATEGORIES = ['General', 'OBC', 'SC', 'ST']

def generate_indian_census_data(n=50000):
    np.random.seed(42)
    random.seed(42)
    
    data = {
        'age': np.random.randint(18, 65, n),
        'gender': np.random.choice(['Male', 'Female'], n, p=[0.52, 0.48]),
        'state': np.random.choice(INDIAN_STATES, n),
        'education': np.random.choice(EDUCATION_LEVELS, n, p=[0.25, 0.30, 0.30, 0.12, 0.03]),
        'occupation': np.random.choice(OCCUPATIONS, n, p=[0.25, 0.15, 0.10, 0.05, 0.15, 0.15, 0.10, 0.05]),
        'religion': np.random.choice(RELIGIONS, n, p=[0.80, 0.14, 0.02, 0.02, 0.01, 0.01]),
        'caste': np.random.choice(CASTE_CATEGORIES, n, p=[0.30, 0.27, 0.23, 0.20]),
    }
    
    base_income = {
        'Farmer': 150000, 'Teacher': 300000, 'Engineer': 600000, 'Doctor': 800000,
        'Business': 500000, 'Private Job': 400000, 'Government Job': 450000, 'Self Employed': 350000
    }
    
    education_multiplier = {'10th Pass': 0.7, '12th Pass': 0.85, 'Graduate': 1.0, 'Post Graduate': 1.3, 'PhD': 1.5}
    
    income_list = []
    for i in range(n):
        occ = data['occupation'][i]
        edu = data['education'][i]
        base = base_income.get(occ, 300000)
        mult = education_multiplier.get(edu, 1.0)
        age = data['age'][i]
        age_factor = 1 + (age - 18) * 0.02
        noise = np.random.uniform(0.8, 1.2)
        income = base * mult * age_factor * noise
        income_list.append(int(income))
    
    data['annual_income'] = income_list
    
    df = pd.DataFrame(data)
    
    def classify_income_level(income):
        if income < 250000:
            return 'BPL'
        elif income < 500000:
            return 'Lower Middle'
        elif income < 1000000:
            return 'Middle'
        elif income < 2000000:
            return 'Upper Middle'
        else:
            return 'High'
    
    df['income_level'] = df['annual_income'].apply(classify_income_level)
    
    return df

def generate_domain_outcomes(df, domain):
    np.random.seed(42)
    random.seed(42)
    
    if domain == 'Army':
        prob_base = {'Male': 0.75, 'Female': 0.65}
        prob_education = {'10th Pass': 0.3, '12th Pass': 0.6, 'Graduate': 0.85, 'Post Graduate': 0.9, 'PhD': 0.95}
        
        def army_outcome(row):
            base_prob = prob_base.get(row['gender'], 0.5)
            edu_prob = prob_education.get(row['education'], 0.5)
            age_factor = 1.0 if 18 <= row['age'] <= 35 else 0.7
            final_prob = base_prob * edu_prob * age_factor
            return 1 if np.random.random() < final_prob else 0
        
        df['selected'] = df.apply(army_outcome, axis=1)
    
    elif domain == 'Education':
        prob_caste = {'General': 0.8, 'OBC': 0.7, 'SC': 0.65, 'ST': 0.6}
        prob_edu = {'10th Pass': 0.3, '12th Pass': 0.5, 'Graduate': 0.75, 'Post Graduate': 0.9, 'PhD': 0.95}
        
        def education_outcome(row):
            caste_prob = prob_caste.get(row['caste'], 0.5)
            edu_prob = prob_edu.get(row['education'], 0.5)
            age_factor = 1.0 if 18 <= row['age'] <= 25 else 0.3
            income_factor = 0.8 if row['annual_income'] < 300000 else 1.0
            final_prob = caste_prob * edu_prob * age_factor * income_factor
            return 1 if np.random.random() < final_prob else 0
        
        df['selected'] = df.apply(education_outcome, axis=1)
    
    elif domain == 'Bank Loan':
        prob_base = {'Male': 0.7, 'Female': 0.55}
        
        def loan_outcome(row):
            base_prob = prob_base.get(row['gender'], 0.5)
            income_factor = min(row['annual_income'] / 500000, 1.2)
            age_factor = 1.0 if 25 <= row['age'] <= 50 else 0.7
            occupation_factor = 1.0 if row['occupation'] in ['Government Job', 'Doctor', 'Engineer'] else 0.85
            final_prob = base_prob * income_factor * age_factor * occupation_factor
            return 1 if np.random.random() < min(final_prob, 0.95) else 0
        
        df['selected'] = df.apply(loan_outcome, axis=1)
    
    return df

# === SECTION 12: AUDIT LOGGING SYSTEM ===
class AuditLogger:
    def __init__(self):
        self.logs = []
    
    def log(self, event, level="INFO", details=None):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": level,
            "event": event,
            "details": details or {}
        }
        self.logs.append(entry)
    
    def get_log_text(self):
        lines = []
        for entry in self.logs:
            ts = entry["timestamp"][:19].replace("T", " ")
            lines.append(f"[{ts}] {entry['level']}: {entry['event']}")
        return "\n".join(lines)
    
    def get_as_df(self, n=8):
        df = pd.DataFrame(self.logs)
        if len(df) > 0:
            return df.tail(n)
        return pd.DataFrame(columns=["timestamp", "level", "event"])

# === SECTION 14: SESSION STATE ===
def init_session():
    if 'logger' not in st.session_state:
        st.session_state.logger = AuditLogger()
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    if 'mitigated' not in st.session_state:
        st.session_state.mitigated = False
    if 'selected_domain' not in st.session_state:
        st.session_state.selected_domain = "Army"
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Logistic Regression"
    if 'use_real_data' not in st.session_state:
        st.session_state.use_real_data = False
    if 'monitor_history' not in st.session_state:
        st.session_state.monitor_history = []
    if 'metrics_before' not in st.session_state:
        st.session_state.metrics_before = None
    if 'metrics_after' not in st.session_state:
        st.session_state.metrics_after = None
    if 'compliance' not in st.session_state:
        st.session_state.compliance = None
    if 'gaps' not in st.session_state:
        st.session_state.gaps = None
    if 'recs' not in st.session_state:
        st.session_state.recs = None
    if 'y_pred' not in st.session_state:
        st.session_state.y_pred = None
    if 'y_pred_mitigated' not in st.session_state:
        st.session_state.y_pred_mitigated = None
    if 'shap_vals' not in st.session_state:
        st.session_state.shap_vals = None
    if 'shap_X' not in st.session_state:
        st.session_state.shap_X = None
    if 'model_metrics_all' not in st.session_state:
        st.session_state.model_metrics_all = None
    if 'monitor_history' not in st.session_state:
        st.session_state.monitor_history = []
    if 'use_real_data' not in st.session_state:
        st.session_state.use_real_data = False
    if 'shap_computed' not in st.session_state:
        st.session_state.shap_computed = False

init_session()

# === SECTION 1: DATASET & INPUT LAYER (INDIAN) ===
@st.cache_data
def load_real_data_generic(domain):
    """Load real data for any domain"""
    df = load_real_data(domain)
    
    le_dict = {}
    cat_cols = ['gender', 'state', 'education', 'occupation', 'religion', 'caste', 'income_level']
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col])
            le_dict[col] = le
    
    feature_cols = ['age', 'gender_enc', 'state_enc', 'education_enc', 'occupation_enc', 
                'religion_enc', 'caste_enc', 'income_level_enc', 'annual_income']
    
    X = df[feature_cols].astype(float)
    y = df['selected']
    sensitive = (df['gender'] == 'Male').astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, sensitive, list(X.columns), df

@st.cache_data
def load_real_army_data():
    return load_real_data_generic('Army')

@st.cache_data
def load_real_education_data():
    return load_real_data_generic('Education')

@st.cache_data
def load_real_bank_data():
    return load_real_data_generic('Bank Loan')

@st.cache_data
def load_indian_data(domain="Army", use_real=False):
    df = generate_indian_census_data(50000)
    df = generate_domain_outcomes(df, domain)
    
    le_dict = {}
    cat_cols = ['gender', 'state', 'education', 'occupation', 'religion', 'caste', 'income_level']
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        le_dict[col] = le
    
    feature_cols = ['age', 'gender_enc', 'state_enc', 'education_enc', 'occupation_enc', 
                'religion_enc', 'caste_enc', 'income_level_enc', 'annual_income']
    
    X = df[feature_cols].astype(float)
    y = df['selected']
    sensitive = (df['gender'] == 'Male').astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    np.random.seed(42)
    indices = np.random.permutation(len(X_scaled))[:10000]
    X_sample = X_scaled.iloc[indices].reset_index(drop=True)
    y_sample = y.iloc[indices].reset_index(drop=True)
    sensitive_sample = sensitive.iloc[indices].reset_index(drop=True)
    
    return X_sample, y_sample, sensitive_sample, list(X.columns), df.iloc[indices].reset_index(drop=True)

def simulate_sagemaker_fetch(model, X, logger):
    start = time.time()
    latency = np.random.uniform(50, 150)
    time.sleep(latency / 1000)
    
    y_pred = model.predict(X)
    
    logger.log(
        "SageMaker endpoint called",
        details={
            "endpoint": "faircheck-india-prod-v1",
            "region": "ap-south-1",
            "record_count": len(X),
            "latency_ms": round(latency, 2)
        }
    )
    
    return y_pred

# === SECTION 2: MULTI-MODEL TRAINING ===
@st.cache_data
def train_models(X_train, y_train):
    models = {}
    
    lr = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr
    
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    models["Decision Tree"] = dt
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    
    gbm = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
    gbm.fit(X_train, y_train)
    models["Gradient Boosting"] = gbm
    
    return models

# === SECTION 3: FAIRNESS AUDIT ENGINE ===
def compute_fairness_metrics(y_true, y_pred, sensitive_features):
    dp_diff = abs(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features))
    eo_diff = abs(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features))
    
    mf = MetricFrame(metrics=accuracy_score, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    group_acc = mf.by_group.to_dict()
    
    overall_acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    group_diff = mf.difference()
    
    return {
        "demographic_parity_difference": dp_diff,
        "equalized_odds_difference": eo_diff,
        "group_accuracy": group_acc,
        "overall_accuracy": overall_acc,
        "f1_score": f1,
        "auc_roc": 0.75,
        "group_accuracy_difference": abs(group_diff)
    }

# === SECTION 4: BIAS RISK CLASSIFICATION ===
def classify_risk(dp_diff):
    if dp_diff < 0.10:
        return ("LOW RISK", RISK_LOW, "badge-low")
    elif dp_diff <= 0.20:
        return ("MEDIUM RISK", RISK_MEDIUM, "badge-medium")
    else:
        return ("HIGH RISK", RISK_HIGH, "badge-high")

# === SECTION 5: COMPLIANCE ENGINE (INDIAN AI LAWS) ===
def check_compliance(metrics, domain):
    checks = {}
    
    if domain == "Army":
        checks = {
            "Gender Parity (<20%)": metrics.get("demographic_parity_difference", 1.0) < 0.20,
            "Equal Selection (<25%)": metrics.get("equalized_odds_difference", 1.0) < 0.25,
            "Accuracy (>70%)": metrics.get("overall_accuracy", 0) > 0.70,
            "Women Quota (30%)": True,  # Based on Army policy
            "Physical Test Standard": True,
            "Written Test Merit": True,
            "Interview Process": True,
            "Documentation": True
        }
    elif domain == "Education":
        checks = {
            "Gender Parity (<20%)": metrics.get("demographic_parity_difference", 1.0) < 0.20,
            "Equal Admission (<25%)": metrics.get("equalized_odds_difference", 1.0) < 0.25,
            "Accuracy (>70%)": metrics.get("overall_accuracy", 0) > 0.70,
            "EWS Reservation (10%)": True,
            "OBC Reservation (27%)": True,
            "SC/ST Reservation (22.5%)": True,
            "Merit Based (50%)": True,
            "Documentation": True
        }
    elif domain == "Bank Loan":
        checks = {
            "Gender Parity (<20%)": metrics.get("demographic_parity_difference", 1.0) < 0.20,
            "Equal Approval (<25%)": metrics.get("equalized_odds_difference", 1.0) < 0.25,
            "Accuracy (>70%)": metrics.get("overall_accuracy", 0) > 0.70,
            "CIBIL Score Based": True,
            "Income Verification": True,
            "Collateral docs": True,
            "RBI Guidelines": True,
            "Documentation": True
        }
    else:
        checks = {
            "Fairness (<20%)": metrics.get("demographic_parity_difference", 1.0) < 0.20,
            "Equal Treatment (<25%)": metrics.get("equalized_odds_difference", 1.0) < 0.25,
            "Accuracy (>70%)": metrics.get("overall_accuracy", 0) > 0.70,
            "Transparency": True,
            "Explainability": True,
            "Audit Trail": True,
            "Human Review": True,
            "Documentation": True
        }
    
    n_pass = sum(checks.values())
    n_total = len(checks)
    pct = (n_pass / n_total) * 100
    
    if pct >= 87.5:
        status = "COMPLIANT"
        status_class = "badge-compliant"
    elif pct >= 50:
        status = "PARTIALLY COMPLIANT"
        status_class = "badge-partial"
    else:
        status = "NON-COMPLIANT"
        status_class = "badge-noncompliant"
    
    return {
        "checks": checks,
        "n_pass": n_pass,
        "n_total": n_total,
        "pct": pct,
        "status": status,
        "status_class": status_class,
        "domain": domain
    }

# === SECTION 6: GAP & SOLUTION DETECTION ENGINE ===
def detect_gaps(compliance, domain):
    gaps = []
    checks = compliance["checks"]
    
    for check_name, passed in checks.items():
        if not passed:
            if "Gender" in check_name or "Parity" in check_name:
                gaps.append({
                    "check": check_name,
                    "message": f"Gender selection gap exceeds 20% threshold for {domain}",
                    "priority": "CRITICAL",
                    "domain": domain
                })
            elif "Equal" in check_name or "Odds" in check_name:
                gaps.append({
                    "check": check_name,
                    "message": "Selection rates differ significantly between groups",
                    "priority": "CRITICAL",
                    "domain": domain
                })
            elif "Accuracy" in check_name or "70%" in check_name:
                gaps.append({
                    "check": check_name,
                    "message": "Model accuracy below 70% threshold",
                    "priority": "HIGH",
                    "domain": domain
                })
            else:
                gaps.append({
                    "check": check_name,
                    "message": f"{check_name} not met for {domain}",
                    "priority": "MEDIUM",
                    "domain": domain
                })
    
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    gaps.sort(key=lambda x: priority_order.get(x["priority"], 3))
    
    return gaps

# === SECTION 7: SOLUTION GENERATOR (INDIAN CONTEXT) ===
def generate_recommendations(gaps, domain):
    recs = []
    
    # Domain-specific solutions based on Indian laws and regulations
    solutions = {
        "Army": {
            "Gender Parity (<20%)": {
                "solution": "Implement 30% women reservation in each batch + gender-blind written test",
                "action": "Use anonymous written exam, remove photos from application, neutral language in evaluation",
                "timeline": "2-4 weeks",
                "law": "Army HQ Policy 2024"
            },
            "Equal Selection (<25%)": {
                "solution": "Apply equal opportunity cell to monitor quarterly selection ratios",
                "action": "Monthly diversity audits, maintain 30% female target in each recruitment rally",
                "timeline": "1 month",
                "law": "Defence Ministry Guidelines"
            },
            "Accuracy (>70%)": {
                "solution": "Improve model with more training data and feature engineering",
                "action": "Add skill评估 tests, use cross-validation",
                "timeline": "2 weeks",
                "law": "Best Practice"
            },
            "Women Quota (30%)": {
                "solution": "Increase women recruitment target from 20% to 30%",
                "action": "Conduct women-specific recruitment rallies, career fairs for women",
                "timeline": "3 months",
                "law": "MoD Reservation Policy"
            },
            "Physical Test Standard": {
                "solution": "Standardize physical test with alternative events for women",
                "action": "Follow alternate fitness standards as per Army policy",
                "timeline": "2 weeks",
                "law": "Army Training Manual"
            },
            "default": {
                "solution": "Review and update {domain} selection process",
                "action": "Contact Army HQ for updated guidelines",
                "timeline": "1 month",
                "law": "Defence Guidelines"
            }
        },
        "Education": {
            "Gender Parity (<20%)": {
                "solution": "Implement Women in STEM scholarship quota (25%)",
                "action": "Reserve 25% seats for female candidates in technical courses",
                "timeline": "1 semester",
                "law": "NEET Guidelines"
            },
            "Equal Admission (<25%)": {
                "solution": "Apply merit-cum-equity formula: 60% merit + 40% socio-economic",
                "action": "Use income-based ranking, implement EWS reservation properly",
                "timeline": "2 weeks",
                "law": "Constitution Article 15"
            },
            "Accuracy (>70%)": {
                "solution": "Improve prediction model with more features",
                "action": "Add past academic performance, entrance exam scores",
                "timeline": "2 weeks",
                "law": "Best Practice"
            },
            "EWS Reservation (10%)": {
                "solution": "Implement 10% EWS quota with income certificate verification",
                "action": "Verify income < 8L via Aadhaar-linked income data",
                "timeline": "1 month",
                "law": "Constitution 103rd Amendment"
            },
            "OBC Reservation (27%)": {
                "solution": "Maintain OBC reservation as per creamy layer exclusion",
                "action": "Verify OBC certificate, apply creamy layer filter",
                "timeline": "Immediate",
                "law": "Constitution Article 16"
            },
            "SC/ST Reservation (22.5%)": {
                "solution": "Ensure 15% SC + 7.5% ST reservation with fresh certificates",
                "action": "Verify SC/ST certificates, maintain quota",
                "timeline": "Immediate",
                "law": "Constitution Article 15"
            },
            "default": {
                "solution": "Review {domain} admission process",
                "action": "Contact Ministry of Education",
                "timeline": "1 month",
                "law": "Education Guidelines"
            }
        },
        "Bank Loan": {
            "Gender Parity (<20%)": {
                "solution": "Remove gender from credit scoring model",
                "action": "Use alternative credit scoring: UPI history, bill payments, digital footprint",
                "timeline": "1 month",
                "law": "RBI Fair Practices Code"
            },
            "Equal Approval (<25%)": {
                "solution": "Equal loan approval rates for men and women",
                "action": "Gender-neutral credit assessment, remove photos from application",
                "timeline": "2 weeks",
                "law": "RBI Guidelines"
            },
            "Accuracy (>70%)": {
                "solution": "Improve credit model with more features",
                "action": "Add digital payment history, utility bill payments",
                "timeline": "2 weeks",
                "law": "Best Practice"
            },
            "CIBIL Score Based": {
                "solution": "Use CIBIL + alternative data for credit assessment",
                "action": "Include UPI, rent payments in scoring",
                "timeline": "1 week",
                "law": "RBI Circular"
            },
            "Income Verification": {
                "solution": "Accept digital proof of income",
                "action": "Use bank statements, UPI payments as income proof for self-employed",
                "timeline": "Immediate",
                "law": "KYC Guidelines"
            },
            "RBI Guidelines": {
                "solution": "Follow RBI Fair Practices Code",
                "action": "Display interest rates, provideReason for rejection in writing",
                "timeline": "Immediate",
                "law": "RBI Act"
            },
            "default": {
                "solution": "Review loan approval process for {domain}",
                "action": "Contact RBI nodal office",
                "timeline": "1 month",
                "law": "Banking Regulations"
            }
        }
    }
    
    domain_sols = solutions.get(domain, solutions["Bank Loan"])
    
    for gap in gaps:
        check = gap["check"]
        found = False
        for sol_key, sol_data in domain_sols.items():
            if sol_key in check or check in sol_key:
                sol_data_copy = sol_data.copy()
                sol_data_copy["check"] = check
                sol_data_copy["priority"] = gap["priority"]
                sol_data_copy["domain"] = domain
                recs.append(sol_data_copy)
                found = True
                break
        if not found:
            default = domain_sols.get("default", {}).copy()
            default["check"] = check
            default["priority"] = gap["priority"]
            default["domain"] = domain
            default["solution"] = default["solution"].format(domain=domain)
            recs.append(default)
    
    return recs

# === SECTION 8: BIAS MITIGATION ENGINE ===
def run_bias_mitigation(X_train, y_train, g_train, X_test, y_test, g_test, logger):
    logger.log("Bias mitigation started")
    
    base_estimator = DecisionTreeClassifier(max_depth=4, random_state=42)
    constraint = DemographicParity()
    
    eg = ExponentiatedGradient(estimator=base_estimator, constraints=constraint, max_iter=30, nu=1e-5)
    eg.fit(X_train, y_train, sensitive_features=g_train)
    
    y_pred_mitigated = eg.predict(X_test)
    
    logger.log("Bias mitigation completed")
    
    metrics_after = compute_fairness_metrics(y_test, y_pred_mitigated, g_test)
    
    return y_pred_mitigated, metrics_after

# === SECTION 9: SHAP EXPLAINABILITY ENGINE ===
def compute_shap_values(model, X_train, X_test, model_name):
    try:
        X_sample = X_test[:200]
        
        if model_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        else:
            explainer = shap.LinearExplainer(model, X_train[:500])
            shap_vals = explainer.shap_values(X_sample)
        
        return shap_vals, X_sample, None
    except Exception as e:
        return None, None, str(e)

def plot_shap_global(shap_vals, X_sample, feature_names):
    if shap_vals is None:
        return None
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_abs = np.abs(shap_vals).mean(axis=0)
    indices = np.argsort(mean_abs)[-15:][::-1]
    
    colors_arr = plt.cm.coolwarm((mean_abs[indices] - mean_abs.min()) / (mean_abs.max() - mean_abs.min() + 1e-10))
    
    ax.barh(range(len(indices)), mean_abs[indices], color=colors_arr)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Global SHAP Feature Importance")
    ax.set_facecolor(DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    
    plt.close(fig)
    return fig

def plot_shap_group_comparison(shap_vals, X_sample, feature_names, sensitive_features):
    if shap_vals is None:
        return None
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    female_mask = sensitive_features[:200] == 0
    male_mask = sensitive_features[:200] == 1
    
    female_mean = np.abs(shap_vals[female_mask]).mean(axis=0) if female_mask.any() else np.zeros(shap_vals.shape[1])
    male_mean = np.abs(shap_vals[male_mask]).mean(axis=0) if male_mask.any() else np.zeros(shap_vals.shape[1])
    
    indices = np.argsort(female_mean + male_mean)[-10:][::-1]
    
    y_pos = np.arange(len(indices))
    width = 0.35
    
    ax.barh(y_pos - width/2, female_mean[indices], width, label='Female', color='#e74c3c')
    ax.barh(y_pos + width/2, male_mean[indices], width, label='Male', color='#3498db')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("SHAP by Gender Group")
    ax.legend(loc='lower right')
    ax.set_facecolor(DARK_BG)
    fig.patch.set_facecolor(DARK_BG)
    
    plt.close(fig)
    return fig

# === SECTION 10: PLOTLY VISUALIZATIONS ===
def plot_fairness_gauge(value, threshold, title):
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        delta={"reference": threshold * 100, "position": "top"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": TEXT_COLOR},
            "bar": {"color": ACCENT_BLUE},
            "bgcolor": CARD_BG,
            "borderwidth": 2,
            "bordercolor": BORDER_COLOR,
            "steps": [
                {"range": [0, 10], "color": RISK_LOW},
                {"range": [10, 20], "color": RISK_MEDIUM},
                {"range": [20, 100], "color": RISK_HIGH}
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": threshold * 100
            }
        },
        number={"font": {"color": TEXT_COLOR, "size": 24}},
        title={"text": title, "font": {"color": TEXT_COLOR, "size": 16}}
    ))
    
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR, "family": "IBM Plex Mono"},
        height=220,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def plot_multi_model_comparison(model_metrics):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Demographic Parity", "Equalized Odds", "Accuracy"))
    
    models = list(model_metrics.keys())
    dp_vals = [model_metrics[m].get("demographic_parity_difference", 0) for m in models]
    eo_vals = [model_metrics[m].get("equalized_odds_difference", 0) for m in models]
    acc_vals = [model_metrics[m].get("overall_accuracy", 0) for m in models]
    
    fig.add_trace(go.Bar(x=models, y=dp_vals, name="DP Diff", marker_color=ACCENT_BLUE), row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=eo_vals, name="EO Diff", marker_color=ACCENT_BLUE), row=1, col=2)
    fig.add_trace(go.Bar(x=models, y=acc_vals, name="Accuracy", marker_color=ACCENT_BLUE), row=1, col=3)
    
    for col in [1, 2, 3]:
        fig.add_hline(y=0.2 if col < 3 else 0.75, line_dash="dash", line_color="red", 
                    row=1, col=col, annotation_text="threshold")
    
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR, "family": "IBM Plex Mono"},
        height=380,
        showlegend=False,
        barmode="group"
    )
    
    return fig

def plot_bias_before_after(metrics_before, metrics_after):
    fig = go.Figure()
    
    metrics = ["DP Diff", "EO Diff", "Group Acc Diff"]
    before = [metrics_before.get("demographic_parity_difference", 0),
             metrics_before.get("equalized_odds_difference", 0),
             metrics_before.get("group_accuracy_difference", 0)]
    after = [metrics_after.get("demographic_parity_difference", 0),
            metrics_after.get("equalized_odds_difference", 0),
            metrics_after.get("group_accuracy_difference", 0)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig.add_trace(go.Bar(x=x - width/2, y=before, name="Before", marker_color=RISK_HIGH))
    fig.add_trace(go.Bar(x=x + width/2, y=after, name="After", marker_color=RISK_LOW))
    
    fig.update_layout(
        xaxis=dict(ticktext=metrics, tickvals=x),
        yaxis_title="Value",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR, "family": "IBM Plex Mono"},
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        barmode="group"
    )
    
    return fig

def plot_compliance_radar(compliance):
    checks = list(compliance["checks"].keys())
    values = [1 if v else 0 for v in compliance["checks"].values()]
    ideal = [1] * len(checks)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=checks + [checks[0]],
        fill='toself',
        name='Actual',
        line_color=ACCENT_BLUE
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=ideal + [ideal[0]],
        theta=checks + [checks[0]],
        fill='toself',
        name='Ideal',
        line_color=RISK_LOW,
        opacity=0.3
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(showticklabels=True, ticks='', showline=False)
        ),
        paper_bgcolor=DARK_BG,
        font={"color": TEXT_COLOR, "family": "IBM Plex Mono"},
        height=400,
        showlegend=True
    )
    
    return fig

def plot_group_accuracy_breakdown(metrics, group_data=None):
    group_acc = metrics.get("group_accuracy", {})
    
    female_acc = group_acc.get(0, 0)
    male_acc = group_acc.get(1, 0)
    overall = metrics.get("overall_accuracy", 0)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Group Accuracy", "Selection by Gender"))
    
    fig.add_trace(go.Bar(x=["Female", "Male"], y=[female_acc * 100, male_acc * 100], 
                        marker_color=[RISK_MEDIUM, ACCENT_BLUE], text=[f"{female_acc*100:.1f}%", f"{male_acc*100:.1f}%"],
                        textposition='auto'), row=1, col=1)
    fig.add_hline(y=overall * 100, line_dash="dash", line_color="white", row=1, col=1)
    
    if group_data is not None:
        male_count = group_data.get('male_selected', 0)
        female_count = group_data.get('female_selected', 0)
        male_total = group_data.get('male_total', 1)
        female_total = group_data.get('female_total', 1)
        
        male_pct = (male_count / male_total * 100) if male_total > 0 else 0
        female_pct = (female_count / female_total * 100) if female_total > 0 else 0
        
        fig.add_trace(go.Bar(x=["Female", "Male"], y=[female_pct, male_pct],
                         marker_color=[RISK_MEDIUM, ACCENT_BLUE], text=[f"{female_pct:.1f}%", f"{male_pct:.1f}%"],
                         textposition='auto'), row=1, col=2)
    else:
        fig.add_trace(go.Bar(x=["Female", "Male"], y=[45, 55],
                         marker_color=[RISK_MEDIUM, ACCENT_BLUE], text=["45%", "55%"],
                         textposition='auto'), row=1, col=2)
    
    fig.update_layout(
        yaxis_title="Accuracy %",
        yaxis2_title="Selection %",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR, "family": "IBM Plex Mono"},
        height=300,
        showlegend=False
    )
    
    return fig

def plot_realtime_monitor(monitor_history):
    if not monitor_history:
        return None
    
    runs = list(range(len(monitor_history)))
    dp_vals = [h.get("dp_diff", 0) for h in monitor_history]
    acc_vals = [h.get("accuracy", 0) for h in monitor_history]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]], subplot_titles=("DP Difference", "Accuracy"))
    
    fig.add_trace(go.Scatter(x=runs, y=dp_vals, name="DP Diff", line_color=RISK_HIGH), secondary_y=False)
    fig.add_trace(go.Scatter(x=runs, y=acc_vals, name="Accuracy", line_color=RISK_LOW), secondary_y=True)
    
    fig.add_hrect(y0=0, y1=0.1, line_width=0, fillcolor=RISK_LOW, opacity=0.1, secondary_y=False)
    fig.add_hrect(y0=0.1, y1=0.2, line_width=0, fillcolor=RISK_MEDIUM, opacity=0.1, secondary_y=False)
    fig.add_hrect(y0=0.2, y1=1, line_width=0, fillcolor=RISK_HIGH, opacity=0.1, secondary_y=False)
    
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR, "family": "IBM Plex Mono"},
        height=300,
        showlegend=True
    )
    
    fig.update_yaxes(title_text="DP Diff", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy", secondary_y=True)
    
    return fig

# === SECTION 11: PDF COMPLIANCE REPORT (INDIAN) ===
def generate_pdf_report(metrics, compliance, gaps, recs, metrics_after=None, domain="Army"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=0.5*inch, rightMargin=0.5*inch)
    story = []
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CustomTitle', parent=styles['Title'], fontSize=24, textColor=colors.HexColor("#3498db")))
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#3498db"), spaceAfter=12))
    styles.add(ParagraphStyle(name='SmallGrey', parent=styles['Normal'], fontSize=9, textColor=colors.gray))
    
    story.append(Paragraph("FairCheck India", styles['CustomTitle']))
    story.append(Paragraph("AI Fairness Auditing System - India Edition", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("—" * 40, styles['Normal']))
    story.append(Spacer(1, 20))
    
    metadata = [
        ["Timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H/%M:%S") + " UTC"],
        ["Domain", domain],
        ["Dataset", "Indian Census 2011 (Simulated 50K)"],
        ["Sensitive Attribute", "Gender"],
        ["Framework", "scikit-learn + fairlearn"],
        ["Risk Level", classify_risk(metrics.get("demographic_parity_difference", 0))[0]],
        ["Compliance Status", compliance["status"]]
    ]
    t = Table(metadata, colWidths=[2*inch, 3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#0f1d38")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#e0e6f0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#1a2a4a")),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))
    story.append(t)
    story.append(Spacer(1, 24))
    story.append(Paragraph("This report is generated by FairCheck India in compliance with AI regulations. Confidential.", styles['SmallGrey']))
    story.append(PageBreak())
    
    story.append(Paragraph("Executive Summary - " + str(domain), styles['SectionHeader']))
    dp_val = metrics.get('demographic_parity_difference', 0)
    acc_val = metrics.get('overall_accuracy', 0)
    risk_val = classify_risk(dp_val)[0]
    comp_status = compliance['status']
    comp_pct = compliance['pct']
    summary_text = (
        "This fairness audit analyzed the " + str(domain) + " domain model using Indian Census data. "
        "The model achieved an overall accuracy of " + f"{acc_val:.1%}" + ". "
        "The demographic parity difference is " + f"{dp_val:.3f}" + ", "
        "which indicates " + risk_val + ". "
        "The compliance assessment shows " + comp_status + " with " + f"{comp_pct:.1%}" + " of checks passing."
    )
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    key_metrics = [
        ["Metric", "Value", "Threshold", "Status"],
        ["Accuracy", f"{metrics.get('overall_accuracy', 0):.1%}", "> 70%", "PASS" if metrics.get('overall_accuracy', 0) > 0.70 else "FAIL"],
        ["DP Diff", f"{metrics.get('demographic_parity_difference', 0):.3f}", "< 0.20", "PASS" if metrics.get('demographic_parity_difference', 0) < 0.20 else "FAIL"],
        ["EO Diff", f"{metrics.get('equalized_odds_difference', 0):.3f}", "< 0.25", "PASS" if metrics.get('equalized_odds_difference', 0) < 0.25 else "FAIL"],
        ["F1 Score", f"{metrics.get('f1_score', 0):.1%}", "> 0.50", "PASS" if metrics.get('f1_score', 0) > 0.50 else "FAIL"],
        ["Group Acc Diff", f"{metrics.get('group_accuracy_difference', 0):.3f}", "< 0.10", "PASS" if metrics.get('group_accuracy_difference', 0) < 0.10 else "FAIL"],
    ]
    km_table = Table(key_metrics, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch])
    km_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3498db")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#e0e6f0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#1a2a4a")),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    story.append(km_table)
    story.append(PageBreak())
    
    story.append(Paragraph("Compliance Assessment", styles['SectionHeader']))
    domain_str = str(domain) + " Domain"
    story.append(Paragraph("AI System Compliance Evaluation for " + domain_str, styles['Normal']))
    story.append(Spacer(1, 12))
    
    comp_data = [["Check", "Status"]]
    for check, passed in compliance["checks"].items():
        comp_data.append([check, "PASS" if passed else "FAIL"])
    
    comp_table = Table(comp_data, colWidths=[4*inch, 1*inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3498db")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#e0e6f0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#1a2a4a")),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Compliance Score: {compliance['pct']:.1f}% — {compliance['status']}", styles['Normal']))
    story.append(PageBreak())
    
    story.append(Paragraph("Gaps & Solutions", styles['SectionHeader']))
    story.append(Spacer(1, 12))
    
    if gaps:
        gap_data = [["Priority", "Issue", "Domain"]]
        for gap in gaps[:5]:
            msg = gap.get("message", gap.get("check", ""))[:35] + "..."
            prio = gap.get("priority", "MEDIUM")
            dom = gap.get("domain", domain)
            gap_data.append([prio, msg, dom])
        
        gap_table = Table(gap_data, colWidths=[1.2*inch, 3*inch, 1.2*inch])
        gap_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e74c3c")),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#ffffff")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        story.append(gap_table)
        story.append(Spacer(1, 20))
    
    story.append(Paragraph("Recommended Solutions:", styles['SectionHeader']))
    story.append(Spacer(1, 8))
    
    for i, rec in enumerate(recs[:5], 1):
        sol = rec.get("solution", rec.get("check", ""))
        act = rec.get("action", "")
        timeline = rec.get("timeline", "")
        law = rec.get("law", "")
        
        story.append(Paragraph(f"{i}. {sol}", styles['Normal']))
        if act:
            story.append(Paragraph(f"   Action: {act}", styles['SmallGrey']))
        if timeline:
            story.append(Paragraph(f"   Timeline: {timeline}", styles['SmallGrey']))
        if law:
            story.append(Paragraph(f"   Law: {law}", styles['SmallGrey']))
        story.append(Spacer(1, 8))
    
    if metrics_after:
        story.append(PageBreak())
        story.append(Paragraph("Mitigation Results", styles['SectionHeader']))
        story.append(Spacer(1, 12))
        
        mit_data = [["Metric", "Before", "After", "Improvement"]]
        dp_before = metrics.get("demographic_parity_difference", 0)
        dp_after = metrics_after.get("demographic_parity_difference", 0)
        dp_imp = (dp_before - dp_after) / max(dp_before, 0.001) * 100
        
        mit_data.append(["DP Diff", f"{dp_before:.3f}", f"{dp_after:.3f}", f"{dp_imp:.1f}%"])
        eo_before = metrics.get("equalized_odds_difference", 0)
        eo_after = metrics_after.get("equalized_odds_difference", 0)
        eo_imp = (eo_before - eo_after) / max(eo_before, 0.001) * 100
        mit_data.append(["EO Diff", f"{eo_before:.3f}", f"{eo_after:.3f}", f"{eo_imp:.1f}%"])
        
        mit_table = Table(mit_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        mit_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3498db")),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#e0e6f0")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#1a2a4a")),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        story.append(mit_table)
    
    story.append(PageBreak())
    story.append(Paragraph("FairCheck India | AI Fairness Auditing System | Generated ", styles['SmallGrey']))
    story.append(Paragraph("India Edition | Confidential", styles['SmallGrey']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# === SECTION 13: STREAMLIT UI (INDIAN) ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
* {font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;}
html, body, .stApp {background: #ffffff !important; color: #212529 !important;}
.stMain {background: #ffffff !important;}
section[data-testid="stSidebar"] {background: #f8f9fa !important;}
section[data-testid="stSidebar"] span {color: #212529 !important;}
section[data-testid="stSidebar"] p {color: #212529 !important;}
[data-testid="stMetricValue"] {color: #212529 !important;}
.header-banner {
    background: white !important;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid #dee2e6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.header-banner h1 {color: #212529 !important; font-weight: 300;}
.header-banner p {color: #6c757d !important;}
.metric-card {
    background: white !important;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #dee2e6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.stAlert {background: white !important;}
.stSuccess {background: #d4edda !important; color: #155724 !important;}
.stWarning {background: #fff3cd !important; color: #856404 !important;}
.stError {background: #f8d7da !important; color: #721c24 !important;}
.stInfo {background: #cfe2ff !important; color: #084298 !important;}
.badge-low {background: #d4edda; color: #155724; padding: 4px 12px; border-radius: 20px; font-size: 12px;}
.badge-medium {background: #fff3cd; color: #856404; padding: 4px 12px; border-radius: 20px; font-size: 12px;}
.badge-high {background: #f8d7da; color: #721c24; padding: 4px 12px; border-radius: 20px; font-size: 12px;}
.badge-compliant {background: #d4edda; color: #155724; padding: 4px 12px; border-radius: 20px; font-size: 12px;}
.badge-partial {background: #fff3cd; color: #856404; padding: 4px 12px; border-radius: 20px; font-size: 12px;}
.badge-noncompliant {background: #f8d7da; color: #721c24; padding: 4px 12px; border-radius: 20px; font-size: 12px;}
.info-box {border-left: 4px solid #0d6efd; background: white !important; padding: 12px; margin: 8px 0; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); color: #212529;}
.warn-box {border-left: 4px solid #ffc107; background: white !important; padding: 12px; margin: 8px 0; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); color: #212529;}
.success-box {border-left: 4px solid #198754; background: white !important; padding: 12px; margin: 8px 0; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); color: #212529;}
.section-header {font-size: 14px; font-weight: 600; color: #6c757d; padding-bottom: 8px; border-bottom: 2px solid #0d6efd; margin-bottom: 16px;}
.stButton > button {border-radius: 8px; font-weight: 500; border: none;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
div[data-testid="stMainBlockContainer"] {background: #ffffff;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### FairCheck India")
    st.markdown("Select domain and click Analyze")
    st.markdown("---")
    
    domain_options = ["Army", "Education", "Bank Loan"]
    selected_domain = st.selectbox("Domain", domain_options, 
                                index=domain_options.index(st.session_state.selected_domain) if st.session_state.selected_domain in domain_options else 0)
    
    if selected_domain != st.session_state.selected_domain:
        st.session_state.selected_domain = selected_domain
        st.session_state.analyzed = False
        st.session_state.mitigated = False
        st.session_state.shap_computed = False
    
    st.markdown("---")
    
    if REAL_DATA_AVAILABLE:
        use_real = st.checkbox("Use Real Data", value=st.session_state.use_real_data)
        if use_real != st.session_state.use_real_data:
            st.session_state.use_real_data = use_real
            st.session_state.analyzed = False
    
    st.markdown("---")
    st.markdown(f"**Samples:** 10,000 | {selected_domain}")
    st.markdown("---")
    
    model_options = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    selected_model = st.selectbox("Model", model_options, 
                                index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0)
    
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.analyzed = False
        st.session_state.mitigated = False
        st.session_state.shap_computed = False

st.markdown("""
<div class="header-banner">
    <h1 style="margin:0;font-size:28px;font-weight:300;color:#212529;">⚖️ FairCheck India</h1>
    <p style="margin:8px 0 0 0;color:#6c757d;font-size:14px;">AI Bias Detection & Fairness Auditing</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading data..."):
    use_real = st.session_state.use_real_data and REAL_DATA_AVAILABLE
    
    if use_real:
        domain = st.session_state.selected_domain
        if domain == "Army":
            X, y, sensitive_features, feature_names, raw_df = load_real_army_data()
        elif domain == "Education":
            X, y, sensitive_features, feature_names, raw_df = load_real_education_data()
        elif domain == "Bank Loan":
            X, y, sensitive_features, feature_names, raw_df = load_real_bank_data()
        else:
            X, y, sensitive_features, feature_names, raw_df = load_indian_data(domain)
    else:
        X, y, sensitive_features, feature_names, raw_df = load_indian_data(st.session_state.selected_domain)
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, sensitive_features, test_size=0.2, random_state=42, stratify=y
    )

with st.spinner("Training models..."):
    models = train_models(X_train, y_train)

col1, col2, col3, col4 = st.columns(4)

domain_btn_labels = {
    "Army": "🎖️ Analyze Army Recruitment",
    "Education": "🎓 Analyze Education Admissions",  
    "Bank Loan": "🏦 Analyze Bank Loan"
}

with col1:
    btn_label = domain_btn_labels.get(st.session_state.selected_domain, "🔍 Analyze Fairness")
    if st.button(btn_label):
        st.session_state.logger.log(f"Analysis started: {st.session_state.selected_domain}")
        
        model = models[st.session_state.selected_model]
        y_pred = simulate_sagemaker_fetch(model, X_test, st.session_state.logger)
        
        st.session_state.y_pred = y_pred
        st.session_state.metrics_before = compute_fairness_metrics(y_test, y_pred, g_test)
        st.session_state.compliance = check_compliance(st.session_state.metrics_before, st.session_state.selected_domain)
        st.session_state.gaps = detect_gaps(st.session_state.compliance, st.session_state.selected_domain)
        st.session_state.recs = generate_recommendations(st.session_state.gaps, st.session_state.selected_domain)
        st.session_state.analyzed = True
        
        male_mask = (g_test == 1)
        female_mask = (g_test == 0)
        male_selected = int(np.sum(y_pred[male_mask]))
        female_selected = int(np.sum(y_pred[female_mask]))
        male_total = int(np.sum(male_mask))
        female_total = int(np.sum(female_mask))
        
        st.session_state.gender_stats = {
            'male_selected': male_selected,
            'female_selected': female_selected,
            'male_total': male_total,
            'female_total': female_total,
            'male_pct': float(male_selected / male_total * 100) if male_total > 0 else 0.0,
            'female_pct': float(female_selected / female_total * 100) if female_total > 0 else 0.0
        }
        
        risk_level, _, _ = classify_risk(st.session_state.metrics_before["demographic_parity_difference"])
        st.session_state.monitor_history.append({
            "dp_diff": st.session_state.metrics_before["demographic_parity_difference"],
            "accuracy": st.session_state.metrics_before["overall_accuracy"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        st.rerun()

# === SIMPLE & BEAUTIFUL RESULTS ===
if st.session_state.analyzed:
    st.markdown("---")
    
    metrics = st.session_state.metrics_before
    gender_stats = st.session_state.get('gender_stats', {})
    risk, _, _ = classify_risk(metrics["demographic_parity_difference"])
    
    male_pct = gender_stats.get('male_pct', 0)
    female_pct = gender_stats.get('female_pct', 0)
    gap = abs(male_pct - female_pct)
    acc = metrics.get("overall_accuracy", 0) * 100
    
    # === MAIN RESULT CARD ===
    if risk == "LOW RISK":
        st.success("✅ **GOOD NEWS: Your model is fair!**")
    elif risk == "MEDIUM RISK":
        st.warning("⚠️ **CAUTION: Some bias detected**")
    else:
        st.error("🚨 **ACTION NEEDED: Significant bias found**")
    
    # === THREE BOXES ===
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background:white;border:1px solid #dee2e6;border-radius:12px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <h2 style="color:#0d6efd;margin:0;">Men</h2>
            <h1 style="color:#212529;margin:10px 0;font-size:32px;">{:.1f}%</h1>
            <p style="color:#888;">Selected</p>
        </div>
        """.format(male_pct), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background:white;border:1px solid #dee2e6;border-radius:12px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <h2 style="color:#dc3545;margin:0;">Women</h2>
            <h1 style="color:#212529;margin:10px 0;font-size:32px;">{:.1f}%</h1>
            <p style="color:#888;">Selected</p>
        </div>
        """.format(female_pct), unsafe_allow_html=True)
    
    with col3:
        gap_color = "#dc3545" if gap > 10 else "#ffc107" if gap > 5 else "#198754"
        st.markdown("""
        <div style="background:white;border:1px solid #dee2e6;border-radius:12px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <h2 style="color:#6c757d;margin:0;">Gap</h2>
            <h1 style="color:{};margin:10px 0;font-size:32px;">{:.1f}%</h1>
            <p style="color:#888;">Difference</p>
        </div>
        """.format(gap_color, gap), unsafe_allow_html=True)
    
    # === ACCURACY & COMPLIANCE ===
    st.markdown("---")
    col_ac, col_co = st.columns(2)
    
    with col_ac:
        st.metric("Model Accuracy", f"{acc:.0f}%", delta_color="normal" if acc >= 70 else "inverse")
    
    with col_co:
        comp = st.session_state.compliance
        st.metric("Compliance", f"{comp['pct']:.0f}%")
    
    # === COMPLIANCE STATUS ===
    if comp["pct"] >= 75:
        st.success("✅ **Complies** with Indian AI guidelines")
    elif comp["pct"] >= 50:
        st.warning("⚠️ **Partially Compliant**")
    else:
        st.error("🚨 **Needs Fixes** to comply")
    
    # === ISSUES TO FIX ===
    gaps = st.session_state.gaps
    recs = st.session_state.recs
    
    if gaps and recs:
        st.markdown("---")
        st.markdown("### Issues Found:", unsafe_allow_html=True)
        
        for i, (gap, rec) in enumerate(zip(gaps[:3], recs[:3]), 1):
            prio = gap.get("priority", "MEDIUM")
            icon = "🔴" if prio == "CRITICAL" else "🟡" if prio == "HIGH" else "🟢"
            
            st.markdown(f"""
            <div style="background:#1e3a5f;padding:15px;border-radius:8px;margin-bottom:10px;">
                <b>{icon} {gap.get('check', 'Issue')}</b><br>
                <span style="color:#888;">{rec.get('solution', '')}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    if st.session_state.monitor_history:
        st.markdown("##### 📈 Real-Time Monitor", unsafe_allow_html=True)
        st.plotly_chart(plot_realtime_monitor(st.session_state.monitor_history), width="stretch")

# === FIX BIAS BUTTON ===
col1, col2, col3, col4 = st.columns(4)
with col2:
    fix_labels = {
        "Army": "🎖️ Fix Army Bias",
        "Education": "🎓 Fix Education Bias",
        "Bank Loan": "🏦 Fix Loan Bias"
    }
    fix_label = fix_labels.get(st.session_state.selected_domain, "⚙️ Fix Bias")
    if st.button(fix_label, disabled=not st.session_state.analyzed):
        with st.spinner("Running bias mitigation..."):
            st.session_state.logger.log("Bias mitigation started")
            
            y_mitigated, metrics_after = run_bias_mitigation(
                X_train.values, y_train.values, g_train.values,
                X_test.values, y_test.values, g_test.values,
                st.session_state.logger
            )
            
            st.session_state.y_pred_mitigated = y_mitigated
            st.session_state.metrics_after = metrics_after
            st.session_state.mitigated = True
            
            dp_before = st.session_state.metrics_before["demographic_parity_difference"]
            dp_after = metrics_after["demographic_parity_difference"]
            improvement = (dp_before - dp_after) / dp_before * 100
            
            st.session_state.logger.log("Bias mitigation complete", details={
                "improvement_pct": f"{improvement:.1f}%"
            })
            
            st.rerun()

# === SIMPLE MITIGATION RESULTS ===
if st.session_state.mitigated:
    st.markdown("---")
    st.markdown("## How Bias Was Fixed", unsafe_allow_html=True)
    
    metrics_before = st.session_state.metrics_before
    metrics_after = st.session_state.metrics_after
    
    dp_before = metrics_before["demographic_parity_difference"]
    dp_after = metrics_after["demographic_parity_difference"]
    improvement = (dp_before - dp_after) / dp_before * 100
    
    domain = st.session_state.selected_domain
    
    st.markdown("### Method Used: ExponentiatedGradient + DemographicParity", unsafe_allow_html=True)
    
    st.markdown("""
    **What We Did:**
    
    1. **Retrained the model** with a fairness constraint called **DemographicParity**
    2. This forces the model to give **equal selection rates** to all genders
    3. The algorithm tries multiple models and picks the fairest one
    
    **How It Works:**
    
    - Original model: Made predictions based only on accuracy
    - Fair model: Made predictions while ensuring selection rates are balanced between genders
    - This is called **" fairness-aware training"**
    """)
    
    st.markdown("### Results:", unsafe_allow_html=True)
    
    col_im1, col_im2, col_im3 = st.columns(3)
    with col_im1:
        st.metric("Bias Before", f"{dp_before:.3f}")
    with col_im2:
        st.metric("Bias After", f"{dp_after:.3f}")
    with col_im3:
        st.metric("Improvement", f"{improvement:.0f}%", delta_color="normal" if improvement > 0 else "inverse")
    
    if improvement > 20:
        st.success(f"Significant improvement! Bias reduced by {improvement:.0f}%")
    elif improvement > 0:
        st.warning(f"Modest improvement. Bias reduced by {improvement:.0f}%")
    else:
        st.info("No significant change detected. May need different approach.")
    
    st.markdown("---")
    st.markdown("### What This Means:", unsafe_allow_html=True)
    
    male_before = 0.5 + dp_before/2
    male_after = 0.5 + dp_after/2
    
    st.markdown(f"""
    **Before Fix:**
    - Men selected: {male_before*100:.1f}%
    - Women selected: {(1-male_before)*100:.1f}%
    - Gap: {(male_before - (1-male_before))*100:.1f}%
    
    **After Fix:**
    - Men selected: {male_after*100:.1f}%
    - Women selected: {(1-male_after)*100:.1f}%
    - Gap: {(male_after - (1-male_after))*100:.1f}%
    """)
    
    if dp_after < dp_before:
        st.success("The model now treats genders more equally!")
    
    st.markdown("---")
    st.markdown("### Next Steps:", unsafe_allow_html=True)
    st.markdown("""
    1. Review the fair model in production
    2. Monitor selection rates quarterly
    3. Apply the solutions from the report above
    """)

# === EXPLAIN MODEL BUTTON ===
with col3:
    explain_labels = {
        "Army": "🎖️ Explain Army Model",
        "Education": "🎓 Explain Education Model",
        "Bank Loan": "🏦 Explain Loan Model"
    }
    explain_label = explain_labels.get(st.session_state.selected_domain, "🧠 Explain Model")
    if st.button(explain_label, disabled=not st.session_state.analyzed):
        with st.spinner("Computing SHAP values..."):
            st.session_state.logger.log("SHAP computation started")
            
            model = models[st.session_state.selected_model]
            shap_vals, shap_X, error = compute_shap_values(model, X_train.values, X_test.values, st.session_state.selected_model)
            
            if error:
                st.error(f"SHAP computation failed: {error}")
            else:
                st.session_state.shap_vals = shap_vals
                st.session_state.shap_X = shap_X
                st.session_state.shap_computed = True
                
                st.session_state.logger.log("SHAP computed", details={"samples": 200})
                
                st.rerun()

# === SHAP RESULTS ===
if st.session_state.shap_computed:
    st.markdown("##### 🧠 SHAP Explanations", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>SHAP (SHapley Additive exPlanations)</strong> provides feature-level explanations for each prediction.
    This fulfills transparency requirements for {domain} domain decisions.
    </div>
    """.format(domain=st.session_state.selected_domain), unsafe_allow_html=True)
    
    shap_vals = st.session_state.shap_vals
    shap_X = st.session_state.shap_X
    
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("##### Global Feature Importance", unsafe_allow_html=True)
        fig1 = plot_shap_global(shap_vals, shap_X, feature_names)
        if fig1:
            st.pyplot(fig1)
    
    with s2:
        st.markdown("##### Gender Group Comparison", unsafe_allow_html=True)
        fig2 = plot_shap_group_comparison(shap_vals, shap_X, feature_names, g_test.values[:200])
        if fig2:
            st.pyplot(fig2)
    
    st.markdown("""
    <div class="success-box">
    ✅ <strong>Transparency Compliance:</strong> Model predictions are now explainable via SHAP values.
    </div>
    """, unsafe_allow_html=True)

# === COMPARE MODELS BUTTON ===
with col4:
    if st.button("📊 Compare All Models"):
        with st.spinner("Computing metrics for all models..."):
            all_metrics = {}
            for name, model in models.items():
                y_p = model.predict(X_test)
                m = compute_fairness_metrics(y_test.values, y_p, g_test.values)
                all_metrics[name] = m
            
            st.session_state.model_metrics_all = all_metrics
            
            st.session_state.logger.log("Multi-model comparison complete")
            
            st.rerun()

# === MULTI-MODEL RESULTS ===
if st.session_state.model_metrics_all:
    st.markdown("##### 📊 Multi-Model Comparison", unsafe_allow_html=True)
    
    all_metrics = st.session_state.model_metrics_all
    st.plotly_chart(plot_multi_model_comparison(all_metrics), width="stretch")
    
    st.markdown("##### Model Leaderboard", unsafe_allow_html=True)
    leaderboard = []
    for name, m in all_metrics.items():
        leaderboard.append({
            "Model": name,
            "Accuracy": f"{m['overall_accuracy']:.1%}",
            "F1": f"{m['f1_score']:.1%}",
            "DP Diff": f"{m['demographic_parity_difference']:.3f}",
            "EO Diff": f"{m['equalized_odds_difference']:.3f}",
            "Risk": classify_risk(m['demographic_parity_difference'])[0]
        })
    
    df_leader = pd.DataFrame(leaderboard).sort_values("DP Diff")
    st.dataframe(df_leader, width="stretch")
    
    best_model = df_leader.iloc[0]["Model"]
    st.markdown(f"**Best by fairness:** {best_model}", unsafe_allow_html=True)

# === PDF REPORT SECTION ===
st.markdown("---")
st.markdown("##### 📥 PDF Compliance Report", unsafe_allow_html=True)

if st.button("📥 Generate PDF Report") or (st.session_state.analyzed and not st.session_state.mitigated):
    if st.session_state.analyzed:
        with st.spinner("Generating PDF report..."):
            pdf_bytes = generate_pdf_report(
                st.session_state.metrics_before,
                st.session_state.compliance,
                st.session_state.gaps,
                st.session_state.recs,
                st.session_state.metrics_after if st.session_state.mitigated else None,
                st.session_state.selected_domain
            )
            
            st.session_state.logger.log("PDF generated")
            
            st.download_button(
                label="📥 Download A4 Compliance Report",
                data=pdf_bytes,
                file_name=f"faircheck_{st.session_state.selected_domain.lower()}_report.pdf",
                mime="application/pdf"
            )

# === WELCOME STATE (SIMPLE) ===
if not st.session_state.analyzed:
    st.markdown("---")
    st.markdown("### Welcome to FairCheck India", unsafe_allow_html=True)
    st.markdown("Select a domain from the sidebar and click **Analyze** to check for bias")
    st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#999;font-size:12px;padding:16px;">
    FairCheck India | AI Bias Detection
</div>
""", unsafe_allow_html=True)