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

st.markdown("""
<style>
    /* White Theme */
    .stApp {
        background: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background: #f8f9fa;
    }
    /* Card styles */
    .metric-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    /* Header */
    .header-banner {
        background: white;
        border-bottom: 3px solid #0d6efd;
        padding: 20px;
        margin-bottom: 20px;
    }
    .header-banner h1 {
        color: #212529;
        font-weight: 300;
        letter-spacing: -1px;
    }
    /* Status badges */
    .badge-compliant {
        background: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 20px;
    }
    .badge-partial {
        background: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 20px;
    }
    .badge-noncompliant {
        background: #f8d7da;
        color: #721c24;
        padding: 4px 12px;
        border-radius: 20px;
    }
    /* Info boxes */
    .info-box {
        background: white;
        border-left: 4px solid #0d6efd;
        padding: 16px;
        margin: 12px 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #198754;
        padding: 16px;
        margin: 12px 0;
        border-radius: 0 8px 8px 0;
    }
    .warn-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 16px;
        margin: 12px 0;
        border-radius: 0 8px 8px 0;
    }
    /* Section header */
    .section-header {
        background: #f8f9fa;
        padding: 12px 16px;
        border-radius: 8px;
        font-weight: 600;
        color: #495057;
    }
    /* Remove Streamlit decorations */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Clean buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }
    /* Metrics */
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px;
    }
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
        <div style="background:#1e3a5f;padding:20px;border-radius:10px;text-align:center;">
            <h2 style="color:#3498db;margin:0;">👨 Men</h2>
            <h1 style="color:#fff;margin:10px 0;">{:.1f}%</h1>
            <p style="color:#888;">Selected</p>
        </div>
        """.format(male_pct), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background:#1e3a5f;padding:20px;border-radius:10px;text-align:center;">
            <h2 style="color:#e91e63;margin:0;">👩 Women</h2>
            <h1 style="color:#fff;margin:10px 0;">{:.1f}%</h1>
            <p style="color:#888;">Selected</p>
        </div>
        """.format(female_pct), unsafe_allow_html=True)
    
    with col3:
        gap_color = "#e74c3c" if gap > 10 else "#f39c12" if gap > 5 else "#27ae60"
        st.markdown("""
        <div style="background:#1e3a5f;padding:20px;border-radius:10px;text-align:center;">
            <h2 style="color:{};margin:0;">⚖️ Gap</h2>
            <h1 style="color:{};margin:10px 0;">{:.1f}%</h1>
            <p style="color:#888;">Difference</p>
        </div>
        """.format(gap_color, gap_color, gap), unsafe_allow_html=True)
    
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
<div style="text-align:center;color:#999;font-size:12px;padding:10px;">
    FairCheck India | AI Fairness Detection
</div>
""", unsafe_allow_html=True)