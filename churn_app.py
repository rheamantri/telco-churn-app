import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit.components.v1 as components
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_curve
)

# ===============================================================
# CONFIG & HELPER FUNCTIONS
# ===============================================================
st.set_page_config(page_title="Telco Retention Engine", layout="wide")

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

@st.cache_resource
def load_model():
    return joblib.load('final_model.joblib')

@st.cache_data
def load_data():
    X = pd.read_csv("X_test_data.csv")
    y = pd.read_csv("y_test_data.csv")
    return X, y.values.ravel()

# ===============================================================
# APP LAYOUT
# ===============================================================

# 1. LOAD RESOURCES
try:
    model = load_model()
    X_test, y_test = load_data()
except FileNotFoundError:
    st.error("Model or Data files not found. Please run 'train_and_save.py' first.")
    st.stop()

st.title("📡 Telco Customer Retention Command Center")

# 2. SIDEBAR: CONTROL PANEL
st.sidebar.header("⚙️ Business Constraints")
COST_FN = st.sidebar.number_input("Cost of Lost Customer (LTV $)", 100, 2000, 500, 50)
COST_FP = st.sidebar.number_input("Cost of Retention Offer ($)", 10, 200, 50, 5)
st.sidebar.divider()
st.sidebar.info("Adjust inputs to optimize the strategy.")

# 3. GLOBAL PREDICTIONS (Run Once)
if 'y_probs' not in st.session_state:
    st.session_state['y_probs'] = model.predict_proba(X_test)[:, 1]
y_probs = st.session_state['y_probs']

# 4. OPTIMIZATION ENGINE (Live Calculation)
thresholds = np.linspace(0, 1, 101)
costs = []
for t in thresholds:
    y_pred_t = (y_probs >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
    else:
        TN, FP, FN, TP = 0, 0, 0, 0
    total_cost = (FN * COST_FN) + (FP * COST_FP)
    costs.append(total_cost)

min_cost_idx = np.argmin(costs)
best_thr = thresholds[min_cost_idx]
min_cost = costs[min_cost_idx]

# Assign Segments
results_df = X_test.copy()
results_df['Churn_Prob'] = y_probs
results_df['Actual_Churn'] = y_test

def assign_segment(p):
    if p >= 0.85: return '🔴 Critical Risk', 'Call / Win-Back'
    elif p >= best_thr: return '🟡 Target Zone', 'Send Discount'
    else: return '🟢 Safe', 'No Action'

results_df['Segment'], results_df['Action'] = zip(*results_df['Churn_Prob'].apply(assign_segment))

# ===============================================================
# TABS INTERFACE
# ===============================================================
tab1, tab2, tab3 = st.tabs(["🚀 Strategy & Operations", "🔍 Deep Dive Analysis", "🤖 Model Lab & EDA"])

# ---------------------------------------------------------------
# TAB 1: EXECUTIVE DASHBOARD (Strategy)
# ---------------------------------------------------------------
with tab1:
    # A. KPI ROW
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Optimal Risk Threshold", f"{best_thr:.1%}", delta="Dynamic Cutoff")
    with col2:
        st.metric("Projected Total Cost", f"${min_cost:,.0f}", delta="Minimized")
    with col3:
        churn_count = np.sum(y_test)
        do_nothing_cost = churn_count * COST_FN
        savings = do_nothing_cost - min_cost
        st.metric("Estimated Savings", f"${savings:,.0f}", delta="vs. No Action")
    with col4:
        at_risk_revenue = results_df[results_df['Churn_Prob'] >= best_thr]['MonthlyCharges'].sum()
        st.metric("Monthly Revenue at Risk", f"${at_risk_revenue:,.0f}", delta="Targeted Group")

    st.divider()

    # B. COST CURVE (Live Plot)
    st.subheader("📉 Financial Optimization")
    col_chart, col_seg = st.columns([2, 1])
    
    with col_chart:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(thresholds, costs, color='green', linewidth=2, label='Total Cost')
        ax.axvline(best_thr, color='red', linestyle='--', label=f'Optimal: {best_thr:.2f}')
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Cost ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col_seg:
        st.write("**Customer Segmentation**")
        seg_counts = results_df['Segment'].value_counts()
        fig_pie = px.pie(names=seg_counts.index, values=seg_counts.values, hole=0.4, color_discrete_sequence=['green', 'red', 'gold'])
        fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    # C. ACTION LIST
    st.subheader("📋 Actionable Customer List")
    c1, c2 = st.columns(2)
    with c1:
        filter_segment = st.multiselect("Filter by Segment:", ['🔴 Critical Risk', '🟡 Target Zone'], default=['🔴 Critical Risk', '🟡 Target Zone'])
    with c2:
        filter_contract = st.multiselect("Filter by Contract:", results_df['Contract'].unique(), default=results_df['Contract'].unique())

    filtered_df = results_df[
        (results_df['Segment'].isin(filter_segment)) & 
        (results_df['Contract'].isin(filter_contract))
    ].sort_values('Churn_Prob', ascending=False)

    st.dataframe(
        filtered_df[['Segment', 'Churn_Prob', 'Action', 'tenure', 'MonthlyCharges', 'Contract', 'PaymentMethod']],
        use_container_width=True,
        column_config={
            "Churn_Prob": st.column_config.ProgressColumn("Risk Score", format="%.2f", min_value=0, max_value=1),
            "MonthlyCharges": st.column_config.NumberColumn("Bill Amount", format="$%.2f")
        }
    )

# ---------------------------------------------------------------
# TAB 2: DEEP DIVE (The "Why")
# ---------------------------------------------------------------
with tab2:
    st.subheader("🕵️‍♀️ Segment Profiler")
    col_prof1, col_prof2 = st.columns(2)
    
    target_stats = results_df[results_df['Churn_Prob'] >= best_thr][['tenure', 'MonthlyCharges']].mean()
    safe_stats = results_df[results_df['Churn_Prob'] < best_thr][['tenure', 'MonthlyCharges']].mean()
    
    with col_prof1:
        st.info("High Risk Customers Pay More")
        metric_df = pd.DataFrame({'High Risk': target_stats, 'Safe': safe_stats}).T
        st.bar_chart(metric_df['MonthlyCharges'])
    
    with col_prof2:
        st.info("High Risk Customers have Lower Tenure")
        st.bar_chart(metric_df['tenure'])

    st.divider()
    
    st.subheader("💡 Individual Customer Diagnosis (SHAP)")
    if len(filtered_df) > 0:
        selected_cust_id = st.selectbox("Select Customer Index:", filtered_df.index[:20])
        if st.button("Analyze This Customer"):
            with st.spinner("Calculating Risk Factors..."):
                preprocessor = model.named_steps['preprocessor']
                X_encoded = preprocessor.transform(X_test)
                feature_names = preprocessor.get_feature_names_out()
                explainer = shap.TreeExplainer(model.named_steps['classifier'])
                shap_values = explainer.shap_values(X_encoded)
                
                loc_idx = X_test.index.get_loc(selected_cust_id)
                st.write(f"**Customer {selected_cust_id} Risk Score: {y_probs[loc_idx]:.1%}**")
                st_shap(shap.force_plot(explainer.expected_value, shap_values[loc_idx,:], X_encoded[loc_idx,:], feature_names=feature_names), height=150)

# ---------------------------------------------------------------
# TAB 3: MODEL LAB (Performance & Trust)
# ---------------------------------------------------------------
with tab3:
    st.header("🤖 Model Performance & EDA Gallery")
    
    # Live Metrics (Calculated on the fly)
    y_pred_current = (y_probs >= best_thr).astype(int)
    report = classification_report(y_test, y_pred_current, output_dict=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{report['accuracy']:.2%}")
    m2.metric("Precision (Churn)", f"{report['1']['precision']:.2%}")
    m3.metric("Recall (Churn)", f"{report['1']['recall']:.2%}")
    m4.metric("F1-Score", f"{report['1']['f1-score']:.2f}")

    st.divider()

    st.subheader("1. Dynamic Diagnostics (Updates with Settings)")
    col_dyn1, col_dyn2 = st.columns(2)
    
    # DYNAMIC CONFUSION MATRIX
    with col_dyn1:
        st.write("**Confusion Matrix (Live)**")
        cm = confusion_matrix(y_test, y_pred_current)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        ax_cm.set_title(f'Threshold = {best_thr:.2f}')
        st.pyplot(fig_cm)
    
    # DYNAMIC THRESHOLD PLOT
    with col_dyn2:
        st.write("**Threshold Trade-offs (Live)**")
        precisions, recalls, thr_curve = precision_recall_curve(y_test, y_probs)
        fig_thr, ax_thr = plt.subplots()
        ax_thr.plot(thr_curve, precisions[:-1], label='Precision', color='blue')
        ax_thr.plot(thr_curve, recalls[:-1], label='Recall', color='green')
        ax_thr.axvline(best_thr, color='red', linestyle='--', label='Current Thr')
        ax_thr.set_xlabel('Threshold')
        ax_thr.set_title("Precision vs Recall Impact")
        ax_thr.legend()
        ax_thr.grid(True, alpha=0.3)
        st.pyplot(fig_thr)

    st.divider()

    st.subheader("2. Static Artifacts (Model Quality)")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if os.path.exists("roc_curve.png"): st.image("roc_curve.png", caption="ROC Curve (Invariant)")
        if os.path.exists("calibration_curve.png"): st.image("calibration_curve.png", caption="Calibration Curve")
    with col_img2:
        if os.path.exists("actual_vs_predicted_decile_plot.png"): st.image("actual_vs_predicted_decile_plot.png", caption="Risk Decile Calibration")
        if os.path.exists("feature_importance.png"): st.image("feature_importance.png", caption="Global Feature Importance")

    st.divider()
    
    st.subheader("3. Full EDA Gallery")
    all_files = os.listdir('.')
    
    st.markdown("#### 📊 Categorical Churn Rates")
    cat_plots = [f for f in all_files if f.startswith('churn_rate_by_')]
    if cat_plots:
        cols = st.columns(2)
        for i, img_file in enumerate(cat_plots):
            with cols[i % 2]:
                st.image(img_file, use_container_width=True)
    
    st.markdown("#### 🎻 Numerical Distributions")
    num_plots = [f for f in all_files if f.startswith('violin_plot_')]
    if num_plots:
        cols = st.columns(2)
        for i, img_file in enumerate(num_plots):
            with cols[i % 2]:
                st.image(img_file, use_container_width=True)
    
    st.markdown("#### 🔥 Correlations")
    if os.path.exists("heatmap_correlation.png"): 
        st.image("heatmap_correlation.png")