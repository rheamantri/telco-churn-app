import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit.components.v1 as components
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Telco Retention Engine", layout="wide")

# ===============================================================
# PATH CONFIGURATION
# ===============================================================
# We assume the app runs from the root of the repo
IMAGES_DIR = os.path.join(os.getcwd(), "images")
MODELS_DIR = os.path.join(os.getcwd(), "models")

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# ===============================================================
# SELF-HEALING & LOADING
# ===============================================================
def retrain_model_on_fly():
    """Trains model if file is missing (simplified for brevity)"""
    st.error("CRITICAL: Model files not found in 'models/' directory. Please run 'train_and_save.py' locally and push the folder.")
    st.stop()

@st.cache_resource
def load_resources():
    try:
        model = joblib.load(os.path.join(MODELS_DIR, 'final_model.joblib'))
        X_test = pd.read_csv(os.path.join(MODELS_DIR, "X_test_data.csv"))
        y_test = pd.read_csv(os.path.join(MODELS_DIR, "y_test_data.csv")).values.ravel()
        return model, X_test, y_test
    except Exception as e:
        # Fallback to creating folders if they don't exist (e.g. first run on cloud)
        os.makedirs(MODELS_DIR, exist_ok=True)
        retrain_model_on_fly()

# ===============================================================
# MAIN APP
# ===============================================================
try:
    model, X_test, y_test = load_resources()
except:
    st.stop()

st.title("📡 Telco Customer Retention Command Center")

# Sidebar
st.sidebar.header("⚙️ Business Constraints")
COST_FN = st.sidebar.number_input("Cost of Lost Customer (LTV $)", 100, 2000, 200, 50)
COST_FP = st.sidebar.number_input("Cost of Retention Offer ($)", 10, 200, 50, 5)

if 'y_probs' not in st.session_state:
    st.session_state['y_probs'] = model.predict_proba(X_test)[:, 1]
y_probs = st.session_state['y_probs']

# Optimization
thresholds = np.linspace(0, 1, 101)
costs = []
for t in thresholds:
    y_pred_t = (y_probs >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    TN, FP, FN, TP = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    costs.append((FN * COST_FN) + (FP * COST_FP))

best_thr = thresholds[np.argmin(costs)]
min_cost = min(costs)

# Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Strategy", "🔍 Deep Dive", "🤖 Model Lab"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal Threshold", f"{best_thr:.1%}")
    c2.metric("Projected Cost", f"${min_cost:,.0f}")
    c3.metric("Savings", f"${(sum(y_test)*COST_FN) - min_cost:,.0f}")
    
    st.subheader("Financial Optimization")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(thresholds, costs, color='green')
    ax.axvline(best_thr, color='red', linestyle='--')
    st.pyplot(fig)

with tab2:
    st.subheader("Individual Diagnosis")
    if st.checkbox("Run SHAP"):
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
        preprocessor = model.named_steps['preprocessor']
        X_enc = preprocessor.transform(X_test)
        shap_values = explainer.shap_values(X_enc)
        st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_enc[0,:], feature_names=preprocessor.get_feature_names_out()), height=150)

with tab3:
    st.header("Model Performance")
    
    col1, col2 = st.columns(2)
    # Helper to load images safely
    def show_img(name, col):
        path = os.path.join(IMAGES_DIR, name)
        if os.path.exists(path): col.image(path)
    
    with col1:
        st.subheader("Model Diagnostics")
        show_img("roc_curve.png", st)
        show_img("calibration_curve.png", st)
        show_img("actual_vs_predicted_decile_plot.png", st)

    with col2:
        st.subheader("Feature Importance")
        show_img("feature_importance.png", st)
        
    st.divider()
    st.subheader("Full EDA Gallery")
    if os.path.exists(IMAGES_DIR):
        all_imgs = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')]
        rows = st.columns(2)
        for i, img in enumerate(all_imgs):
            # Skip the main ones we already showed
            if img not in ["roc_curve.png", "calibration_curve.png", "feature_importance.png"]:
                rows[i % 2].image(os.path.join(IMAGES_DIR, img), caption=img)