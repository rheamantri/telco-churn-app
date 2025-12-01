import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt
import seaborn as sns  # <--- THIS WAS MISSING
import plotly.express as px
import streamlit.components.v1 as components
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
IMAGES_DIR = os.path.join(os.getcwd(), "images")
MODELS_DIR = os.path.join(os.getcwd(), "models")

# ===============================================================
# CONFIG & HELPER FUNCTIONS
# ===============================================================
st.set_page_config(page_title="Telco Retention Engine", layout="wide")

def st_shap(plot, height=None):
    """Helper to display SHAP JS plots in Streamlit"""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# ===============================================================
# SELF-HEALING TRAINING LOGIC
# ===============================================================
def feature_engineer_robust(df):
    """Replicated feature engineering for on-the-fly training"""
    df_feat = df.copy()
    internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    mask_no_internet = df_feat['InternetService'] == 'No'
    df_feat.loc[mask_no_internet, internet_cols] = 'No internet service'
    mask_no_phone = df_feat['PhoneService'] == 'No'
    df_feat.loc[mask_no_phone, 'MultipleLines'] = 'No phone service'
    df_feat.loc[(df_feat['tenure'] < 0) | (df_feat['tenure'] > 120), 'tenure'] = np.nan
    df_feat.loc[df_feat['MonthlyCharges'] < 0, 'MonthlyCharges'] = np.nan
    df_feat['Tenure_Bucket'] = pd.cut(df_feat['tenure'], bins=[-1, 12, 24, 48, 80], labels=['0-1y', '1-2y', '2-4y', '4-6y+'])
    df_feat['Service_Count'] = (df_feat[['PhoneService', 'MultipleLines', 'Partner', 'Dependents'] + internet_cols] == 'Yes').sum(axis=1)
    df_feat['Avg_Historical_Charge'] = df_feat['TotalCharges'] / (df_feat['tenure'] + 1)
    if 'PaymentMethod' in df_feat.columns:
        df_feat['Payment_Simple'] = df_feat['PaymentMethod'].apply(lambda x: "Automatic" if "automatic" in x else "Manual")
    return df_feat

def retrain_model_on_fly():
    """Trains the model specifically for the current environment"""
    with st.spinner("⚠️ Version Mismatch Detected. Retraining model for this server... (This takes 10s)"):
        # 1. Load
        try:
            df = pd.read_csv("Churn_Telco.csv")
        except FileNotFoundError:
            st.error("CRITICAL: 'Churn_Telco.csv' not found. Please upload it to your GitHub repository.")
            st.stop()

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['Churn_Target'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # 2. Engineer
        df_final = feature_engineer_robust(df)
        #X = df_final.drop(columns=['customerID', 'Churn', 'Churn_Target'])
        X = df_final.drop(columns=['Churn', 'Churn_Target'])
        y = df_final['Churn_Target']
        
        # 3. Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # 4. Pipeline
        num_cols = X.select_dtypes(include='number').columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'customerID' in cat_cols: 
            cat_cols.remove('customerID')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy="median"), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ]
        )
        
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(objective='binary:logistic', eval_metric='auc', scale_pos_weight=pos_weight, random_state=42, n_jobs=-1))
        ])
        
        #model.fit(X_train, y_train)
        model.fit(X_train.drop(columns=['customerID'], errors='ignore'), y_train)
        
        # 5. Save compatible version
        joblib.dump(model, os.path.join(MODELS_DIR, 'final_model.joblib'))
        X_test.to_csv(os.path.join(MODELS_DIR, "X_test_data.csv"), index=False)
        y_test.to_csv(os.path.join(MODELS_DIR, "y_test_data.csv"), index=False)
        
        return model, X_test, y_test.values.ravel()

@st.cache_resource
def load_resources():
    """Robust loader that handles version mismatches"""
    try:
        # --- UPDATE PATHS HERE ---
        model = joblib.load(os.path.join(MODELS_DIR, 'final_model.joblib'))
        X_test = pd.read_csv(os.path.join(MODELS_DIR, "X_test_data.csv"))
        y_test = pd.read_csv(os.path.join(MODELS_DIR, "y_test_data.csv")).values.ravel()
        return model, X_test, y_test
    except:
        return retrain_model_on_fly()

# ===============================================================
# APP LAYOUT
# ===============================================================

# 1. LOAD RESOURCES (ROBUST)
model, X_test, y_test = load_resources()

st.title("📡 Telco Customer Retention Command Center")

# 2. SIDEBAR: CONTROL PANEL
st.sidebar.header("⚙️ Business Constraints")
COST_FN = st.sidebar.number_input("Cost of Lost Customer (LTV $)", 100, 2000, 200, 50)
COST_FP = st.sidebar.number_input("Cost of Retention Offer ($)", 10, 200, 50, 5)
st.sidebar.divider()
st.sidebar.info("Adjust inputs to optimize the strategy.")

# 3. GLOBAL PREDICTIONS (Run Once)
X_test_noid = X_test.drop(columns=['customerID'], errors='ignore')

if 'y_probs' not in st.session_state:
    st.session_state['y_probs'] = model.predict_proba(X_test_noid)[:, 1]
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
        filter_segment = st.multiselect("Filter by Segment:", ['🔴 Critical Risk', '🟡 Target Zone', '🟢 Safe'], default=['🔴 Critical Risk', '🟡 Target Zone', '🟢 Safe'])
    with c2:
        filter_contract = st.multiselect("Filter by Contract:", results_df['Contract'].unique(), default=results_df['Contract'].unique())

    filtered_df = results_df[
        (results_df['Segment'].isin(filter_segment)) & 
        (results_df['Contract'].isin(filter_contract))
    ].sort_values('Churn_Prob', ascending=False)

    cols = ['customerID', 'Segment', 'Churn_Prob', 'Action', 'tenure', 'MonthlyCharges', 'Contract', 'PaymentMethod']

    st.dataframe(
        filtered_df[cols],  # <--- Use the new list here
        use_container_width=True,
        column_config={
            "Churn_Prob": st.column_config.ProgressColumn("Risk Score", format="%.2f", min_value=0, max_value=1),
            "MonthlyCharges": st.column_config.NumberColumn("Bill Amount", format="$%.2f")
        }
    )

    st.divider()
    st.subheader("📥 Export Data")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        # Option 1: Download ONLY the filtered list (what they see on screen)
        csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered List (CSV)",
            data=csv_filtered,
            file_name="filtered_churn_risk.csv",
            mime="text/csv",
            help="Downloads only the rows currently shown in the table above."
        )
        
    with col_dl2:
        # Option 2: Download EVERYTHING (Full database with scores)
        csv_full = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Customer Database (CSV)",
            data=csv_full,
            file_name="full_customer_risk_scores.csv",
            mime="text/csv",
            help="Downloads the entire dataset with Risk Scores and Segments for all customers."
        )

# ---------------------------------------------------------------
# TAB 2: DEEP DIVE (The "Why")
# ---------------------------------------------------------------
with tab2:
    # --- PART 1: SEGMENT PROFILER (The "Why" at a Macro Level) ---
    st.subheader("🕵️‍♀️ Segment Profiler")
    col_prof1, col_prof2 = st.columns(2)
    
    # 1. Calculate Stats
    target_stats = results_df[results_df['Churn_Prob'] >= best_thr][['tenure', 'MonthlyCharges']].mean()
    safe_stats = results_df[results_df['Churn_Prob'] < best_thr][['tenure', 'MonthlyCharges']].mean()
    
    # 2. Prepare Data for Plotly
    metric_df = pd.DataFrame({'High Risk': target_stats, 'Safe': safe_stats}).T.reset_index()
    metric_df.columns = ['Group', 'tenure', 'MonthlyCharges']
    
    # 3. Define Exact Colors
    custom_colors = {"High Risk":"#F74E3E", "Safe":"#6BE395"}

    # 4. Plot Monthly Charges
    with col_prof1:
        st.info("High Risk Customers Pay More")
        fig_mc = px.bar(metric_df, x='Group', y='MonthlyCharges', color='Group', 
                        labels={'MonthlyCharges': 'Average Monthly Charges ($)', 'Group': 'Group'},
                        color_discrete_map=custom_colors, text_auto='.2f')
        fig_mc.update_layout(bargap=0.1,
                             showlegend=False,
                             height=250,
                             margin=dict(l=0,r=0,t=0,b=0),
                             yaxis=dict(showgrid=False), xaxis=dict(showgrid=False)
                             )
        st.plotly_chart(fig_mc, use_container_width=True)
    
    # 5. Plot Tenure
    with col_prof2:
        st.info("High Risk Customers have Lower Tenure")
        fig_ten = px.bar(metric_df, x='Group', y='tenure', color='Group', 
                         labels={'tenure': 'Average Tenure', 'Group': 'Group'},
                         color_discrete_map=custom_colors, text_auto='.1f')
        fig_ten.update_layout(bargap=0.1, 
                              showlegend=False, 
                              height=250, 
                              margin=dict(l=0,r=0,t=0,b=0),
                              yaxis=dict(showgrid=False), xaxis=dict(showgrid=False)
                              )
        st.plotly_chart(fig_ten, use_container_width=True)
        fig_ten.update_xaxes(showgrid=False)
        fig_ten.update_yaxes(showgrid=False)
  
    # -----------------------------------------------------------
    # Monthly Charges Line Distribution — High Risk Customers
    # -----------------------------------------------------------
    st.subheader("Monthly Charges Distribution — High Risk Customers")

    # 1. User chooses the segment
    segment_options = ["🔴 Critical Risk", "🟡 Target Zone", "🟢 Safe"]
    selected_segment = st.selectbox("Choose Segment:", segment_options)

    # 2. Map segment names to your rules
    segment_filter_map = {
    "🔴 Critical Risk": results_df['Segment'] == "🔴 Critical Risk",
    "🟡 Target Zone": results_df['Segment'] == "🟡 Target Zone",
    "🟢 Safe": results_df['Segment'] == "🟢 Safe"
    }

    # 3. Filter data
    seg_df = results_df[segment_filter_map[selected_segment]].copy()
    # Safety check
    if len(seg_df) == 0:
        st.warning(f"No customers found in segment: {selected_segment}")
    else:
        # 4. Sort values for smooth line shape
        seg_sorted = seg_df.sort_values('MonthlyCharges').reset_index(drop=True)

        # 5. Build line chart
        fig_seg = px.line(
            seg_sorted,
            x=seg_sorted.index,
            y='MonthlyCharges',
            hover_data=['customerID', 'Contract', 'tenure', 'MonthlyCharges'],
            labels={
                'x': 'Customer Index (sorted by monthly charge)',
                'MonthlyCharges': 'Monthly Charges ($)'
            },
            title=None
        )

        # 6. Styling
        color_map = {
            "🔴 Critical Risk": "#ff4d4f",
            "🟡 Target Zone": "#f2c94c",
            "🟢 Safe": "#6BE395"
        }

        fig_seg.update_traces(line=dict(color=color_map[selected_segment], width=3))

        fig_seg.update_layout(
            showlegend=False,
            height=350,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        # Remove grid lines
        fig_seg.update_xaxes(showgrid=False)
        fig_seg.update_yaxes(showgrid=False)

        st.plotly_chart(fig_seg, use_container_width=True)
    
    # -----------------------------------------------------------
    # Tenure Distribution — Interactive Per Segment
    # -----------------------------------------------------------

    tenure_titles = {
        "🔴 Critical Risk": "Tenure Distribution — Critical Risk Customers",
        "🟡 Target Zone": "Tenure Distribution — Target Zone Customers",
        "🟢 Safe": "Tenure Distribution — Safe Customers"
    }

    st.subheader(tenure_titles[selected_segment])

    seg_df_ten = results_df[segment_filter_map[selected_segment]].copy()
    seg_df_ten = seg_df_ten.dropna(subset=['tenure'])

    if len(seg_df_ten) == 0:
        st.warning(f"No customers found in segment: {selected_segment}")
    else:
        # Convert months → years
        seg_df_ten["tenure_years"] = seg_df_ten["tenure"] / 12

        # Dynamic text for hover
        def format_tenure(x):
            return f"{int(x)} months" if x < 12 else f"{x/12:.1f} years"

        seg_df_ten["tenure_display"] = seg_df_ten["tenure"].apply(format_tenure)

        seg_sorted_ten = seg_df_ten.sort_values('tenure').reset_index(drop=True)

        fig_ten_line = px.line(
            seg_sorted_ten,
            x=seg_sorted_ten.index,
            y='tenure_years',
            hover_data={
                'customerID': True,
                'Contract': True,
                'tenure_display': True,
                'MonthlyCharges': True,
                'tenure_years': False,
                'tenure': False
            },
            labels={
                'x': 'Customer Index (sorted by tenure)',
                'tenure_years': 'Tenure (Years)'
            }
        )

        fig_ten_line.update_traces(line=dict(color=color_map[selected_segment], width=3))
        fig_ten_line.update_layout(
            showlegend=False,
            height=350,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        fig_ten_line.update_xaxes(showgrid=False)
        fig_ten_line.update_yaxes(showgrid=False)

        st.plotly_chart(fig_ten_line, use_container_width=True)


    st.divider()
    
    # --- PART 2: INDIVIDUAL DIAGNOSIS (The "Why" at a Micro Level) ---
    st.subheader("💡 Individual Customer Diagnosis (SHAP)")
    
    # Filter for High Risk customers only (Battleground + Critical)
    high_risk_custom = results_df[results_df['Churn_Prob'] > best_thr]
    
    if len(high_risk_custom) > 0:
        # A. Selector: Choose by Customer ID (Not Index)
        selected_cust_id = st.selectbox("Select Customer ID:", high_risk_custom['customerID'].head(50).tolist())
        
        if st.button("Analyze This Customer"):
            with st.spinner("Calculating Risk Factors..."):
                
                # B. Find the row index for this Customer ID
                # We need the integer location (0, 1, 2) to grab the right SHAP values
                loc_idx = results_df.index[results_df['customerID'] == selected_cust_id][0]
                
                # C. Prepare Data for SHAP (Drop ID first)
                X_test_noid = X_test.drop(columns=['customerID'], errors='ignore')
                
                # D. Calculate SHAP Values
                preprocessor = model.named_steps['preprocessor']
                X_encoded = preprocessor.transform(X_test_noid)
                feature_names = preprocessor.get_feature_names_out()
                explainer = shap.TreeExplainer(model.named_steps['classifier'])
                shap_values = explainer.shap_values(X_encoded)
                
                # E. Extract Specific Data for this Customer
                cust_shap = shap_values[loc_idx, :]
                raw_row = X_test.iloc[loc_idx] # Contains the actual values (e.g., "Fiber Optic")

                # F. Sort Risk Drivers by Impact
                df_shap = pd.DataFrame({"Feature": feature_names, "Impact": cust_shap})
                df_shap['Abs_Impact'] = df_shap['Impact'].abs()
                df_shap = df_shap.sort_values("Abs_Impact", ascending=False)
                top_risk = df_shap[df_shap['Impact'] > 0].head(3)
                
                # --- DISPLAY SECTION ---
                
                # 1. Risk Score Header
                st.markdown(f"### 🛑 Risk Profile: Customer {selected_cust_id}")
                st.write(f"Predicted Probability: **{y_probs[loc_idx]:.1%}**")

                # 2. Top 3 Risk Factors (Red Boxes with Values)
                cols = st.columns(3)
                for i, (idx, row) in enumerate(top_risk.iterrows()):
                    # Clean the feature name (remove technical prefixes)
                    feat_name = row['Feature']
                    clean_feat = feat_name.replace("cat__", "").replace("num__", "")
                    
                    # Find the Actual Value in the raw data
                    val = "N/A"
                    # Try exact match first
                    if clean_feat in raw_row:
                        val = raw_row[clean_feat]
                    else:
                        # Try fuzzy match (e.g., "Contract_Month-to-month" -> "Contract")
                        for col in X_test.columns:
                            if clean_feat.startswith(col):
                                val = raw_row[col]
                                break
                    
                    # Format numbers nicely
                    if isinstance(val, (float, int)):
                        val = f"{val:.2f}"
                    
                    # Display the Red Box
                    with cols[i]:
                        st.error(f"**{clean_feat}**\n\nValue: **{val}**")

                # 3. Full Customer Profile Table
                st.write("**Full Customer Profile:**")
                st.dataframe(raw_row.to_frame().T, use_container_width=True)

                # 4. The SHAP Plot
                st.caption("Technical Decision Path:")
                st_shap(shap.force_plot(explainer.expected_value, cust_shap, X_encoded[loc_idx,:], feature_names=feature_names), height=150)
    else:
        st.success("No High Risk customers found at the current threshold!")# ---------------------------------------------------------------
# TAB 3: MODEL LAB (Performance & Trust)
# ---------------------------------------------------------------
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

    st.subheader("2. Static Artifacts (From Training)")
    st.caption("These plots were generated during the last model training run.")
    
    # Helper to load images safely
    def show_img(name, col, label):
        # Look in images folder first, then root
        paths = [os.path.join(IMAGES_DIR, name), name]
        for p in paths:
            if os.path.exists(p):
                col.image(p, caption=label, use_container_width=True)
                return

    # Layout: 3 Columns for better visibility
    c1, c2 = st.columns(2)
    
    with c1:
        show_img("roc_curve.png", st, "ROC Curve (Model Power)")
        show_img("calibration_curve.png", st, "Calibration Curve (Reliability)")
        show_img("probability_distribution.png", st, "Probability Distribution (Separation)") # <--- ADDED
        
    with c2:
        show_img("feature_importance.png", st, "Global Feature Importance")
        show_img("actual_vs_predicted_decile_plot.png", st, "Actual vs Predicted by Decile")
        show_img("threshold_metrics.png", st, "Static Threshold Metrics") # <--- ADDED

    st.divider()
    
    st.subheader("3. Full EDA Gallery")
    
    # Collect all images
    all_imgs = []
    if os.path.exists(IMAGES_DIR):
        all_imgs.extend([os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if f.endswith('.png')])
    if not all_imgs:
        all_imgs.extend([f for f in os.listdir('.') if f.endswith('.png')])

    # List of files we ALREADY displayed above so we don't duplicate them
    shown_files = [
        "roc_curve.png", "calibration_curve.png", "feature_importance.png", 
        "actual_vs_predicted_decile_plot.png", "probability_distribution.png", 
        "threshold_metrics.png", "confusion_matrix_optimized.png"
    ]

    if all_imgs:
        # 1. Categorical Plots
        cat_imgs = [img for img in all_imgs if "churn_rate_by_" in img]
        if cat_imgs:
            st.markdown("#### 📊 Categorical Churn Rates")
            cols = st.columns(2)
            for i, img in enumerate(cat_imgs):
                cols[i % 2].image(img, caption=os.path.basename(img).replace("churn_rate_by_", "").replace(".png", ""), use_container_width=True)

        # 2. Numerical Plots
        num_imgs = [img for img in all_imgs if "violin_plot_" in img]
        if num_imgs:
            st.markdown("#### 🎻 Numerical Distributions")
            cols = st.columns(2)
            for i, img in enumerate(num_imgs):
                cols[i % 2].image(img, caption=os.path.basename(img).replace("violin_plot_", "").replace(".png", ""), use_container_width=True)

        # 3. Heatmap
        heatmap_file = next((img for img in all_imgs if "heatmap_correlation.png" in img), None)
        if heatmap_file:
            st.markdown("#### 🔥 Correlations")
            st.image(heatmap_file, caption="Feature Correlation Heatmap")