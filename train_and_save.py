import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    RocCurveDisplay,
    classification_report
)

# ===============================================================
# 0. SETUP: ENSURE FILES SAVE TO CURRENT FOLDER
# ===============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_plot(filename):
    """Helper to save plots to the script directory"""
    filepath = os.path.join(SCRIPT_DIR, filename)
    plt.savefig(filepath)
    print(f"Saved Plot: {filename}")
    plt.close()

# ===============================================================
# 1. ROBUST FEATURE ENGINEERING
# ===============================================================
def feature_engineer_robust(df):
    print("Feature Engineering started...")
    df_feat = df.copy()

    # Fix inconsistencies
    internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    mask_no_internet = df_feat['InternetService'] == 'No'
    df_feat.loc[mask_no_internet, internet_cols] = 'No internet service'
    
    mask_no_phone = df_feat['PhoneService'] == 'No'
    df_feat.loc[mask_no_phone, 'MultipleLines'] = 'No phone service'

    # Sanity Checks
    df_feat.loc[(df_feat['tenure'] < 0) | (df_feat['tenure'] > 120), 'tenure'] = np.nan
    df_feat.loc[df_feat['MonthlyCharges'] < 0, 'MonthlyCharges'] = np.nan

    # Feature Creation
    df_feat['Tenure_Bucket'] = pd.cut(df_feat['tenure'], bins=[-1, 12, 24, 48, 80], labels=['0-1y', '1-2y', '2-4y', '4-6y+'])
    df_feat['Service_Count'] = (df_feat[['PhoneService', 'MultipleLines', 'Partner', 'Dependents'] + internet_cols] == 'Yes').sum(axis=1)
    df_feat['Avg_Historical_Charge'] = df_feat['TotalCharges'] / (df_feat['tenure'] + 1)

    if 'PaymentMethod' in df_feat.columns:
        df_feat['Payment_Simple'] = df_feat['PaymentMethod'].apply(lambda x: "Automatic" if "automatic" in x else "Manual")

    return df_feat

# ===============================================================
# 2. METRICS & COST EVALUATION (NEW)
# ===============================================================
def evaluate_and_save_metrics(model, X_test, y_test, cost_fn=200, cost_fp=50):
    print("\nCalculating Metrics & Cost Trade-offs...")
    y_probs = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Capture output to string list for saving
    report_lines = []
    def log(s):
        print(s)
        report_lines.append(s)

    log(f"\n============================================================")
    log(f"   COST COMPARISON (FN=${cost_fn}, FP=${cost_fp})")
    log(f"============================================================")
    log(f"{'Method':<15} | {'Threshold':<10} | {'Total Cost ($)':<15} | {'Recall':<10} | {'Precision':<10}")
    log("-" * 75)

    # 1. Beta Comparison
    betas = [1.0, 1.5, 2.0]
    for b in betas:
        num = (1 + b**2) * precisions * recalls
        den = (b**2 * precisions + recalls)
        f_scores = np.divide(num, den, out=np.zeros_like(num), where=(den!=0))
        
        idx = np.argmax(f_scores[:-1]) if len(f_scores) > 1 else 0
        thr = thresholds[idx]
        
        # Calculate Cost
        y_pred = (y_probs >= thr).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()
        cost = (FN * cost_fn) + (FP * cost_fp)
        
        log(f"Beta {b:<10} | {thr:.4f}     | ${cost:,.2f}      | {recalls[idx]:.4f}     | {precisions[idx]:.4f}")

    # 2. Minimum Cost Optimization
    scan_thresholds = np.linspace(0, 1, 101)
    min_cost = float('inf')
    best_thr = 0.5
    
    for t in scan_thresholds:
        y_pred_t = (y_probs >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred_t)
        # Handle small dataset edge cases
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        else:
            TN, FP, FN, TP = 0, 0, 0, 0
            
        total_cost = (FN * cost_fn) + (FP * cost_fp)
        if total_cost < min_cost:
            min_cost = total_cost
            best_thr = t

    # Metrics for Optimal Threshold
    y_pred_opt = (y_probs >= best_thr).astype(int)
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    TN, FP, FN, TP = cm_opt.ravel()
    rec_opt = TP / (TP+FN) if (TP+FN)>0 else 0
    prec_opt = TP / (TP+FP) if (TP+FP)>0 else 0
    spec_opt = TN / (TN+FP) if (TN+FP)>0 else 0

    log(f"{'MIN COST':<15} | {best_thr:.4f}     | ${min_cost:,.2f}      | {rec_opt:.4f}     | {prec_opt:.4f}")
    log("-" * 75)

    log("\n--- Classification Report at Cost-Optimal Threshold ---")
    log(classification_report(y_test, y_pred_opt))

    log("\n--- Key Classification Metrics ---")
    log(f"Sensitivity (Recall): {rec_opt:.4f}")
    log(f"Precision: {prec_opt:.4f}")
    log(f"Specificity: {spec_opt:.4f}")
    log(f"All evaluation plots saved.")

    # Save to file
    report_path = os.path.join(SCRIPT_DIR, "model_performance_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n[INFO] Full report saved to: {report_path}")
    
    return best_thr

# ===============================================================
# 3. VISUALIZATION GENERATORS
# ===============================================================
def generate_eda_plots(df):
    print("Generating EDA plots...")
    
    # 1. Heatmap
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        save_plot("heatmap_correlation.png")

    # 2. Categorical Bars
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    ignore = ['customerID', 'Churn', 'Churn_Target']
    for col in cat_cols:
        if col not in ignore and col in df.columns:
            try:
                plt.figure(figsize=(8, 5))
                rate = df.groupby(col)['Churn_Target'].mean().reset_index().sort_values('Churn_Target', ascending=False)
                sns.barplot(x=col, y='Churn_Target', data=rate, palette='viridis', hue=col, legend=False)
                plt.title(f"Churn Rate by {col}")
                plt.tight_layout()
                save_plot(f"churn_rate_by_{col}.png")
            except: pass

    # 3. Violins
    plot_df = df.copy()
    plot_df['Churn_Label'] = plot_df['Churn_Target'].apply(lambda x: 'Yes' if x==1 else 'No')
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        if col in plot_df.columns:
            plt.figure(figsize=(8, 5))
            sns.violinplot(x='Churn_Label', y=col, data=plot_df, palette="muted", hue='Churn_Label', legend=False)
            plt.title(f"Distribution of {col} by Churn")
            plt.tight_layout()
            save_plot(f"violin_plot_{col}_vs_churn.png")

def generate_model_plots(model, X_test, y_test, optimal_thr=0.5):
    print("Generating Model Performance plots...")
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # 1. Feature Importance
    try:
        names = model.named_steps['preprocessor'].get_feature_names_out()
        imp = model.named_steps['classifier'].feature_importances_
        df_imp = pd.DataFrame({'feature': names, 'importance': imp}).sort_values('importance', ascending=False)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=df_imp.head(20), palette='Reds_d')
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        save_plot("feature_importance.png")
    except: pass

    # 2. ROC
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test, y_probs, name="XGBoost")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve")
    save_plot("roc_curve.png")

    # 3. Calibration
    plt.figure(figsize=(8, 8))
    CalibrationDisplay.from_predictions(y_test, y_probs, n_bins=10, name="XGBoost")
    plt.title("Calibration Curve")
    save_plot("calibration_curve.png")

    # 4. Prob Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(x=y_probs, hue=y_test, bins=30, kde=True, element="step", palette=["blue", "red"])
    plt.axvline(optimal_thr, color='black', linestyle='--', label=f"Optimal Thr: {optimal_thr:.2f}")
    plt.legend()
    plt.title("Predicted Probability Distribution")
    save_plot("probability_distribution.png")

    # 5. Decile Plot (Actual vs Predicted)
    df_res = pd.DataFrame({'actual': y_test, 'prob': y_probs})
    df_res['decile'] = pd.qcut(df_res['prob'], 10, labels=False, duplicates='drop')
    decile_stats = df_res.groupby('decile').agg(actual=('actual', 'mean'), pred=('prob', 'mean')).reset_index()
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(decile_stats))
    width = 0.35
    plt.bar(x - width/2, decile_stats['actual'], width, label='Actual Churn Rate', color='orange', alpha=0.8)
    plt.bar(x + width/2, decile_stats['pred'], width, label='Predicted Prob', color='blue', alpha=0.6)
    plt.xticks(x, decile_stats['decile'])
    plt.legend()
    plt.xlabel("Risk Decile (0=Low, 9=High)")
    plt.title("Calibration: Actual vs Predicted by Decile")
    save_plot("actual_vs_predicted_decile_plot.png")

    # 6. Confusion Matrix (at Optimal Threshold)
    y_pred = (y_probs >= optimal_thr).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"])
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues', ax=plt.gca(), values_format='d')
    plt.title(f"Confusion Matrix (Thr={optimal_thr:.2f})")
    save_plot("confusion_matrix_optimized.png")

# ===============================================================
# 4. MAIN EXECUTION
# ===============================================================
if __name__ == "__main__":
    # A. Load
    print("Loading data...")
    csv_path = os.path.join(SCRIPT_DIR, "Churn_Telco.csv")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find 'Churn_Telco.csv' in {SCRIPT_DIR}")
        exit()

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn_Target'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # B. Engineer
    df_final = feature_engineer_robust(df)

    # C. EDA Plots
    generate_eda_plots(df_final)

    # D. Split & Save Data
    X = df_final.drop(columns=['customerID', 'Churn', 'Churn_Target'])
    y = df_final['Churn_Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_test.to_csv(os.path.join(SCRIPT_DIR, "X_test_data.csv"), index=False)
    y_test.to_csv(os.path.join(SCRIPT_DIR, "y_test_data.csv"), index=False)

    # E. Train
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

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

    print("Training XGBoost Model...")
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(SCRIPT_DIR, 'final_model.joblib'))

    # F. EVALUATE & PLOT
    # We first calculate the BEST threshold based on default costs ($500/$50)
    # This report is printed to console AND saved to text file
    optimal_thr = evaluate_and_save_metrics(model, X_test, y_test, cost_fn=200, cost_fp=50)

    # We pass this optimal threshold to the plotting function so the CM matches the report
    generate_model_plots(model, X_test, y_test, optimal_thr)
    
    print("\nSUCCESS: All files, images, and report saved to project folder!")