# 📡 Telco Retention Engine: Optimizing for Profit, Not Just Accuracy

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_LINK_HERE)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-Production-green)

## 💡 The "Why"
In churn prediction, **Accuracy is a trap.**

A model can achieve 95% accuracy by simply predicting "No Churn" for everyone (since churn is rare). Even balanced models often fail because they treat all errors the same.
- **Missing a Churner (False Negative):** Costs **\$500+** in lost Customer Lifetime Value (LTV).
- **Wrongly Flagging a Loyal User (False Positive):** Costs **\$50** in unnecessary discounts.

If you optimize for Accuracy, you bleed money. **This engine optimizes for Profit.**

---

## ⚡ Distinctive Features

### 1. 💰 Financial Optimization Layer
Instead of picking a random probability threshold (like the default 0.5), this app calculates the **Expected Value** of every decision.
- Users input their specific `Cost_FN` and `Cost_FP` in the dashboard.
- The engine runs a simulation across 100+ thresholds to find the exact "sweet spot" that minimizes financial loss.

### 2. 🛡️ "Self-Healing" Pipeline
Production environments are messy. Python versions change, dependencies break.
- I implemented a defensive loading mechanism.
- If the app detects a version mismatch (e.g., Local Python 3.9 vs Cloud Python 3.13) or corrupted model file, it automatically triggers a **Hot-Retrain** sequence to rebuild the model on the fly without crashing.

### 3. 🔍 "Glass Box" Explainability
Stakeholders don't trust "Black Box" predictions.
- I integrated **SHAP (SHapley Additive exPlanations)** to explain *why* specific customers are flagged.
- Marketing teams can see if a high-risk customer is leaving because of **Price** (send a coupon) or **Service** (send a tech).

---

## 📸 How it Works
There are 3 tabs: 
1. **Strategy and Operations**
<img width="1457" height="782" alt="image" src="https://github.com/user-attachments/assets/8be6fc09-c7fc-4907-b3df-9c0eefc7fa2e" />
| The model recalculates the decision boundary instantly as you adjust business costs. | Customers are bucketed into "Critical" (Call them), "Target" (Discount them), and "Safe" (Ignore them).|

2. **Deep Dive Analysis**
<img width="1454" height="741" alt="image" src="https://github.com/user-attachments/assets/7bfd1125-5ed3-4aa0-a173-59e50f6753ab" />

3. Model & Lab EDA
<img width="1434" height="764" alt="image" src="https://github.com/user-attachments/assets/26c36b05-d241-4e05-9d43-9bbcd56640d5" />
<img width="1233" height="756" alt="image" src="https://github.com/user-attachments/assets/6cd519a1-b6db-4275-a705-56581048791b" />
<img width="1247" height="761" alt="image" src="https://github.com/user-attachments/assets/b95def0c-b867-4e8d-ba3f-8ff45aadd03f" />





---

## 🛠️ How to Run Locally

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/rheamantri/telco-churn-app.git](https://github.com/rheamantri/telco-churn-app.git)
   cd telco-churn-app
   
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   
3. **Launch the Command Center:**
   ```bash
   streamlit run churn_app.py
