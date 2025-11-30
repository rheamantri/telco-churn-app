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

| **1. Dynamic Cost Optimization** | **2. Actionable Segmentation** |
|:<img width="1395" height="298" alt="image" src="https://github.com/user-attachments/assets/54a33e30-0807-41f2-8f44-101703e32d1e" />
:|:<img width="1120" height="560" alt="image" src="https://github.com/user-attachments/assets/76080ca1-6c6a-43dc-a7cc-f80e18d3e802" />
:|
| The model recalculates the decision boundary instantly as you adjust business costs. | Customers are bucketed into "Critical" (Call them), "Target" (Discount them), and "Safe" (Ignore them). |
| *<img width="1126" height="438" alt="image" src="https://github.com/user-attachments/assets/f1435da2-37ab-48af-aea8-82638999c1d8" />
* |

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
