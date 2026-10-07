# 🏢 Employee Turnover Prediction — Logistic Regression + Regularization

A machine learning project built as part of my **Supervised ML coursework (Apna College Prime Batch)**.  
The goal: predict whether an employee is likely to leave the company using Logistic Regression with L1 and L2 Regularization.

---

## 📌 What This Project Does

Given features like job satisfaction, monthly income, work-life balance, training hours, and annual bonus — the model predicts whether an employee will **leave (1)** or **stay (0)**.

This project covers:
- Building a baseline Logistic Regression model
- Improving it using L1 (Lasso) and L2 (Ridge) Regularization
- Comparing all 3 models and recommending the best one

---

## 📂 Dataset

- **Source:** TalentCore Pvt. Ltd. (Assignment Dataset)
- **Size:** 1,350 rows × 16 columns
- **Target:** `Employee_Turnover` (1 = Left, 0 = Stayed)
- **Missing Values:** None — clean dataset

**Key Features:**

| Feature | Description |
|---|---|
| Job_Satisfaction | Satisfaction level with job |
| Performance_Rating | Employee performance score |
| Years_At_Company | Years worked in company |
| Work_Life_Balance | Balance between work and personal life |
| Monthly_Income | Monthly salary |
| Annual_Bonus | Yearly bonus received |
| Training_Hours | Training hours attended |
| Annual_Bonus_Squared | Engineered feature (bonus²) |
| Annual_Bonus_Training_Hours_Interaction | Interaction between bonus and training |

---

## ⚙️ Tech Stack

- **Python 3**
- **Pandas** — data manipulation
- **Scikit-learn** — model training & evaluation

---

## 🔄 ML Pipeline

### 1. Data Loading & Exploration
- Checked shape, data types, and missing values
- Dataset was already pre-scaled (all values between 0 and 1)
- No preprocessing needed

### 2. Feature & Target Split
- X = all columns except `Employee_Turnover`
- y = `Employee_Turnover`

### 3. Train/Test Split
- 80% training / 20% testing
- `random_state=42` for reproducibility

### 4. Model Training
Three models trained and compared:
- **Baseline** — Logistic Regression with no regularization
- **L2 Ridge** — penalty='l2', C=1.0
- **L1 Lasso** — penalty='l1', C=0.5, solver='liblinear'

### 5. Evaluation

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Baseline | 85.93% | 85.95% | 83.20% | 84.55% |
| L2 Ridge | 85.93% | 87.18% | 81.60% | 84.30% |
| **L1 Lasso** | **87.04%** | **88.14%** | **83.20%** | **85.60%** |

---

## 🏆 Recommendation

**L1 Lasso is the best model** because:

1. Highest accuracy (87.04%) among all three models
2. Highest precision (88.14%) — fewer false alarms
3. Same Recall as baseline (83.20%) — catches same number of at-risk employees
4. Best F1 Score (85.60%) — best balance of precision and recall
5. Automatically eliminates useless features → simpler and more interpretable for the HR team

> For HR problems, **Recall is the most critical metric** — missing an employee who is about to leave costs the company in recruitment, training, and lost productivity. L1 Lasso maintains the best recall while outperforming on all other metrics.

---

## 💡 What I Learned

- Why Logistic Regression is used for binary classification instead of Linear Regression
- How the Sigmoid function converts outputs to probabilities
- The difference between L1 and L2 regularization — L1 eliminates features, L2 just shrinks them
- How the C parameter controls regularization strength (C = 1/λ)
- Why scaling is unnecessary when data is already normalized
- How to evaluate classification models using Accuracy, Precision, Recall, and F1 Score
- Why Recall matters more than Accuracy in HR/medical problems

---



*Built by Adnan | First Year CS Student | [LinkedIn](https://www.linkedin.com/in/md-adnan96)*
