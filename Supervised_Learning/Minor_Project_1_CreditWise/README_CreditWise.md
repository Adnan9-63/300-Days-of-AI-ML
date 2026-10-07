# 🏦 CreditWise Loan Approval System
### Minor Project 1 — Supervised ML | Apna College Prime AI/ML Batch

A machine learning project that automates loan approval decisions for **SecureTrust Bank** using historical applicant data. The system predicts whether a loan should be **Approved or Rejected** before final human verification.

---

## 📌 Problem Statement

SecureTrust Bank handles hundreds of loan applications daily through a **manual verification process** — checking income proofs, employment details, credit history, and documents. This process is:
- ⏱️ Time-consuming
- ⚖️ Biased and inconsistent
- ❌ Error-prone

This leads to two costly mistakes:
1. **Good customers get rejected** → Loss of business
2. **High-risk customers get approved** → Financial losses

**Solution:** Build an intelligent ML system that learns from historical loan data and predicts approvals automatically, accurately, and without bias.

---

## 📂 Dataset

- **Source:** SecureTrust Bank Historical Loan Data (Assignment Dataset)
- **Size:** 1,000 rows × 20 columns
- **Target:** `Loan_Approved` (1 = Approved, 0 = Rejected)
- **Class Distribution:** ~65% Rejected / ~35% Approved → **Imbalanced!**
- **Missing Values:** 50 missing values in every column → needed imputation

**Features Overview:**

| Feature | Type | Description |
|---|---|---|
| Applicant_Income | Numerical | Monthly income of applicant |
| Coapplicant_Income | Numerical | Monthly income of co-applicant |
| Age | Numerical | Applicant age |
| Credit_Score | Numerical | Credit bureau score ⭐ Most Important |
| DTI_Ratio | Numerical | Debt-to-Income ratio |
| Savings | Numerical | Savings balance |
| Collateral_Value | Numerical | Value of collateral provided |
| Loan_Amount | Numerical | Loan amount requested |
| Loan_Term | Numerical | Loan duration (months) |
| Existing_Loans | Numerical | Number of already running loans |
| Dependents | Numerical | Number of dependents |
| Employment_Status | Categorical | Salaried / Self-Employed / Business |
| Marital_Status | Categorical | Married / Single |
| Loan_Purpose | Categorical | Home / Education / Personal / Business |
| Property_Area | Categorical | Urban / Semi-Urban / Rural |
| Education_Level | Categorical | Graduate / Postgraduate / Undergraduate |
| Gender | Categorical | Male / Female |
| Employer_Category | Categorical | Govt / Private / Self |

---

## ⚙️ Tech Stack

- **Python 3**
- **Pandas & NumPy** — data manipulation
- **Matplotlib & Seaborn** — visualizations
- **Scikit-learn** — preprocessing, model training, evaluation, GridSearchCV

---

## 🔄 ML Pipeline

### 1. Data Loading & Exploration (EDA)
- Checked shape, data types, missing values
- Analyzed class imbalance in target variable
- Visualized feature distributions using boxplots and bar charts
- Generated correlation heatmap for numerical features

### 2. Data Preprocessing
- Dropped `Applicant_ID` (unique identifier — no predictive value)
- Filled **numerical** missing values with **median** (robust to outliers)
- Filled **categorical** missing values with **mode** (most frequent value)
- Encoded target: `Yes → 1`, `No → 0` using `.map()`
- Encoded categorical features using `LabelEncoder`

### 3. Train/Test Split
- 80% training / 20% testing
- Used `stratify=y` to maintain 65/35 class ratio in both splits
- `random_state=42` for reproducibility

### 4. Feature Scaling
- Applied `StandardScaler` for Logistic Regression and KNN
- Decision Trees don't need scaling — used raw features

### 5. Models Trained

| Model | Approach |
|---|---|
| Decision Tree (Baseline) | No pruning, default parameters |
| Decision Tree (Pre-Pruned) | Tuned max_depth + min_samples_leaf |
| Decision Tree (Post-Pruned) | Used ccp_alpha cost complexity pruning |
| Decision Tree (GridSearchCV) | Automated hyperparameter tuning with 5-fold CV |
| Logistic Regression (Baseline) | No regularization |
| Logistic Regression (L2 Ridge) | penalty='l2', class_weight='balanced' |
| Logistic Regression (L1 Lasso) | penalty='l1', solver='liblinear' |
| KNN | K tuned via GridSearchCV |
| Naive Bayes | GaussianNB — for continuous features |

---

## 📊 Results

| Model | Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|
| **Decision Tree (GridSearch)** ⭐ | **0.900** | **0.8507** | **0.7703** | **0.9500** |
| Decision Tree (Pre-Pruned) | 0.900 | 0.8507 | 0.7703 | 0.9500 |
| Decision Tree (Post-Pruned) | 0.900 | 0.8507 | 0.7703 | 0.9500 |
| Decision Tree (Baseline) | 0.900 | 0.8438 | 0.7941 | 0.9000 |
| Naive Bayes | 0.820 | 0.7000 | 0.7000 | 0.7000 |
| Logistic Regression (Baseline) | 0.800 | 0.6825 | 0.6515 | 0.7167 |
| Logistic Regression (L1) | 0.785 | 0.7034 | 0.6000 | 0.8500 |
| Logistic Regression (L2) | 0.785 | 0.7034 | 0.6000 | 0.8500 |
| KNN | 0.735 | 0.5546 | 0.5593 | 0.5500 |

---

## 🏆 Best Model — Decision Tree (GridSearchCV)

**F1 Score: 0.8507 | Accuracy: 90% | Recall: 95%**

The GridSearch tuned Decision Tree outperformed all other models across every metric.

---

## 💡 Why F1 Score Instead of Accuracy?

The dataset is **imbalanced** — 65% rejected, 35% approved.

A dumb model that always predicts "Rejected" would get **65% accuracy** but catch **zero approved customers** — completely useless!

F1 Score balances **Precision** and **Recall**:
- **Precision** — of everyone predicted as approved, how many actually should be?
- **Recall** — of all actually approvable customers, how many did we catch?

> F1 = 2 × (Precision × Recall) / (Precision + Recall)

**F1 cannot be fooled by class imbalance. Accuracy can.** ✅

---

## 🔑 Most Important Feature: Credit_Score

Feature importance analysis revealed that **Credit_Score** is by far the strongest predictor of loan approval — exactly as expected in real-world banking.

Top features driving loan decisions:
1. **Credit_Score** — most influential ⭐
2. **DTI_Ratio** — debt burden relative to income
3. **Savings** — financial cushion available

---

## 🏢 Business Recommendation

> The bank should focus its primary screening on **Credit_Score**, **DTI_Ratio**, and **Savings**. The Decision Tree model proves these three features drive the vast majority of loan decisions.

**Practical Impact:**
- Reduces manual review time significantly
- Eliminates human bias in approval decisions
- Recall of 95% means the model catches almost all genuinely approvable customers — minimizing business loss
- Can be used as a pre-screening tool before final human verification

---

## 💡 What I Learned

- How to handle missing values systematically — median for numerical, mode for categorical
- Why `stratify=y` is critical for imbalanced datasets in train/test split
- Difference between Pre-Pruning (stop early) and Post-Pruning (grow then cut) in Decision Trees
- How `ccp_alpha` controls cost complexity pruning
- Why `class_weight='balanced'` helps models learn minority classes better
- How GridSearchCV automates hyperparameter tuning with cross validation
- Why F1 Score is preferred over Accuracy for imbalanced classification problems
- How to extract and interpret feature importances from Decision Trees
- Why Decision Trees outperformed Logistic Regression here — non-linear relationships in financial data
- Why KNN performed worst — sensitive to scale and high dimensionality (curse of dimensionality)

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/Adnan9-63/300-Days-of-AI-ML

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the notebook
jupyter notebook CreditWise_LoanApproval.ipynb
```

---

## 📁 Project Structure

```
Minor_Project_1_CreditWise/
├── CreditWise_LoanApproval.ipynb   ← Main notebook
├── loan_approval_data.csv           ← Dataset
└── README.md                        ← This file
```

---

*Built by Adnan | First Year CS Student | IIT Patna*
*[LinkedIn](https://www.linkedin.com/in/md-adnan96) | [GitHub](https://github.com/Adnan9-63)*
