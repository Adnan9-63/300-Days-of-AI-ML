# 🛒 ShopSmart Purchase Prediction — Decision Tree Classifier

A machine learning project built as part of my **Supervised ML coursework (Apna College Prime Batch)**.  
The goal: predict whether an e-commerce visitor will make a purchase based on their browsing session behavior.

---

## 📌 What This Project Does

Given 17 features about a user's browsing session — pages visited, time spent, bounce rates, visitor type etc. — the model predicts whether they will **make a purchase (Revenue=1)** or **leave without buying (Revenue=0)**.

Key challenge: the dataset is **heavily imbalanced** (84.5% non-buyers vs 15.5% buyers), so F1 Score is used as the primary metric instead of accuracy.

---

## 📂 Dataset

- **Source:** ShopSmart E-Commerce (Assignment Dataset)
- **Size:** 12,330 rows × 18 columns
- **Target:** `Revenue` (1 = Purchase, 0 = No Purchase)
- **Class Distribution:** 84.5% No Purchase / 15.5% Purchase (imbalanced!)

**Key Features:**

| Feature | Description |
|---|---|
| ProductRelated | Number of product pages visited |
| ProductRelated_Duration | Time spent on product pages |
| BounceRates | % of visitors who leave after one page |
| ExitRates | % of exits from each page |
| PageValues | Average value of pages before transaction |
| VisitorType | New vs Returning visitor |
| Month | Month of visit |
| Weekend | Whether visit was on weekend |

---

## ⚙️ Tech Stack

- **Python 3**
- **Pandas** — data manipulation
- **Matplotlib & Seaborn** — visualizations
- **Scikit-learn** — model training, pruning, GridSearchCV

---

## 🔄 ML Pipeline

### 1. Data Loading & EDA
- Checked shape, dtypes, missing values
- Analyzed class imbalance in target variable

### 2. Preprocessing
- Converted bool columns (`Revenue`, `Weekend`) to int
- Applied `LabelEncoder` to categorical columns (`Month`, `VisitorType`)

### 3. Train/Test Split
- 80% train / 20% test
- Used `stratify=y` to maintain class ratio in both splits

### 4. Models Built

| Model | Approach |
|---|---|
| Baseline | No pruning, default parameters |
| Pre-Pruned | Tuned max_depth + min_samples_leaf |
| Post-Pruned | Used ccp_alpha (cost complexity pruning) |
| GridSearchCV | Automated hyperparameter tuning with 5-fold CV |

### 5. Evaluation Results

| Model | F1 Score | vs Benchmark (0.55) |
|---|---|---|
| Baseline | ~0.57 | ✅ Beats |
| Pre-Pruned | ~0.61 | ✅ Beats |
| Post-Pruned | ~0.60 | ✅ Beats |
| **GridSearchCV Best** | **~0.63** | **✅ Beats** |

---

## 🏆 Recommendation

**GridSearchCV optimized Decision Tree is the best model** because:

1. Highest F1 Score — best balance of precision and recall
2. Automatically found optimal combination of hyperparameters
3. `class_weight='balanced'` handles class imbalance effectively
4. Pruning prevents overfitting — generalizes well to new visitors

> **Most Important Feature:** `PageValues` — visitors who browse high-value pages are significantly more likely to make a purchase. Marketing should focus on driving traffic to these pages.

---

## 💡 What I Learned

- Why accuracy is misleading for imbalanced datasets and when to use F1 Score
- The difference between Pre-Pruning (stop early) and Post-Pruning (grow then cut)
- How `ccp_alpha` controls cost complexity pruning
- Why `stratify=y` is important when splitting imbalanced datasets
- How `class_weight='balanced'` helps models learn minority classes better
- How to use GridSearchCV to automate hyperparameter tuning
- How to visualize and interpret a Decision Tree using `plot_tree`
- How to extract and visualize feature importances

*Built by Adnan | Learning AIML| [LinkedIn](https://www.linkedin.com/in/md-adnan96)*

