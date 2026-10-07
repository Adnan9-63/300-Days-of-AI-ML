# 🏠 House Price Prediction — Linear Regression

A machine learning project built as part of my **Supervised ML coursework (Apna College Prime Batch)**.  
The goal: predict residential house prices using historical property data.

---

## 📌 What This Project Does

Given features like lot area, year built, basement size, zoning type, and building type — the model predicts the **market sale price** of a house.

This project covers the full ML pipeline:
- Data cleaning & preprocessing
- Categorical encoding
- Train/test splitting
- Model training
- Performance evaluation

---

## 📂 Dataset

- **Source:** HomeVista Properties (Assignment Dataset)
- **Size:** 2,919 rows × 13 columns
- **Target:** `SalePrice` (house sale price in USD)

**Features used:**

| Feature | Type | Description |
|---|---|---|
| MSSubClass | Numerical | Building class type |
| MSZoning | Categorical | Zoning classification |
| LotArea | Numerical | Lot size (sq ft) |
| LotConfig | Categorical | Lot configuration |
| BldgType | Categorical | Type of dwelling |
| OverallCond | Numerical | Overall condition (1–10) |
| YearBuilt | Numerical | Construction year |
| YearRemodAdd | Numerical | Remodel year |
| Exterior1st | Categorical | Exterior covering type |
| BsmtFinSF2 | Numerical | Finished basement area |
| TotalBsmtSF | Numerical | Total basement area |

---

## ⚙️ Tech Stack

- **Python 3**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Scikit-learn** — model training & evaluation

---

## 🔄 ML Pipeline

### 1. Data Cleaning
- Dropped the `Id` column (no predictive value)
- Filled missing `SalePrice` values with column mean
- Dropped remaining rows with null values

### 2. Categorical Encoding
- Applied **One-Hot Encoding** using `pd.get_dummies()` with `drop_first=True`
- Converted: `MSZoning`, `LotConfig`, `BldgType`, `Exterior1st`

### 3. Train/Test Split
- 80% training / 20% testing
- Used `random_state=0` for reproducibility

### 4. Model Training
- Trained a **Linear Regression** model using scikit-learn

### 5. Evaluation

| Metric | Score |
|---|---|
| R² Score | 0.374 |
| MAE | $30,829 |
| RMSE | $41,138 |
| MAPE | 18.7% |

> The model explains ~37% of price variation. On average, predictions are off by ~19% of the actual price.

---

## 💡 What I Learned

- How to handle missing values (drop vs fill with mean)
- Why One-Hot Encoding is used instead of label encoding for categorical features
- The importance of proper train/test splitting to avoid data leakage
- How to interpret R², MAE, and RMSE as evaluation metrics
- Why RMSE penalizes large errors more than MAE

---


## 📈 Future Improvements

- Try advanced models (Random Forest, XGBoost)
- Apply feature engineering to improve R² score
- Use cross-validation for more robust evaluation

---

*Built by Adnan | First Year CS Student | [LinkedIn](https://www.linkedin.com/in/md-adnan96)*
