# 🌸 Iris Flower Classification — KNN vs Logistic Regression vs Naive Bayes

A machine learning project built as part of my **Supervised ML coursework (Apna College Prime Batch)**.  
The goal: automatically classify Iris flower species based on sepal and petal measurements using three different classification algorithms.

---

## 📌 What This Project Does

Given 4 physical measurements of an Iris flower — the model predicts which of 3 species it belongs to:
- 🌸 Iris-setosa
- 🌺 Iris-versicolor  
- 🌼 Iris-virginica

Three models are trained, compared, and the best one is recommended.

---

## 📂 Dataset

- **Source:** Iris Dataset (Classic ML benchmark dataset)
- **Size:** 150 rows × 6 columns
- **Target:** `Species` (3 classes — perfectly balanced, 50 each)
- **Missing Values:** None — perfectly clean dataset

**Features:**

| Feature | Description |
|---|---|
| SepalLengthCm | Length of sepal (cm) |
| SepalWidthCm | Width of sepal (cm) |
| PetalLengthCm | Length of petal (cm) |
| PetalWidthCm | Width of petal (cm) |

> **Note:** The Iris dataset is small, balanced and clean — so models perform extremely well here. Real-world data is messier, imbalanced and needs more preprocessing. To simulate a harder task, models were trained on only 50% of data but tested on 100%.

---

## ⚙️ Tech Stack

- **Python 3**
- **Pandas** — data manipulation
- **Scikit-learn** — model training, evaluation, GridSearchCV, Pipeline

---

## 🔄 ML Pipeline

### 1. Data Loading & Exploration
- Checked shape, data types, missing values
- Dropped `Id` column (no predictive value)

### 2. Feature & Target Split
- X = SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
- y = Species

### 3. Train/Test Split
- Trained on **50%** (75 rows) — intentionally limited
- Tested on **100%** (150 rows) — entire dataset
- `random_state=42` for reproducibility

### 4. Feature Scaling
- Applied `StandardScaler` — fit on training data only
- Same scaling applied to test data to avoid data leakage

### 5. Models Trained
- **KNN** — K=5 initially, then tuned using GridSearchCV
- **Logistic Regression** — max_iter=200
- **Naive Bayes** — GaussianNB (best for continuous features)

### 6. Hyperparameter Tuning
- Used **GridSearchCV with 5-Fold Cross Validation** to find best K for KNN
- Tested K values: [1, 3, 5, 7, 9, 11, 13, 15]
- **Best K = 3** with CV Score = 89.33%

### 7. Pipeline
- Built sklearn Pipeline combining StandardScaler + KNN
- Cleaner code, zero risk of data leakage

---

## 📊 Results

| Model | Accuracy | Notes |
|---|---|---|
| KNN (K=5) | 94.67% | Confused Virginica & Versicolor 5 times |
| **Logistic Regression** | **96.00%** | Best overall performance |
| Naive Bayes | 95.33% | Close second, simplest model |
| Pipeline KNN (K=3) | 95.33% | Improved after hyperparameter tuning |

**Confusion Matrix — Logistic Regression (Best Model):**
```
              Setosa  Versicolor  Virginica
Setosa      [  50        0          0  ]   ← Perfect!
Versicolor  [   0       47          3  ]
Virginica   [   0        3         47  ]
```

---

## 🏆 Recommendation

**Logistic Regression is the best model** because:

1. Highest accuracy (96%) among all models
2. Perfect classification of Iris-setosa (100% precision & recall)
3. Best F1 scores across all 3 classes
4. Simple, interpretable and fast
5. No hyperparameter tuning needed — works well out of the box

> The slight confusion between Versicolor and Virginica is expected — these two species have naturally overlapping measurements, making them harder to separate for any algorithm.

---

## 💡 What I Learned

- Difference between KNN, Logistic Regression and Naive Bayes
- When to use GaussianNB vs MultinomialNB vs BernoulliNB
- How GridSearchCV automates hyperparameter tuning using Cross Validation
- How sklearn Pipeline prevents data leakage and simplifies code
- Why Standardization is critical for distance-based algorithms like KNN
- How to read and interpret a multiclass confusion matrix
- Why a perfectly accurate model on clean data doesn't reflect real-world performance

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/Adnan9-63/300-Days-of-AI-ML

# Install dependencies
pip install pandas scikit-learn

# Run the notebook
jupyter notebook Assingment_3.ipynb
```

---

*Built by Adnan | First Year CS Student | [LinkedIn](https://www.linkedin.com/in/md-adnan96)*
