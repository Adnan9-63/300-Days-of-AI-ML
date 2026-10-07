# 🛒 SmartCart Customer Segmentation System

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised-blue)
![Python](https://img.shields.io/badge/Python-3.x-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📌 Problem Statement
SmartCart is a growing e-commerce platform with **2240 customers** and **22 features**.
The company used generic marketing for all customers — resulting in inefficient campaigns
and missed revenue opportunities.

**Goal:** Segment customers into meaningful groups using Unsupervised ML to enable
personalised marketing and improve customer retention.

---

## 📊 Dataset
- **2240 customers**, **22 features**
- Demographics, Purchase Behaviour, Purchase Frequency, Customer Feedback
- Source: SmartCart internal customer data

---

## 🔧 Tech Stack
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SciPy (Hierarchical Clustering)

---

## 🚀 Project Pipeline

Data Loading → EDA → Preprocessing → K-Means → Hierarchical → DBSCAN → Insights


---

## 📈 What I Did

### 1. EDA
- Income distribution analysis
- Spending pattern visualization
- Correlation heatmap — discovered kids at home = lower spending

### 2. Preprocessing
- Removed 28 outliers (Income > 200k, Age > 100)
- Filled 24 missing Income values with median
- Feature Engineering: Age, Total_Spending, Total_Children, Total_Purchases
- Encoded Education (ordinal) and Marital Status (binary)
- StandardScaler applied for fair distance calculation

### 3. K-Means Clustering (K=3)
- Elbow Method + Silhouette Score used to find optimal K
- **3 distinct customer segments discovered**

### 4. Hierarchical Clustering
- Ward linkage method
- Dendrogram confirmed K=3
- 95% agreement with K-Means results

### 5. DBSCAN
- K-Distance graph used to find optimal eps=3.5
- Detected **94 VIP outlier customers** (4.25%)

---

## 🎯 Key Results

| Segment | Customers | Avg Income | Avg Spending |
|---|---|---|---|
| 💎 Premium | 572 (26%) | ₹75,988 | ₹1,383 |
| 🏠 Moderate | 600 (27%) | ₹58,394 | ₹746 |
| 💰 Budget | 1040 (47%) | ₹35,029 | ₹99 |
| 🌟 VIP Outliers | 94 (4%) | ₹72,780 | ₹1,218 |

---

## 💡 Business Recommendations

- **💎 Premium Customers** → Exclusive wine & meat campaigns, catalog offers
- **🏠 Moderate Customers** → Family promotions, in-store discounts, upsell
- **💰 Budget Customers** → Deal campaigns, affordable recommendations
- **🌟 VIP Outliers** → Personal account managers, early product access

---

## 📁 Project Structure

Minor_Project_2_SmartCart/
├── SmartCart_Clustering.ipynb # Main notebook
├── data/
│ └── smartcart_customers.csv # Dataset
└── outputs/
└── plots/ # All visualizations


---

## 🔗 Links
- 📓 Kaggle Notebook: [SmartCart Clustering](https://kaggle.com/mdadnan96/smartcart-clustering)
- 💻 GitHub: [300-Days-of-AI-ML](https://github.com/Adnan9-63/300-Days-of-AI-ML)
- Linkedin : [Md Adnan](www.linkedin.com/in/md-adnan96)

---

## 👨‍💻 Author
**Md. Adnan** | 300 Days of AI-ML Challenge | Day 141
Student @ BCE Patna + IIT Patna BS-MS AI & Cybersecurity
