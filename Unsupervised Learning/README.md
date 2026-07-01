# 🧠 Unsupervised Machine Learning

Welcome to the **Unsupervised Machine Learning** module of my [300-Days-of-AI-ML](https://github.com/Adnan9-63/300-Days-of-AI-ML) journey! (Days 104 - 118)

In the Supervised Learning phase, algorithms were trained on data with definitive "answer keys" (labels). In this phase, the training wheels come off. This folder contains my code, notes, and experiments focused on finding hidden structures, patterns, and anomalies in strictly **unlabeled data**.

## 🚀 Topics Covered

### 1. Clustering
Grouping similar data points together without prior labels.
*   **K-Means Clustering:** Understanding centroids, the Random Initialization Trap, and hyperparameter tuning.
*   **Hierarchical Clustering:** Building and interpreting Dendrograms using Ward's linkage.
*   **DBSCAN:** Density-based clustering designed to handle complex, non-linear data shapes where K-Means fails.
*   **Evaluation Metrics:** Implementing the **Elbow Method** and calculating **Silhouette Scores** to mathematically determine the optimal number of clusters ($K$).

### 2. Dimensionality Reduction
*   **Principal Component Analysis (PCA):** Step-by-step mathematical reduction of high-dimensional datasets into 2D/3D spaces while preserving maximum variance for visualization and efficiency.

### 3. Anomaly Detection
Identifying rare items, events, or observations that raise suspicions by differing significantly from the majority of the data.
*   **Isolation Forests**
*   **Local Outlier Factor (LOF)**
*   **DBSCAN (for noise/outlier detection)**

---

## 💻 Practical Implementations & Notebooks

This directory includes several hands-on Jupyter Notebooks where I implemented these algorithms from scratch using `scikit-learn`:

*   [`KMeansForIris.ipynb`](./KMeansForIris.ipynb) & [`hierarchical_clustering.ipynb`](./hierarchical_clustering.ipynb): Applying K-Means and Agglomerative Clustering to the classic Iris dataset, including PCA for 2D visualization.
*   [`DBSCAN.ipynb`](./DBSCAN.ipynb): A direct comparison of K-Means vs. DBSCAN on non-linear data (`make_moons`), proving why density-based clustering is necessary for complex shapes.
*   [`isolation_forest.ipynb`](./isolation_forest.ipynb): Real-world anomaly detection! Using Isolation Forests and LOF on a medical **Thyroid Dataset** to successfully isolate high-risk outlier patients from the normal distribution.
*   **Automated K-Selection**: Scripts demonstrating how to programmatically find the elbow point using the `kneed` library.

---

## 🛠️ Tech Stack & Dependencies

To run the notebooks in this folder, you will need the following Python libraries:
*   `pandas` & `numpy` (Data manipulation)
*   `matplotlib` & `seaborn` (Data visualization)
*   `scikit-learn` (ML algorithms and preprocessing)
*   `kneed` (For automating the Elbow Method)

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed
