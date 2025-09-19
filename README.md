# Master's Thesis: Anomalous Behavior Classification in Solar Inverters

This repository contains the code implemented for the development of my Master's Thesis, focused on building a classification model to detect anomalous behavior in solar inverters.  

The project is structured in three main phases:  
1. **Exploratory Data Analysis (EDA)** – understanding the data format and the problem.  
2. **Clustering Models** – detecting patterns in the data.  
3. **Classification Models** – predicting whether an inverter will present condensation failure or not.  

---

## Exploratory Data Analysis (EDA)

- **EDA_Lucia_2.ipynb**: Uses NumPy and Pandas to load and preprocess CSV data. Includes variable type conversion, distribution/boxplot visualizations, and correlation analysis.  
- **Histogramas_Alarmas.ipynb**: Studies alarm frequency to identify the most relevant alarms. Histograms of alarm frequency were created, along with the top 3 alarms per inverter and per day.  

---

## Clustering Models

Due to the high dimensionality of the data, **Principal Component Analysis (PCA)** was applied before testing three clustering methods: **K-Means**, **Fuzzy C-Means**, and **Gaussian Mixture Models (GMM)**.

- **Analisis_PCA.ipynb**: PCA with different numbers of components, selecting 3 as optimal. The top 10 contributing variables per component were analyzed for interpretability, and their correlations were studied for anomalous vs. normal inverters.  
- **Kmeans.ipynb**: K-Means clustering with 3 PCA components. Tested different values of *k* (2–10), using Silhouette score and SSE to evaluate performance. Final model trained with *k=2*, analyzing cluster centers and testing with normal vs. faulty inverters.  
- **CMeans.ipynb**: Fuzzy C-Means with PCA-reduced data. Model trained with 2 clusters using cross-validation. Evaluated robustness with **Fuzzy Partition Coefficient (FPC)** and **Dunn Index**. Cluster centers, membership degrees, and visualizations were analyzed.  
- **GMM.ipynb**: Gaussian Mixture Models with PCA-reduced data. Tested different covariance types and evaluated with BIC. Trained final 2-cluster model, studied centroids, log-probabilities, and tested with normal vs. faulty inverters.  

---

## Classification Models

For this stage, more data was available, requiring a new EDA. Two tree-based models were tested: **Decision Tree** and **XGBoost**.

- **EDA_Total.ipynb**: Extended EDA with larger dataset, applying the same techniques as in *EDA_Lucia_2.ipynb*.  
- **DecisionTree.ipynb**: Trained two models: one with 4 variables, and one with 2 PCA components. Used **GridSearchCV** for hyperparameter tuning, studied feature importance, and evaluated with recall, balanced accuracy, ROC curves, and probability distributions.  
- **XGBoost4_Ejecutado.ipynb**: XGBoost trained on 4 variables (same as Decision Tree). Included GridSearchCV, feature importance, tree depth visualization, and evaluation with train/test metrics, ROC curves, and probability distributions.  
- **XGBoostPCA_Ejecutado.ipynb**: Similar to the previous notebook but trained on the 2 PCA components.  

---

## Considerations and Future Work

This project represents the first step toward deploying a production-ready model for detecting condensation failure in solar inverters.  

Limitations:  
- Work was carried out on a personal computer, preventing the use of more advanced models such as deep neural networks.  
- Results show strong potential but need scaling with more data and computational resources.  

Future directions:  
- Extend the methodology to other types of inverter failures.  
- Apply more advanced machine learning techniques, including neural networks.  
- Contribute to the transition toward predictive maintenance for solar inverters.  

---

## Requirements

- Python 3.8+  
- Jupyter Notebook  
- Main libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `xgboost`  

---

## How to Run

1. Clone this repository:  
   ```bash
   git clone https://github.com/lucialopara/Masters-Thesis.git
   cd Masters-Thesis
