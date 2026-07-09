# 🤖 Machine Learning 
## Salary Predection
<p align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Flask-Web%20Framework-000000?style=for-the-badge&logo=flask&logoColor=white">
<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
<img src="https://img.shields.io/badge/HTML5-Frontend-E34F26?style=for-the-badge&logo=html5&logoColor=white">
<img src="https://img.shields.io/badge/CSS3-Styling-1572B6?style=for-the-badge&logo=css3&logoColor=white">
<img src="https://img.shields.io/badge/Machine%20Learning-Regression-16A34A?style=for-the-badge">
<img src="https://img.shields.io/badge/Data%20Science-AI-7B2CBF?style=for-the-badge">
<img src="https://img.shields.io/badge/Model%20Deployment-Production-0EA5E9?style=for-the-badge">

</p>

---

This project demonstrates an end-to-end **Machine Learning Engineering pipeline** that transforms a noisy real-world employee dataset into a production-ready salary prediction system.

Rather than focusing solely on model training, the project emphasizes the complete ML lifecycle including:

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Feature Encoding
- Pipeline Construction
- Model Training
- Hyperparameter Optimization
- Performance Evaluation
- Flask Deployment
- Production Prediction Interface

The final trained model is deployed through a Flask web application that allows HR personnel to generate salary predictions in real time without requiring any knowledge of machine learning.

---

# 🧠 Machine Learning Algorithms Implemented

The following regression algorithms were implemented and evaluated throughout the experimentation phase.

| Algorithm | Purpose |
|-----------|---------|
| Linear Regression | Baseline model |
| Decision Tree Regressor | Capture non-linear relationships |
| Random Forest Regressor | Ensemble learning |
| GridSearchCV | Hyperparameter Optimization |
| Pipeline | Automated preprocessing & prediction workflow |

---

# ⚙ Machine Learning Pipeline

The complete workflow follows the standard Machine Learning engineering process.

```text
Raw CSV Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Missing Value Handling
        │
        ▼
Datetime Parsing
        │
        ▼
Feature Engineering
        │
        ▼
Categorical Encoding
        │
        ▼
Train/Test Split
        │
        ▼
Pipeline Construction
        │
        ▼
Model Training
        │
        ▼
GridSearchCV
        │
        ▼
Best Model Selection
        │
        ▼
Model Evaluation
        │
        ▼
Flask Deployment
```

---

# 🔬 Pipeline Components

The project utilizes Scikit-Learn's Pipeline architecture to ensure consistent preprocessing during both training and prediction.

### Pipeline Stages

1. Data Cleaning
2. Feature Engineering
3. One-Hot Encoding
4. Feature Scaling (where required)
5. Model Training
6. Prediction

This guarantees that incoming production data follows the exact same transformation sequence used during model training.

---

# 🚀 Hyperparameter Optimization

Instead of relying on default model parameters, GridSearchCV was implemented to automatically search for the optimal hyperparameter combinations.

Benefits include:

- Better model generalization
- Reduced overfitting
- Automated parameter tuning
- Reliable cross-validation
- Improved prediction consistency

---

# 📈 Model Evaluation

Multiple regression models were compared using:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

The evaluation demonstrated that every model converged toward predicting the dataset mean salary.

---

# 💡 Engineering Insight

Initially, the poor model performance suggested that the algorithms required further tuning.

After implementing **GridSearchCV**, the optimized models still produced nearly identical performance.

This confirmed that the issue was **not caused by model selection or hyperparameters**, but by the statistical characteristics of the dataset itself.

Correlation analysis revealed that the available employee attributes had almost **no meaningful relationship with the target salary variable**. Because the features contained little predictive information, even highly optimized models could not learn a reliable mapping.

This experiment demonstrates an important machine learning principle:

> **A machine learning model cannot discover predictive patterns when the dataset itself does not contain meaningful relationships. Better algorithms cannot compensate for uninformative data.**

The primary limitation of this project was therefore **data quality and feature relevance**, rather than the learning algorithms.

---

# 🌐 Production Deployment

The trained model was deployed as a Flask web application developed with assistance from **GitHub Copilot** for rapid UI scaffolding and productivity improvements.

The deployment includes:

- Interactive HR dashboard
- Automatic feature validation
- Dynamic feature mapping
- Model inference
- Real-time salary prediction
- Production-ready prediction pipeline

---

## 📥 Step 1 — Clone the Repository

```bash
git clone https://github.com/AliDeveloper-dev/Employee_Salary_Prediction.git

cd Employee_Salary_Prediction
```

---

## 📦 Step 2 — Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 📚 Step 3 — Install Dependencies

```bash
pip install flask
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install joblib
```

Or install everything together:

```bash
pip install flask pandas numpy scikit-learn matplotlib joblib
```

---

## ▶️ Step 4 — Start the Flask Server

```bash
python app.py
```

---

## 🌍 Step 5 — Open the Application

Launch your browser and navigate to

```text
http://127.0.0.1:5000/
```

---

# 🚀 Using the Dashboard

Simply enter the employee information:

- Age
- Department
- Experience
- Performance Score
- Region
- Employment Status
- Remote Work
- Joining Date

Click **Predict Salary**, and the trained Machine Learning model will instantly estimate the employee's expected salary.

---
