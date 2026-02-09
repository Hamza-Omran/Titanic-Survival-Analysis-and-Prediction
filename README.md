# Titanic Survival Analysis & Prediction (Machine Learning)

An **end-to-end machine learning analysis** of the Titanic dataset, combining **exploratory data analysis (EDA)**, **feature preprocessing**, and **classification models** to understand and predict passenger survival.

**Context:** Machine Learning – University Coursework (2024)  
**Domain:** Supervised Learning · Classification · Data Analysis  
**Type:** Academic ML Project

---

## Project Overview

This project investigates the factors influencing passenger survival during the Titanic disaster and builds **predictive classification models** to estimate survival outcomes.

The work emphasizes:
- Data understanding through EDA
- Feature relevance and correlation analysis
- Model comparison using interpretable metrics
- Generalization and overfitting control

Rather than maximizing accuracy alone, the project focuses on **insight, interpretability, and model behavior**.

---

## Dataset Summary

- **Source:** Titanic dataset  
- **Records:** 887 passengers  
- **Target Variable:** `Survived` (0 = Did not survive, 1 = Survived)  
- **Data Quality:** No missing values, no duplicates  

### Features

- `Pclass` – Passenger class (1st, 2nd, 3rd)
- `Name` – Passenger name (textual)
- `Sex` – Gender
- `Age` – Passenger age
- `Siblings/Spouses Aboard`
- `Parents/Children Aboard`
- `Fare` – Ticket price

---

## Exploratory Data Analysis (EDA)

### Key Observations

#### Survival Patterns
- Majority of passengers did not survive
- **Gender is the strongest survival indicator**
- Higher passenger classes show higher survival rates

#### Demographics
- Most passengers traveled in **third class**
- Median age ≈ 28 years
- Most passengers traveled alone or with small families

#### Fare Analysis
- Median fare ≈ 14.45
- Higher fares correlate positively with survival
- Fare acts as a proxy for socio-economic status

---

## Correlation Analysis

### Strong Predictors
- **Sex:** −0.54 (females significantly more likely to survive)
- **Pclass:** −0.34 (higher class → higher survival)
- **Fare:** +0.26

### Weak Predictors
- Age
- Family size indicators

This confirms that **social factors dominated survival outcomes**.

---

## Data Preprocessing

- Verified dataset integrity (no nulls / duplicates)
- Encoded categorical variables:
  - `Sex`: Female → 0, Male → 1
- Reordered features for modeling clarity
- Selected key predictors based on correlation and domain logic

---

## Machine Learning Models

### 1. Logistic Regression

**Purpose:** Baseline, interpretable classifier

- **Sex only:**  
  - Accuracy: **0.74**
- **Sex + Pclass + Fare:**  
  - Accuracy: **0.74**
- **Insight:**  
  Adding more features did not improve performance beyond gender alone.

---

### 2. Naive Bayes Classifier

**Features:** Sex, Pclass

- **Accuracy:** **0.79**
- **AUC:** **0.79**

**Confusion Matrix Highlights**
- Strong discrimination between survivors and non-survivors
- Balanced precision and recall

**Insight:**  
Naive Bayes achieved the **best overall performance** with minimal feature complexity.

---

### 3. Decision Tree (Regularized)

**Features:** All except Name

**After hyperparameter tuning:**
- Training Accuracy: 0.86
- Testing Accuracy: **0.79**
- Testing F1 Score: 0.70

**Feature Importance**
1. Sex
2. Pclass
3. Fare

**Insight:**  
Regularization reduced overfitting while preserving generalization.

---

## Model Comparison

| Model | Features | Accuracy | Key Strength |
|-----|--------|---------|--------------|
| Logistic Regression | Sex | 0.74 | Interpretability |
| Logistic Regression | Sex, Pclass, Fare | 0.74 | Simplicity |
| Naive Bayes | Sex, Pclass | **0.79** | Best overall |
| Decision Tree | All | **0.79** | Feature importance |

---

## Visualizations

The project includes:
- Feature distribution histograms
- Survival rate bar plots
- Correlation heatmaps
- Boxplots for numerical features
- Confusion matrices
- ROC curves
- Decision tree visualization

These visuals support **transparent model evaluation**.

---

## Technology Stack

- **Python**
- **pandas, numpy** – Data processing
- **matplotlib, seaborn** – Visualization
- **scikit-learn** – ML models & evaluation

---

## Engineering Focus

This project emphasizes:
- Data-driven insight over blind optimization
- Interpretability of classification models
- Proper evaluation beyond accuracy
- Bias awareness in historical datasets
- Model simplicity vs. performance trade-offs

It is designed as a **foundational ML classification project**, not a leaderboard-driven benchmark.

---

## Conclusions

- **Gender** is the dominant survival factor
- **Passenger class** reflects social and physical access to safety
- Simple models perform competitively on structured datasets
- Naive Bayes and regularized trees generalize best

The analysis confirms historical accounts using **quantitative evidence**.

---

## Future Work

- Feature extraction from names (titles)
- Ensemble models (Random Forest, Gradient Boosting)
- Cross-validation for robustness
- Interaction effects between features
- Ethical analysis of bias in historical data

---

## Project Structure

```

ML Project/
├── ML_Project_V2.ipynb
├── titanic.csv
└── README.md

````

---

## How to Run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook ML_Project_V2.ipynb
````
