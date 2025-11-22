# Titanic Survival Analysis and Prediction - 2024

## Academic Context
This project was developed as part of the "Machine Learning" course (2024).

## Project Overview

This project analyzes the Titanic dataset to understand factors affecting passenger survival and builds predictive models for survival classification. The analysis includes comprehensive Exploratory Data Analysis (EDA), data preprocessing, and machine learning models including Logistic Regression, Naive Bayes, and Decision Trees.

## Dataset Summary

- **Source**: Titanic dataset
- **Total Records**: 887 passengers
- **Features**: 8 columns
- **Target Variable**: Survived (Binary: 0 = Not Survived, 1 = Survived)
- **Data Quality**: No missing values, no duplicates

### Dataset Features

1. **Survived**: Binary survival indicator (0 = Did not survive, 1 = Survived)
2. **Pclass**: Passenger class (1 = First, 2 = Second, 3 = Third)
3. **Name**: Full passenger name with titles
4. **Sex**: Gender (male/female)
5. **Age**: Age in years (continuous)
6. **Siblings/Spouses Aboard**: Number of siblings/spouses traveling with passenger
7. **Parents/Children Aboard**: Number of parents/children traveling with passenger
8. **Fare**: Ticket price paid

## Key Findings from EDA

### Survival Statistics
- Majority of passengers did not survive
- Survival rate strongly correlated with gender and passenger class
- Females had significantly higher survival rates than males

### Passenger Demographics
- Most passengers traveled in third class
- Median age: 28 years
- Age range (25th-75th percentile): 20.25 - 38 years
- Most passengers traveled alone or with few family members

### Fare Analysis
- Median fare: 14.45
- Most passengers paid relatively low fares
- Fare range (25th-75th percentile): 7.93 - 31.14

### Correlation Analysis

**Strong Predictors of Survival:**
- **Sex**: -0.54 correlation (females more likely to survive)
- **Pclass**: -0.34 correlation (higher class passengers more likely to survive)
- **Fare**: +0.26 correlation (higher fare passengers slightly more likely to survive)

**Weak Predictors:**
- Siblings/Spouses Aboard: -0.04
- Parents/Children Aboard: +0.08
- Age: -0.06

## Data Preprocessing

1. **Data Cleaning**: Verified no missing values or duplicates
2. **Encoding**: Applied LabelEncoder to 'Sex' column (Female = 0, Male = 1)
3. **Column Reordering**: Placed target variable 'Survived' at the end
4. **Feature Selection**: Identified Sex, Pclass, and Fare as key predictors

## Machine Learning Models

### 1. Logistic Regression

#### Single Predictor Model (Sex only)
- **Accuracy**: 0.74
- **Key Insight**: Sex alone is a strong predictor of survival

#### Multiple Predictor Model (Sex, Pclass, Fare)
- **Accuracy**: 0.74
- **Log Loss**: Calculated for model evaluation
- **Key Insight**: Adding Pclass and Fare does not significantly improve accuracy over Sex alone

#### Performance with Pclass or Fare Alone
- **Accuracy**: 0.66
- **Conclusion**: These features are weaker predictors when used individually

### 2. Naive Bayes Classifier

**Features Used**: Sex, Pclass

**Model Performance:**
- **Accuracy**: 0.79
- **AUC Score**: 0.79

**Confusion Matrix Results:**
- True Positives: 95 (correctly predicted survivors)
- True Negatives: 37 (correctly predicted non-survivors)
- False Positives: 16 (incorrectly predicted as survivors)
- False Negatives: 30 (missed survivors)

**Key Insight**: Naive Bayes achieved the highest accuracy among tested models with good discrimination ability (AUC = 0.79).

### 3. Decision Tree Classifier

**Features Used**: All features except Name

**Model Performance (After Regularization):**
- **Training Accuracy**: 0.86
- **Training F1 Score**: 0.82
- **Testing Accuracy**: 0.79
- **Testing F1 Score**: 0.70
- **Testing Precision**: 0.78
- **Testing Recall**: 0.63

**Hyperparameter Tuning:**
- Grid search applied for max_depth, criterion, min_samples_split, and min_samples_leaf
- Best parameters identified through cross-validation

**Feature Importance (F-scores):**
1. Sex: 313.22
2. Pclass: 83.92
3. Fare: 41.04

**Regularization Impact:**
- Reduced overfitting
- Slight decrease in training accuracy (from 0.87 to 0.86)
- Maintained testing accuracy at 0.79
- Improved generalization capability

## Model Comparison

| Model | Features | Accuracy | Key Strengths |
|-------|----------|----------|---------------|
| Logistic Regression (Single) | Sex | 0.74 | Simple, interpretable |
| Logistic Regression (Multiple) | Sex, Pclass, Fare | 0.74 | No improvement over single |
| Naive Bayes | Sex, Pclass | 0.79 | Best accuracy, good AUC |
| Decision Tree (Regularized) | All except Name | 0.79 | Visual interpretability, feature importance |

## Visualizations

The project includes various visualizations:
- Histograms for age and numerical feature distributions
- Scatter plots showing relationships between features and survival
- Bar plots for survival rates by gender
- Pie charts for survival proportion by passenger class
- Box plots for outlier detection
- Confusion matrices for model evaluation
- ROC curves for model performance assessment
- Decision tree visualizations

## Technologies Used

- **Python**: Primary programming language
- **Libraries**:
  - pandas: Data manipulation
  - numpy: Numerical computing
  - matplotlib, seaborn: Data visualization
  - scikit-learn: Machine learning models and evaluation
  
## Conclusions

1. **Gender Impact**: Sex is the strongest single predictor of survival, with females having significantly higher survival rates.

2. **Class Matters**: Passenger class (Pclass) is strongly correlated with survival, reflecting the "women and children first" protocol being more effectively applied in higher classes.

3. **Economic Factor**: Fare (proxy for wealth) shows moderate positive correlation with survival.

4. **Model Selection**: For this dataset, both Naive Bayes and regularized Decision Trees achieve 0.79 accuracy, outperforming logistic regression models.

5. **Feature Importance**: The analysis confirms historical accounts that gender and social class were primary factors determining survival during the Titanic disaster.

6. **Practical Application**: Simple models with key features (Sex, Pclass) perform as well as complex models, suggesting that survival prediction doesn't require extensive feature engineering for this dataset.

## Future Work

- Explore ensemble methods (Random Forest, Gradient Boosting)
- Feature engineering from Name column (titles extraction)
- Deep learning approaches
- Cross-validation for more robust evaluation
- Analysis of interaction effects between features

## Contributors

- Yasser Ashraf
- Ahmad Khlifa Abdulraouf (22010018)
- Hamza Omran (22011501)

## Project Structure

```
ML Project/
├── ML_Project_V2.ipynb    # Main Jupyter notebook with analysis
├── titanic.csv            # Dataset
└── README.md              # Project documentation
```

## How to Run

1. Ensure Python 3.x is installed
2. Install required libraries: `pip install numpy pandas matplotlib seaborn scikit-learn`
3. Open `ML_Project_V2.ipynb` in Jupyter Notebook or JupyterLab
4. Run cells sequentially to reproduce the analysis

## License

This project is for educational purposes.
