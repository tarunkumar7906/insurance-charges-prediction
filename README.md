# Insurance Premium Prediction using Linear Regression

A machine learning project that predicts insurance charges based on customer profiles using Linear Regression.

## 📋 Project Overview

This project builds a predictive model to estimate insurance premiums based on customer demographics and health factors. Insurance companies need accurate pricing to remain competitive while maintaining profitability. This model provides a data-driven approach to premium calculation.

**Dataset Size:** 1,338 records | **Features:** 6 | **Target Variable:** Charges

## 🎯 Objective

Predict insurance charges (continuous variable) based on:
- Age
- BMI (Body Mass Index)
- Number of children
- Smoking status
- Sex
- Region

## 📊 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Age of the customer (18-64 years) |
| bmi | Numeric | Body Mass Index (11.6-53.1) |
| children | Numeric | Number of dependent children (0-5) |
| smoker | Categorical | Smoking status (yes/no) |
| sex | Categorical | Gender (male/female) |
| region | Categorical | Geographic region (northeast, northwest, southeast, southwest) |
| **charges** | **Numeric** | **Annual insurance charges (Target Variable)** |

## 🔍 Exploratory Data Analysis (EDA)

### Key Findings:

- **No missing values** in the dataset
- **Distribution Insights:**
  - Age: Fairly uniform distribution (18-64 years)
  - BMI: Slightly right-skewed (mean: 30.7)
  - Charges: Highly right-skewed with cluster around $10K-$15K
  
- **Categorical Patterns:**
  - Smoking status is the **strongest predictor** of charges
  - Smokers are charged 3x higher on average than non-smokers
  - Regional differences exist but are less significant than smoking status
  
- **Outliers:** Some high-charge outliers present (likely from smokers with high BMI)

## 🛠️ Data Preprocessing

1. **Encoding Categorical Variables**
   - One-hot encoded `sex`, `smoker`, and `region`
   - Converted boolean values to numerical (0/1)

2. **Feature Scaling**
   - Applied StandardScaler to numerical features (age, bmi, children)
   - Scaling ensures all features have equal weight in the model

3. **Train-Test Split**
   - 80% training set (1,070 records)
   - 20% testing set (268 records)
   - Random state: 42 (for reproducibility)

## 🤖 Model: Linear Regression

### Algorithm Description:

Linear Regression finds the best-fit line (hyperplane) through the data by minimizing prediction errors using the **Mean Squared Error (MSE)** loss function.

**Equation:** `y = mx + b`
- **y:** Predicted charges
- **m:** Slope (feature coefficients)
- **x:** Input features
- **b:** Intercept

### Training Process:

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)  # 0.75
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))  # 0.74
```

## 📈 Model Performance

| Metric | Score | Interpretation |
|--------|-------|-----------------|
| **R² Score** | 0.75 | Model explains 75% of variance in charges |
| **Adjusted R²** | 0.74 | Accounts for number of features (more reliable) |
| **RMSE** | ~$4,500 | Average prediction error ±$4,500 |

### What This Means:

✅ **Strengths:**
- 75% accuracy is reasonable for a first linear model
- Strong linear relationships found in data
- No overfitting (train-test scores similar)

⚠️ **Limitations:**
- 25% of variance remains unexplained
- Linear assumptions may not capture complex charge patterns
- Smoker variable creates non-linear relationships
- Some high-charge outliers not well predicted

## 💡 Key Learnings

1. **EDA is Critical:** Visualizing data revealed smoking status as the dominant predictor, guiding model interpretation

2. **Feature Scaling Matters:** Standardized features ensure fair weight distribution across the model

3. **R² Score Tells the Truth:** Even seemingly high R² scores (0.75) indicate room for improvement

4. **Linear Regression Limitations:** The model assumes linear relationships; charge patterns show non-linearity (especially for smokers)

5. **Train-Test Split Prevents Cheating:** Never evaluate on training data; separate test set reveals true generalization ability

## 🚀 Next Steps for Improvement

### Option 1: Feature Engineering
- Create interaction terms (age × smoker)
- Add polynomial features (age², bmi²)
- Create smoking × BMI interaction to capture non-linear smoker pricing

### Option 2: Non-Linear Models
- **Decision Trees:** Capture non-linear charge patterns
- **Random Forest:** Ensemble approach for better accuracy
- **Polynomial Regression:** Fit higher-degree curves

### Option 3: Model Evaluation
- Implement cross-validation for robust performance estimates
- Analyze residuals to identify systematic prediction errors
- Try gradient boosting models (XGBoost, LightGBM)

**Expected Improvement:** Target R² > 0.85 with non-linear models

## 📁 Files in Repository

```
insurance-premium-prediction/
├── insurance.ipynb          # Complete Jupyter notebook with code & analysis
├── insurance.csv            # Raw dataset (1,338 records)
├── README.md               # This file
```

## 🔧 Installation & Usage

### Prerequisites:
- Python 3.7+
- Jupyter Notebook

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Run the Project:
```bash
jupyter notebook insurance.ipynb
```

### Requirements File:
```
pandas==1.3.5
numpy==1.21.5
scikit-learn==1.0.2
matplotlib==3.5.1
seaborn==0.11.2
```

## 📊 Visualizations Included

- **Histograms + KDE Curves:** Distribution of numerical features
- **Count Plots:** Categorical variable frequencies
- **Box Plots:** Outlier identification and spread analysis
- **Scatter Plots:** Feature vs charges relationships
- **Model Predictions:** Actual vs predicted scatter plot

## 🎓 Learning Outcomes

By completing this project, I gained hands-on experience in:

✅ Loading and exploring real-world datasets
✅ Data cleaning and preprocessing
✅ Feature encoding and scaling
✅ Training and evaluating ML models
✅ Interpreting regression metrics (R², Adjusted R²)
✅ Identifying model limitations and improvement opportunities
✅ Using scikit-learn for ML pipelines
✅ Data visualization with Seaborn and Matplotlib

## 📝 License

This project is open-source and available for educational purposes.

## 🤝 Contact & Social

**Author:** Tarun Kumar  

---

**Last Updated:** July 2026  
**Status:** ✅ Completed | Insurance Premium Prediction v1.0
