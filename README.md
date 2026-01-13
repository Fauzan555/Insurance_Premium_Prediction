# Insurance_Premium_Prediction

Insurance Premium Prediction Using Machine Learning

This project focuses on predicting medical insurance expenses (insurance premium/charges) for individuals using machine learning regression models.
The goal is to analyze how factors such as age, BMI, smoking habits, number of children, and region affect insurance costs and to build a reliable predictive model.

# Dataset Description

The dataset contains customer-level insurance information with the following features:

Feature	Description
age	Age of the policyholder
sex	Gender (male/female)
bmi	Body Mass Index
children	Number of dependents
smoker	Smoking status (yes/no)
region	Residential region
expenses	Medical insurance charges (target variable)

# Exploratory Data Analysis (EDA)

The following analyses were performed:

# Univariate Analysis

Distribution of age, BMI, children, and expenses

Expenses were right-skewed

# Bivariate Analysis

Age vs Expenses → Expenses increase with age

BMI vs Expenses → Strong effect for smokers with BMI > 30

Smoker vs Expenses → Smokers incur significantly higher expenses

# Correlation Analysis

Age and smoking status show strong correlation with expenses

BMI has a weaker correlation

Emphasized correlation ≠ causation

# Data Preprocessing

The following preprocessing steps were applied:

🔹 Target Transformation

Applied log transformation on expenses to reduce skewness

🔹 Feature Engineering

One-hot encoding for categorical variables (sex, smoker, region)

StandardScaler for numerical features

🔹 Pipeline

Used sklearn.pipeline.Pipeline and ColumnTransformer for clean and reproducible preprocessing.

# Machine Learning Models Used:

Multiple regression models were trained and evaluated:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

K-Nearest Neighbors (KNN)

XGBoost Regressor

# Model Evaluation Metrics:

Models were evaluated using:

RMSE (Root Mean Squared Error)

R² Score

# Model Performance Comparison
Model	RMSE	R² Score
Gradient Boosting	4652.33	0.8527
Random Forest	4706.14	0.8493
Decision Tree	4867.31	0.8388
XGBoost	5763.70	0.7739
KNN	5779.96	0.7727
Linear Regression	5796.56	0.7836

# Final Model Selection

Gradient Boosting Regressor was selected as the final model because:

Lowest RMSE

Highest R² score

Best generalization performance

The trained model (including preprocessing pipeline) was saved using pickle.

# Model Deployment

The final model is deployed using Streamlit, allowing users to:

Input personal details (age, BMI, smoker status, etc.)

Get real-time insurance premium predictions
Model Deployment

The final model is deployed using Streamlit, allowing users to:

Input personal details (age, BMI, smoker status, etc.)

Get real-time insurance premium predictions

# Project Structure
Insurance_Premium_Prediction/
│
├── app.py                          # Streamlit app
├── gradient_boosting_regressor_model.pkl
├── insurance.csv                   # Raw dataset
├── clean_data.csv                  # Cleaned dataset
├── Insurance_Premium.ipynb         # Jupyter notebook
├── requirements.txt
└── README.md

# How to Run Locally

pip install -r requirements.txt
streamlit run app.py

# Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

XGBoost

Streamlit

# Key Learnings

Importance of preprocessing and feature scaling

Handling skewed target variables using log transformation

Model comparison and hyperparameter tuning

End-to-end ML pipeline creation

Deploying ML models using Streamlit

# Final Outcome

A production-ready insurance premium prediction system with:

High predictive accuracy

Clean pipeline-based architecture

User-friendly Streamlit interface

