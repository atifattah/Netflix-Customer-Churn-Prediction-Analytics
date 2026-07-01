Netflix Customer Churn Prediction & Analytics Dashboard

An end-to-end Machine Learning and Business Intelligence solution to
predict customer churn, identify high-risk customers, and generate
actionable retention insights using customer behavior, subscription, and
engagement data.

Table of Contents

-   Project Overview
-   Business Problem
-   Project Architecture
-   Machine Learning Workflow
-   Model Performance
-   Feature Importance
-   Database Design
-   Power BI Dashboard
-   Streamlit Application
-   Technologies Used
-   Installation & Setup
-   Future Improvements
-   Disclaimer

Project Overview

Customer churn is one of the biggest challenges for subscription-based
businesses.

This project develops a predictive analytics solution to:

-   Identify customers likely to churn
-   Estimate churn probability
-   Segment customers into risk categories
-   Analyze customer behavior patterns
-   Generate retention insights

Business Problem

The objective is to understand:

-   Who is likely to churn?
-   Why customers are leaving?
-   Which customers require immediate attention?
-   How retention strategies can be improved?

Customers are segmented into:

-   High Risk
-   Medium Risk
-   Low Risk

Project Architecture

Customer Dataset | ↓ Data Cleaning & Transformation | ↓ Feature
Engineering | ↓ Random Forest Churn Model | ↓ Churn Probability
Prediction | ↓ MySQL Database Storage | ↓ Power BI Dashboard & Streamlit
Application

Machine Learning Workflow

Performed:

-   Data cleaning
-   Missing value handling
-   Feature transformation
-   Categorical encoding
-   Feature selection

Engineered features:

-   Engagement Score
-   Payment Risk Score
-   Inactivity Score
-   Customer Health Score

Machine Learning Model

Models Used:

Baseline: - Logistic Regression

Final: - Random Forest Classifier

Random Forest was selected because it:

-   Handles nonlinear patterns
-   Provides feature importance
-   Works well with mixed data
-   Reduces overfitting using ensemble learning

Model Output

The model generates:

-   Churn Probability
-   Prediction (Churn / Stay)
-   Risk Category
-   Model Version

Example:

Customer ID: CUST10006

Churn Probability: 98%

Prediction: Churn

Risk Category: HIGH RISK

Database Design

MySQL tables:

customers: - Customer information

subscriptions: - Subscription details - Payment information

user_engagement: - Viewing and engagement behavior

churn_labels: - Historical churn status

churn_predictions: - ML prediction results

Power BI Dashboard

Dashboard pages:

1.  Introduction Page
2.  Summary Overview
3.  Executive Churn Overview
4.  ML Risk Dashboard
5.  Customer Behaviour Analysis
6.  Model Monitoring

Streamlit Application

Features:

-   Customer input interface
-   Real-time churn prediction
-   Probability score
-   Risk classification
-   Database storage

Workflow:

User Input ↓ Feature Processing ↓ Random Forest Model ↓ Prediction
Result ↓ Save Prediction in MySQL

Technologies Used

Programming: - Python

Machine Learning: - Scikit-learn - Random Forest - Logistic Regression

Data Processing: - Pandas - NumPy

Database: - MySQL - SQLAlchemy - PyMySQL

Visualization: - Power BI - Plotly

Application: - Streamlit

Installation & Setup

Clone repository:

git clone https://github.com/yourusername/Netflix-Churn-Prediction.git

Create environment:

python -m venv venv

Activate:

venv

Install dependencies:

pip install -r requirements.txt

Environment Configuration

Create .env:

DB_HOST=localhost DB_USER=root DB_PASSWORD=your_password
DB_NAME=netflix_churn DB_PORT=3306

Run Application

streamlit run app/app.py

Future Improvements

-   Deploy ML model using FastAPI
-   Create REST API
-   Deploy dashboard on cloud
-   Add automated model retraining
-   Add SHAP explainability
-   Implement MLOps workflow

Disclaimer

This project is developed for educational and portfolio purposes.

Machine learning predictions represent probability-based estimates and
should support, not replace, business decisions.

Author

Atif Mahmood

Data Scientist | Data Analytics | Machine Learning
