🎬 Netflix Customer Churn Prediction & Analytics Dashboard

An End-to-End Machine Learning and Business Intelligence Solution for
Customer Churn Prediction, Risk Analysis, and Retention Strategy
Optimization

------------------------------------------------------------------------

📌 Table of Contents

1.  Project Overview
2.  Business Problem
3.  Project Architecture
4.  Data Pipeline Flow
5.  Machine Learning Workflow
6.  Model Performance
7.  Feature Engineering
8.  Database Architecture
9.  Power BI Dashboard
10. Streamlit Application
11. Technologies Used
12. Installation & Setup
13. Future Improvements
14. Disclaimer
15. Author

------------------------------------------------------------------------

📌 Project Overview

Customer churn is one of the biggest challenges for subscription-based
businesses.

This project develops an end-to-end analytics and machine learning
solution to:

-   Identify customers likely to churn
-   Predict churn probability
-   Classify customers into risk categories
-   Understand customer engagement behavior
-   Generate actionable retention insights

The solution combines:

-   Machine Learning
-   Data Analytics
-   SQL Database Engineering
-   Streamlit Application Development
-   Power BI Dashboard Visualization

------------------------------------------------------------------------

🎯 Business Problem

Subscription businesses need to identify customers who are at risk of
leaving before churn occurs.

This project answers:

Who is likely to churn?

Identify customers with high churn probability.

Why are customers leaving?

Analyze:

-   Engagement behavior
-   Login activity
-   Payment issues
-   Subscription patterns

Which customers require attention?

Customers are categorized into:

    HIGH RISK
        |
        ↓
    MEDIUM RISK
        |
        ↓
    LOW RISK

How can retention improve?

Use predictive insights to design:

-   Targeted offers
-   Customer engagement campaigns
-   Retention strategies

------------------------------------------------------------------------

🏗️ Project Architecture

                    CUSTOMER DATA SOURCES

            ┌─────────────────────────────┐
            │ Customer Information        │
            │ Subscription Data           │
            │ User Engagement Data        │
            │ Historical Churn Labels     │
            └──────────────┬──────────────┘
                           |
                           ↓

            ┌─────────────────────────────┐
            │ Data Cleaning & Processing  │
            │ Missing Value Handling     │
            │ Data Transformation        │
            └──────────────┬──────────────┘
                           |
                           ↓

            ┌─────────────────────────────┐
            │ Feature Engineering         │
            │ Engagement Score            │
            │ Payment Risk Score          │
            │ Inactivity Score            │
            │ Customer Health Score       │
            └──────────────┬──────────────┘
                           |
                           ↓

            ┌─────────────────────────────┐
            │ Machine Learning Model      │
            │ Random Forest Classifier    │
            └──────────────┬──────────────┘
                           |
                           ↓

            ┌─────────────────────────────┐
            │ Churn Probability Prediction│
            │ Risk Classification         │
            └──────────────┬──────────────┘
                           |
              ┌────────────┴────────────┐
              ↓                         ↓

     ┌─────────────────┐       ┌─────────────────┐
     │ MySQL Database  │       │ Streamlit App   │
     │ Predictions     │       │ Real-time ML    │
     └────────┬────────┘       └────────┬────────┘
              |
              ↓

     ┌─────────────────────────┐
     │ Power BI Analytics      │
     │ Executive Dashboard     │
     │ ML Monitoring           │
     └─────────────────────────┘

------------------------------------------------------------------------

🧠 Machine Learning Workflow

    Raw Customer Data

            |
            ↓

    Exploratory Data Analysis

            |
            ↓

    Data Preprocessing

            |
            ↓

    Feature Engineering

            |
            ↓

    Model Training

            |
            ↓

    Model Evaluation

            |
            ↓

    Churn Prediction

            |
            ↓

    Prediction Storage

            |
            ↓

    Dashboard Analytics

------------------------------------------------------------------------

⚙️ Feature Engineering

Created ML features:

  Feature                 Purpose
  ----------------------- -------------------------------------
  Engagement Score        Measures customer activity
  Payment Risk Score      Identifies payment issues
  Inactivity Score        Measures customer inactivity
  Customer Health Score   Overall customer relationship score

------------------------------------------------------------------------

🤖 Machine Learning Model

Baseline Model

Logistic Regression

Final Model

Random Forest Classifier

Random Forest was selected because:

-   Handles nonlinear customer behavior
-   Provides feature importance
-   Works well with structured data
-   Reduces overfitting through ensemble learning

------------------------------------------------------------------------

📊 Model Output

For every customer, the model generates:

    Customer ID
          |
          ↓
    Churn Probability
          |
          ↓
    Prediction
          |
          ↓
    Risk Category
          |
          ↓
    Retention Action

Example:

    Customer ID:
    CUST10006

    Churn Probability:
    98%

    Prediction:
    Churn

    Risk Category:
    HIGH RISK

------------------------------------------------------------------------

🗄️ Database Architecture

                    MySQL Database

                           |
            --------------------------------

            customers

            |
            ↓

            subscriptions

            |
            ↓

            user_engagement

            |
            ↓

            churn_labels

            |
            ↓

            churn_predictions

Database stores:

customers

-   Customer details
-   Demographics
-   Signup information

subscriptions

-   Subscription plan
-   Monthly fee
-   Payment method
-   Payment failures

user_engagement

-   Watch hours
-   Movies watched
-   Series watched
-   Login frequency

churn_predictions

-   Churn probability
-   Prediction
-   Risk category
-   Model version

------------------------------------------------------------------------

📈 Power BI Dashboard

Dashboard Pages:

    Introduction Page

            ↓

    Summary Overview

            ↓

    Executive Churn Overview

            ↓

    ML Risk Dashboard

            ↓

    Customer Behaviour Analysis

            ↓

    Model Monitoring

            ↓

    Disclaimer

Dashboard provides:

-   Executive KPIs
-   Customer churn trends
-   Risk segmentation
-   ML prediction analysis
-   Behavioral insights

------------------------------------------------------------------------

🖥️ Streamlit Application

Workflow:

    Customer Input

          ↓

    Feature Processing

          ↓

    Random Forest Model

          ↓

    Prediction Probability

          ↓

    Risk Classification

          ↓

    Save Result into MySQL

Application Features:

-   Customer input interface
-   Real-time churn prediction
-   Probability calculation
-   Risk classification
-   Database integration

------------------------------------------------------------------------

🛠️ Technologies Used

Programming:

-   Python

Machine Learning:

-   Scikit-learn
-   Random Forest
-   Logistic Regression

Data Processing:

-   Pandas
-   NumPy

Database:

-   MySQL
-   SQLAlchemy
-   PyMySQL

Visualization:

-   Power BI
-   Plotly

Application:

-   Streamlit

------------------------------------------------------------------------

⚙️ Installation & Setup

Clone Repository:

    git clone https://github.com/yourusername/Netflix-Churn-Prediction.git

Create Environment:

    python -m venv venv

Install Dependencies:

    pip install -r requirements.txt

Environment Configuration:

    DB_HOST=localhost
    DB_USER=root
    DB_PASSWORD=your_password
    DB_NAME=netflix_churn
    DB_PORT=3306

Run Application:

    streamlit run app/app.py

------------------------------------------------------------------------

🚀 Future Improvements

-   Deploy ML model using FastAPI
-   Create REST prediction API
-   Cloud deployment
-   Automated model retraining pipeline
-   SHAP explainability
-   ML monitoring system
-   MLOps implementation

------------------------------------------------------------------------

⚠️ Disclaimer

This project is developed for educational and portfolio purposes.

Machine learning predictions represent probability-based estimates and
should support, not replace, business decisions.

------------------------------------------------------------------------

👨‍💻 Author

Atif Mahmood

Data Scientist | Data Analytics | Machine Learning
