# 🎬 Netflix Customer Churn Prediction & Analytics Dashboard

An end-to-end Machine Learning and Business Intelligence solution to predict customer churn, identify high-risk customers, and generate actionable retention insights using customer behavior, subscription, and engagement data.

The project combines:

- Machine Learning
- Data Analytics
- SQL Database Engineering
- Streamlit Application Development
- Power BI Dashboard Visualization


---

# 📌 Project Overview

Customer churn is one of the biggest challenges for subscription-based businesses.

The objective of this project is to develop a predictive analytics solution that can:

- Identify customers who are likely to churn
- Estimate churn probability
- Segment customers into risk categories
- Understand customer behavior patterns
- Provide actionable insights for retention strategies


The solution follows an end-to-end pipeline:

Customer Data
|
↓
Data Cleaning & Transformation
|
↓
Feature Engineering
|
↓
Machine Learning Model
|
↓
Churn Probability Prediction
|
↓
MySQL Database Storage
|
↓
Interactive Dashboard Analytics

---

# 🎯 Business Problem

Customer acquisition is expensive, and retaining existing customers is critical for long-term growth.

This project helps answer:

### Who is likely to churn?

Identify customers with high churn probability.

### Why are customers leaving?

Analyze behavioral and subscription patterns.

### Which customers require immediate attention?

Segment customers into:

- High Risk
- Medium Risk
- Low Risk

### How can retention improve?

Enable targeted retention campaigns using data-driven insights.

---

# 🏗️ Project Architecture

                Customer Dataset
                       |
                       |
                Data Processing
                       |
                       |
              Feature Engineering
                       |
                       |
          Random Forest Churn Model
                       |
                       |
          Churn Probability Output
                       |
         ---------------------------
         |                         |
         |                         |
    MySQL Database          Streamlit App
         |                         |
         |                         |
   Power BI Dashboard     Real-Time Prediction

---

# 🧠 Machine Learning Workflow

## 1. Data Preparation

Performed:

- Data cleaning
- Missing value handling
- Feature transformation
- Categorical encoding
- Feature selection


## 2. Feature Engineering

Created customer behavior features:

- Engagement Score
- Payment Risk Score
- Inactivity Score
- Customer Health Score


These features help capture customer engagement and churn indicators.


---

# 🤖 Machine Learning Model


## Models Used

### Baseline Model

Logistic Regression


### Final Model

Random Forest Classifier


Random Forest was selected because it:

- Handles nonlinear customer behavior patterns
- Provides feature importance
- Performs well with mixed feature types
- Reduces overfitting through ensemble learning


---

# 📊 Model Output

For every customer, the model generates:


| Output | Description |
|-|-|
| Churn Probability | Probability of customer churn |
| Prediction | Churn / Stay classification |
| Risk Category | High / Medium / Low risk |
| Model Version | Tracking model iteration |


Example:

Customer ID:
CUST10006

Churn Probability:
98%

Prediction:
Churn

Risk Category:
HIGH RISK

---

# 🗄️ Database Design


MySQL database contains:


## Customer Information

Table:

customers

Contains:

- Customer ID
- Age
- Gender
- Country
- Signup Date


---

## Subscription Information

Table:


subscriptions


Contains:

- Subscription Plan
- Monthly Fee
- Payment Method
- Payment Failures
- Auto Renewal


---

## Customer Engagement

Table:


user_engagement


Contains:

- Watch Hours
- Movies Watched
- Series Watched
- Login Frequency
- Content Engagement


---

## Churn Labels

Table:


churn_labels


Contains:

- Customer ID
- Churn Status


---

## ML Predictions

Table:


churn_predictions


Stores:

- Churn probability
- Prediction result
- Risk category
- Feature scores
- Model version


---

# 📈 Power BI Dashboard


The dashboard contains:


## 1. Introduction Page

Project overview and analytical objectives.


---

## 2. Summary Overview

Provides executive-level insights:

- Total Customers
- Churn Rate
- Revenue Overview
- Subscription Analysis
- Customer Distribution


---

## 3. Executive Churn Overview

Focuses on:

- Churn segmentation
- Customer risk levels
- Revenue impact
- Subscription churn trends


---

## 4. ML Risk Dashboard

Machine learning prediction analysis:

- Churn probability distribution
- Risk categories
- Prediction outcomes
- Customer risk drivers


---

## 5. Customer Behaviour Analysis

Analyzes:

- Watch activity
- Login behavior
- Engagement patterns
- Content consumption trends


---

## 6. Model Monitoring

Tracks:

- Prediction distribution
- Model confidence
- Customer health score
- Risk monitoring


---

# 🖥️ Streamlit Application


The application provides:

- Customer input interface
- Real-time churn prediction
- Probability score
- Risk classification
- Database storage


Example workflow:


User Input
|
↓
Feature Processing
|
↓
Random Forest Model
|
↓
Prediction Result
|
↓
Save Prediction in MySQL


---

# 🛠️ Technologies Used


## Programming

- Python


## Machine Learning

- Scikit-learn
- Random Forest Classifier
- Logistic Regression


## Data Processing

- Pandas
- NumPy


## Database

- MySQL
- SQLAlchemy
- PyMySQL


## Visualization

- Power BI
- Plotly


## Application

- Streamlit


---

# 📂 Repository Structure

Netflix-Churn-Prediction

│
├── app/
│ └── app.py
│
├── models/
│ ├── random_forest_model.pkl
│ └── random_forest_features.pkl
│
├── src/
│ ├── data_preprocessing.py
│ ├── database_connection.py
│ ├── database_insert.py
│ └── train_model.py
│
├── database/
│ ├── create_tables.sql
│ └── create_views.sql
│
├── notebooks/
│ ├── EDA.ipynb
│ └── Model_Training.ipynb
│
├── powerbi/
│ └── Netflix_Churn_Dashboard.pbix
│
├── screenshots/
│
├── requirements.txt
│
└── README.md


---

# ⚙️ Installation & Setup


## Clone Repository

```bash
git clone https://github.com/yourusername/Netflix-Churn-Prediction.git

Create Virtual Environment
python -m venv venv

## Activate:

Windows:

venv\Scripts\activate

## Install Dependencies
pip install -r requirements.txt

🔐 Environment Configuration

## Create a .env file:

DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=netflix_churn
DB_PORT=3306

▶️ Run Streamlit Application
streamlit run app/app.py

## Application will open:

http://localhost:8501

📌 Future Improvements

## Possible enhancements:

Deploy ML model using FastAPI
Create REST prediction API
Deploy dashboard on cloud
Add automated model retraining pipeline
Implement ML model monitoring
Add explainable AI using SHAP
Implement MLOps workflow
⚠️ Disclaimer

This project is developed for educational and portfolio purposes.

Machine learning predictions represent probability-based estimates and should support, not replace, business decisions.

👨‍💻 Author

## Atif Mahmood

Data Scientist | Data Analytics | Machine Learning
