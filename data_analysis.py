import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from database_connection import load_churn_data

df = load_churn_data()

print("Dataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nMissing Values")
print(df.isnull().sum())

print("\nChurn Distribution")

print(df['churn'].value_counts())

# Churn Percentage
print("\nChurn Percentage")
print(df['churn'].value_counts(normalize=True)*100)

numeric_df = df.select_dtypes(
    include=['int64','float64']
)

# Correlation Analysis
plt.figure(figsize=(12,8))

sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Plot churn
sns.countplot(data=df, x="churn")

plt.title("Netflix Customer Churn Distribution")
plt.show()

# Subscription churn
subscription_churn = (
    df.groupby(
        "subscription_plan"
    )["churn"]
    .mean()
)

print(subscription_churn)

subscription_churn.plot(
    kind="bar"
)

plt.title(
    "Churn Rate By Subscription"
)

plt.ylabel(
    "Churn Rate"
)

plt.show()

payment_churn = (
    df.groupby(
        "payment_failures"
    )["churn"]
    .mean()
)


print(payment_churn)

print(
df.groupby(
'churn'
)[
'watch_hours_last_30_days'
]
.mean()
)

print(
df.groupby(
'churn'
)[
'days_since_last_login'
]
.mean()
)

print(df.groupby('churn')['watch_hours_last_30_days'].mean())