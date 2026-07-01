import pandas as pd
import matplotlib.pyplot as plt


importance = pd.read_csv(
    "feature_importance.csv"
)


top_features = importance.head(15)


plt.figure(figsize=(8,6))


plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)


plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top Churn Prediction Features"
)


plt.gca().invert_yaxis()


plt.tight_layout()

plt.savefig(
    "feature_importance_plot.png"
)

plt.show()