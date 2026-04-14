import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model, features = joblib.load("model.pkl")

# Feature importance
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=True)

# Plot
plt.figure()
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Impact on CGPA")
plt.title("Subject Impact on CGPA")
plt.show()