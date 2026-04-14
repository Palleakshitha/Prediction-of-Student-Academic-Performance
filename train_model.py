import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

base_dir = Path(__file__).resolve().parent.parent
data_path = base_dir / "data" / "student_performance_engineering_subjects_500.xlsx"
model_path = base_dir / "model" / "model.pkl"

df = pd.read_excel(data_path)

raw_cols = [
    "Attendance_%",
    "Engineering_Mathematics_Marks",
    "Data_Structures_Marks",
    "Operating_Systems_Marks",
    "Computer_Networks_Marks",
    "Database_Management_Marks",
    "Assignment_Marks",
    "Reading_Time_hrs",
    "Writing_Time_hrs",
]

subj_cols = [
    "Engineering_Mathematics_Marks",
    "Data_Structures_Marks",
    "Operating_Systems_Marks",
    "Computer_Networks_Marks",
    "Database_Management_Marks",
]

X_raw = df[raw_cols].copy()
X_raw["Study_Hours"] = X_raw["Reading_Time_hrs"] + X_raw["Writing_Time_hrs"]
X_raw["Avg_Subject_Marks"] = X_raw[subj_cols].mean(axis=1)
X_raw["Core_CS_Avg"] = X_raw[
    ["Data_Structures_Marks", "Operating_Systems_Marks", "Computer_Networks_Marks", "Database_Management_Marks"]
].mean(axis=1)
X_raw["Attendance_Adj"] = X_raw["Attendance_%"] / 100.0
X_raw["Attendance_Weighted_Avg"] = X_raw["Avg_Subject_Marks"] * X_raw["Attendance_Adj"]
X_raw["Assignment_Interaction"] = X_raw["Assignment_Marks"] * X_raw["Avg_Subject_Marks"] / 100.0
X_raw["High_Score_Count_75"] = (X_raw[subj_cols] >= 75).sum(axis=1)
X_raw["Low_Score_Count_60"] = (X_raw[subj_cols] < 60).sum(axis=1)
X_raw["Marks_Variance"] = X_raw[subj_cols].var(axis=1)
X_raw["Weighted_Avg_CS"] = (
    0.2 * X_raw["Engineering_Mathematics_Marks"]
    + 0.25 * X_raw["Data_Structures_Marks"]
    + 0.2 * X_raw["Operating_Systems_Marks"]
    + 0.2 * X_raw["Computer_Networks_Marks"]
    + 0.15 * X_raw["Database_Management_Marks"]
)
X_raw["Study_Efficiency"] = X_raw["Assignment_Marks"] / (X_raw["Study_Hours"] + 0.5)

feature_cols = [
    "Attendance_%",
    "Engineering_Mathematics_Marks",
    "Data_Structures_Marks",
    "Operating_Systems_Marks",
    "Computer_Networks_Marks",
    "Database_Management_Marks",
    "Assignment_Marks",
    "Study_Hours",
    "Avg_Subject_Marks",
    "Core_CS_Avg",
    "Attendance_Weighted_Avg",
    "Assignment_Interaction",
    "High_Score_Count_75",
    "Low_Score_Count_60",
    "Marks_Variance",
    "Weighted_Avg_CS",
    "Study_Efficiency",
]

X = X_raw[feature_cols].copy()
y = df["CGPA"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

joblib.dump((model, feature_cols), model_path)

print("Model trained and saved")
