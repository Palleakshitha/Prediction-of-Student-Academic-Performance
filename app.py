import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestRegressor
model_path = Path(__file__).resolve().parent.parent / "model" / "model.pkl"
model, features = joblib.load(model_path)
st.set_page_config(page_title="Student CGPA Predictor", layout="wide")
ui_scale = 1.2
st.sidebar.header("Display")
ui_scale = st.sidebar.slider("UI scale", 1.0, 1.6, ui_scale, 0.05)
_css = f"""
<style>
html, body, [data-testid="stAppViewContainer"] * {{
  font-size: {16*ui_scale}px;
}}
h1 {{ font-size: {36*ui_scale}px; }}
h2 {{ font-size: {28*ui_scale}px; }}
h3 {{ font-size: {22*ui_scale}px; }}
div[data-testid="stMetricValue"] {{
  font-size: {28*ui_scale}px;
}}
div[data-testid="stMetricLabel"] {{
  font-size: {16*ui_scale}px;
}}
label, .stCheckbox, .stRadio, .stSelectbox, .stTextInput, .stNumberInput {{
  font-size: {16*ui_scale}px;
}}
</style>
"""
st.markdown(_css, unsafe_allow_html=True)
st.title("Prediction of Student Academic Performance")
st.subheader("Enter Student Academic Details")
attendance = st.slider("Attendance (%)", 50, 100, 75)
maths = st.number_input("Engineering Mathematics Marks", 0, 100, 45)
ds = st.number_input("Data Structures Marks", 0, 100, 45)
os = st.number_input("Operating Systems Marks", 0, 100, 45)
cn = st.number_input("Computer Networks Marks", 0, 100, 45)
dbms = st.number_input("Database Management Marks", 0, 100, 45)
assignment = st.number_input("Assignment Marks", 0, 100, 70)
reading = st.slider("Reading Time (hrs/day)", 0.0, 5.0, 1.5)
writing = st.slider("Writing Time (hrs/day)", 0.0, 5.0, 1.0)
vals = {}
vals["Attendance_%"] = attendance
vals["Engineering_Mathematics_Marks"] = maths
vals["Data_Structures_Marks"] = ds
vals["Operating_Systems_Marks"] = os
vals["Computer_Networks_Marks"] = cn
vals["Database_Management_Marks"] = dbms
vals["Assignment_Marks"] = assignment
vals["Reading_Time_hrs"] = float(reading)
vals["Writing_Time_hrs"] = float(writing)
subj = [maths, ds, os, cn, dbms]
study_hours = float(reading) + float(writing)
avg_marks = float(np.mean(subj))
core_cs_avg = float(np.mean([ds, os, cn, dbms]))
attendance_adj = attendance / 100.0
vals["Study_Hours"] = study_hours
vals["Avg_Subject_Marks"] = avg_marks
vals["Core_CS_Avg"] = core_cs_avg
vals["Attendance_Adj"] = attendance_adj
vals["Attendance_Weighted_Avg"] = avg_marks * attendance_adj
vals["Assignment_Interaction"] = assignment * avg_marks / 100.0
vals["High_Score_Count_75"] = int(np.sum(np.array(subj) >= 75))
vals["Low_Score_Count_60"] = int(np.sum(np.array(subj) < 60))
vals["Marks_Variance"] = float(np.var(subj, ddof=0))
vals["Weighted_Avg_CS"] = (
    0.2 * maths + 0.25 * ds + 0.2 * os + 0.2 * cn + 0.15 * dbms
)
vals["Study_Efficiency"] = assignment / (study_hours + 0.5)
row = [vals.get(col, 0.0) for col in features]
input_data = pd.DataFrame([row], columns=features)
if st.button("Predict CGPA"):
    base_pred = float(model.predict(input_data)[0])
    avg_marks_v = float(input_data.iloc[0]["Avg_Subject_Marks"]) if "Avg_Subject_Marks" in input_data.columns else avg_marks
    att_v = float(input_data.iloc[0]["Attendance_%"]) if "Attendance_%" in input_data.columns else attendance
    low_cnt = int(input_data.iloc[0]["Low_Score_Count_60"]) if "Low_Score_Count_60" in input_data.columns else int(np.sum(np.array(subj) < 60))
    w_avg_cs = float(input_data.iloc[0]["Weighted_Avg_CS"]) if "Weighted_Avg_CS" in input_data.columns else (0.2 * maths + 0.25 * ds + 0.2 * os + 0.2 * cn + 0.15 * dbms)
    
    adj_pred = base_pred
    # Behavior adjustment: reading time reduces CGPA, writing time increases it
    behavior_adj = (writing * 0.1) - (reading * 0.1)
    adj_pred += behavior_adj
    
    if avg_marks_v >= 98 and att_v >= 98 and low_cnt == 0 and w_avg_cs >= 98 and assignment >= 98:
        predicted_cgpa = 10.0
    else:
        if avg_marks_v >= 95 and att_v >= 95 and low_cnt == 0 and w_avg_cs >= 92:
            adj_pred = max(adj_pred, 9.2)
        elif avg_marks_v >= 90 and att_v >= 90 and low_cnt == 0 and w_avg_cs >= 88:
            adj_pred = max(adj_pred, 8.7)
        elif avg_marks_v >= 85 and att_v >= 85 and low_cnt == 0 and w_avg_cs >= 85:
            adj_pred = max(adj_pred, 8.2)
        predicted_cgpa = min(10.0, adj_pred)

    st.success(f" Predicted CGPA: {predicted_cgpa:.2f}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted CGPA", f"{predicted_cgpa:.2f}")
    c2.metric("Avg Subject Marks", f"{avg_marks_v:.1f}")
    c3.metric("Weighted Avg CS", f"{w_avg_cs:.1f}")
    c4.metric("Attendance", f"{att_v:.0f}%")
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Subjects", "Engineered Features", "Importance"])
    with tab1:
        cgpa_df = pd.DataFrame({"part": ["CGPA", "Remaining"], "value": [predicted_cgpa, max(0.0, 10.0 - predicted_cgpa)]})
        chart = alt.Chart(cgpa_df).mark_bar().encode(
            x=alt.X("sum(value):Q", scale=alt.Scale(domain=[0, 10])),
            color=alt.Color("part:N", scale=alt.Scale(range=["#2E7D32", "#E0E0E0"]), legend=None)
        ).properties(height=100, width="container").configure_axis(
            labelFontSize=int(12*ui_scale),
            titleFontSize=int(14*ui_scale)
        )
        st.altair_chart(chart, use_container_width=True)
    with tab2:
        subjects_df = pd.DataFrame({
            "Subject": ["Mathematics", "Data Structures", "Operating Systems", "Computer Networks", "DBMS"],
            "Marks": [maths, ds, os, cn, dbms]})

   
        bars = alt.Chart(subjects_df).mark_bar(color="#4C78A8").encode(
            x=alt.X("Subject:N", sort=None),
            y=alt.Y("Marks:Q", scale=alt.Scale(domain=[0, 100]))
        ).properties(height=320)

        r60 = alt.Chart(pd.DataFrame({"y": [60]})).mark_rule(
            color="#FF9800", strokeDash=[4, 4]
        ).encode(y="y:Q")

        r75 = alt.Chart(pd.DataFrame({"y": [75]})).mark_rule(
            color="#F44336", strokeDash=[4, 4]
        ).encode(y="y:Q")

        r90 = alt.Chart(pd.DataFrame({"y": [90]})).mark_rule(
            color="#2E7D32", strokeDash=[2, 2]
        ).encode(y="y:Q")

    # ✅ Combine properly
        chart = alt.layer(bars, r60, r75, r90).configure_axis(
            labelFontSize=int(12 * ui_scale),
            titleFontSize=int(14 * ui_scale)
        )

        st.altair_chart(chart, use_container_width=True)
    with tab3:
        feats = {
            "Avg_Subject_Marks": avg_marks_v,
            "Core_CS_Avg": core_cs_avg,
            "Attendance_Weighted_Avg": float(vals["Attendance_Weighted_Avg"]),
            "Study_Hours": float(vals["Study_Hours"]),
            "Assignment_Interaction": float(vals["Assignment_Interaction"]),
            "Marks_Variance": float(vals["Marks_Variance"])
        }
        df_feats = pd.DataFrame({"Feature": list(feats.keys()), "Value": list(feats.values())})
        show_df = df_feats.copy()
        st.dataframe(show_df, use_container_width=True, hide_index=True)
        plot_feats = df_feats[df_feats["Feature"].isin(["Avg_Subject_Marks", "Core_CS_Avg", "Attendance_Weighted_Avg"])]
        if not plot_feats.empty:
            ch = alt.Chart(plot_feats).mark_bar(color="#7B1FA2").encode(
                x=alt.X("Feature:N", sort=None),
                y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 100]))
            ).properties(height=240).configure_axis(
                labelFontSize=int(12*ui_scale),
                titleFontSize=int(14*ui_scale)
            )
            st.altair_chart(ch, use_container_width=True)
    with tab4:
        importance_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=True).tail(12)
        imp_chart = alt.Chart(importance_df).mark_bar(color="#1976D2").encode(
            y=alt.Y("Feature:N", sort=None),
            x=alt.X("Importance:Q")
        ).properties(height=340).configure_axis(
            labelFontSize=int(12*ui_scale),
            titleFontSize=int(14*ui_scale)
        )
        st.altair_chart(imp_chart, use_container_width=True)
    importance_df = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
    subjects = {
        "Engineering_Mathematics_Marks": maths,
        "Data_Structures_Marks": ds,
        "Operating_Systems_Marks": os,
        "Computer_Networks_Marks": cn,
        "Database_Management_Marks": dbms,
    }
    weak_subjects = [sub for sub, mark in subjects.items() if mark < 60]
    if weak_subjects:
        priority = importance_df[importance_df["Feature"].isin(weak_subjects)].sort_values(by="Importance", ascending=False)
        st.warning(" Subjects to Improve :")
        for sub in priority["Feature"]:
            st.write("➡️", sub.replace("_Marks", "").replace("_", " "))
    else:
        st.success(" Great! No weak subjects detected.")
