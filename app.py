import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import mysql.connector
from joblib import load
from datetime import datetime

# --- Load Model and Feature Columns ---
model = load("dropout_model_rf.joblib")
features = pd.read_csv("model_features.csv").squeeze().tolist()

# --- Load studentdata.csv to infer data types for manual input form ---
full_data = pd.read_csv("studentdata.csv")
feature_types = full_data.drop(columns=["Target", "Target_encoded","Student_ID"], errors='ignore').dtypes.to_dict()

# --- Database Configuration ---
DB_CONFIG = {
    "host": st.secrets["connections"]["mysql"]["host"],
    "port": st.secrets["connections"]["mysql"]["port"],
    "user": st.secrets["connections"]["mysql"]["user"],
    "password": st.secrets["connections"]["mysql"]["password"],
    "database": st.secrets["connections"]["mysql"]["database"]
}


# --- Database Access ---
def get_student_data(student_id):
    conn = mysql.connector.connect(**DB_CONFIG)
    query = f"SELECT * FROM student_data WHERE Student_ID = {student_id}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --- SHAP Computation ---
def compute_shap_values(input_df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # Handle SHAP output variations
    if isinstance(shap_values, list) and len(shap_values) == 2:
        return shap_values[1][0].tolist()
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[0, :, 1].tolist()
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        return shap_values[0].tolist()
    else:
        raise ValueError("Unexpected SHAP structure.")

# --- Store Prediction + SHAP in DB ---
def store_prediction(student_id, prediction,probability, shap_values=None):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cols = ["Student_ID", "Prediction_Result","Dropout_Probability"] + features
    columns_sql = ", ".join([f"`{col}`" for col in cols])  # Escape column names
    placeholders = ", ".join(["%s"] * len(cols))
    insert_sql = f"""
        INSERT INTO predictions ({columns_sql})
        VALUES ({placeholders})
    """

    if shap_values:
        values = [student_id, prediction, probability] + [float(v) for v in shap_values]
    else:
        values = [student_id, prediction , probability] + [None] * len(features)

    cursor.execute(insert_sql, values)
    conn.commit()
    cursor.close()
    conn.close()

# --- Streamlit UI ---
st.set_page_config(page_title="Student Dropout Predictor", layout="centered")
st.title("🎓 Student Dropout Prediction")

# --- Prediction from Existing Database Entry ---
st.subheader(" Predict Using Student ID from Database")

student_id = st.number_input("Enter Student ID", step=1, format="%d")

if st.button("Predict Dropout"):
    student_df = get_student_data(student_id)

    if student_df.empty:
        st.error(" Student ID not found in the database.")
    else:
        input_df = student_df[features]
        prediction = int(model.predict(input_df)[0])
        probability = round(float(model.predict_proba(input_df)[0][1]) * 100, 2)

        st.success(f" Prediction: {'Dropout' if prediction == 1 else 'Not Dropout'}")
        st.info(f" Dropout Probability: {probability}%")

        st.markdown("### Student Feature Snapshot")
        st.dataframe(input_df.T)

        if prediction == 1:
            st.warning(" Student at risk of dropping out — computing explanations...")
            shap_values = compute_shap_values(input_df)
            store_prediction(student_id, prediction, probability, shap_values)
            st.success(" SHAP explanations stored in the database.")

            shap_series = pd.Series(shap_values, index=features)
            top_factors = shap_series.abs().sort_values(ascending=False).head(10)
            top_features = top_factors.index
            top_values = shap_series[top_features]

            fig, ax = plt.subplots()
            top_values.plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title("Top 10 Contributing Factors to Dropout")
            ax.set_xlabel("SHAP Value")
            ax.invert_yaxis()
            st.pyplot(fig)

        else:
            store_prediction(student_id, prediction, probability)
            st.success(" Prediction stored without SHAP values (student not at risk).")

# # --- Manual Input Section (shown only after button click) ---
# st.markdown("---")
# show_manual = st.button("📝 Or Click Here for Manual Entry")
#
# if show_manual:
#     st.header("📝 Manual Entry for New Student")
#
#     # Define dropdown categories
#     binary_dropdowns = {
#         "Living_With_Family", "StudentWorker", "Health_Issues_Reported", "Teacher_Consultancy",
#         "Transport_User", "Mental_Depression", "Mental_Anxiety", "International",
#         "Scholarship holder", "Tuition fees up to date", "Debtor",
#         "Educational special needs", "Displaced", "Daytime/evening attendance"
#     }
#
#     level_dropdowns = {
#         "Physical_Activity", "Sleep_Quality_Proxy", "Health_Risk_Proxy"
#     }
#
#     level_options = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
#     gender_options = {"Male": 1, "Female": 0}
#
#     with st.form("manual_entry_form"):
#         manual_inputs = {}
#         st.markdown("#### Enter Feature Values")
#
#         # Student ID input first
#         student_id_manual = st.number_input("Enter Student ID (5-digit)", min_value=10000, max_value=99999, step=1, format="%d")
#
#         for feature, dtype in feature_types.items():
#             if feature == "Student_ID":
#                 continue  # skip since we're collecting it separately
#             if feature == "Gender":
#                 gender = st.selectbox(f"{feature}", options=list(gender_options.keys()), index=0)
#                 manual_inputs[feature] = gender_options[gender]
#             elif feature in binary_dropdowns:
#                 choice = st.selectbox(f"{feature}", options=["Yes", "No"], index=0)
#                 manual_inputs[feature] = 1 if choice == "Yes" else 0
#             elif feature in level_dropdowns:
#                 level = st.selectbox(f"{feature}", options=list(level_options.keys()), index=0)
#                 manual_inputs[feature] = level_options[level]
#             elif dtype == 'int64':
#                 manual_inputs[feature] = st.number_input(f"{feature}", step=1, format="%d")
#             elif dtype == 'float64':
#                 manual_inputs[feature] = st.number_input(f"{feature}", step=0.01, format="%.2f")
#
#         manual_submit = st.form_submit_button("Submit and Predict")
#
#     if manual_submit:
#         input_df = pd.DataFrame([manual_inputs])
#         prediction = int(model.predict(input_df[features])[0])
#         probability = round(float(model.predict_proba(input_df)[0][1]) * 100, 2)
#
#         st.success(f"🎯 Prediction: {'Dropout' if prediction == 1 else 'Not Dropout'}")
#         st.info(f"📊 Dropout Probability: {probability}%")
#
#         st.markdown("### Entered Student Features")
#         st.dataframe(input_df[features].T)
#
#         if prediction == 1:
#             st.warning("⚠️ Student at risk of dropping out — computing explanations...")
#             shap_values = compute_shap_values(input_df[features])
#             store_prediction(student_id=student_id_manual, prediction=prediction,probability=probability, shap_values=shap_values)
#             st.success("✅ SHAP explanations stored in the database.")
#
#             shap_series = pd.Series(shap_values, index=features)
#             top_factors = shap_series.abs().sort_values(ascending=False).head(10)
#             top_features = top_factors.index
#             top_values = shap_series[top_features]
#
#             fig, ax = plt.subplots()
#             top_values.plot(kind='barh', ax=ax, color='skyblue')
#             ax.set_title("Top 10 Contributing Factors to Dropout")
#             ax.set_xlabel("SHAP Value")
#             ax.invert_yaxis()
#             st.pyplot(fig)
#
#         else:
#             store_prediction(student_id=student_id_manual, prediction=prediction,probability=probability)
#             st.success("✅ Prediction stored without SHAP values (student not at risk).")
