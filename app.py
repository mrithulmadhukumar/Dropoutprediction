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
        raise ValueError("❌ Unexpected SHAP structure.")

# --- Store Prediction + SHAP in DB ---
def store_prediction(student_id, prediction, shap_values=None):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cols = ["Student_ID", "Prediction_Result"] + features
    columns_sql = ", ".join([f"`{col}`" for col in cols])  # Escape column names
    placeholders = ", ".join(["%s"] * len(cols))
    insert_sql = f"""
        INSERT INTO predictions ({columns_sql})
        VALUES ({placeholders})
    """

    if shap_values:
        values = [student_id, prediction] + [float(v) for v in shap_values]
    else:
        values = [student_id, prediction] + [None] * len(features)

    cursor.execute(insert_sql, values)
    conn.commit()
    cursor.close()
    conn.close()

# --- Streamlit UI ---
st.set_page_config(page_title="Student Dropout Predictor", layout="centered")
st.title("🎓 Student Dropout Prediction")

student_id = st.number_input("Enter Student ID", step=1, format="%d")

if st.button("Predict Dropout"):
    student_df = get_student_data(student_id)

    if student_df.empty:
        st.error("❌ Student ID not found in the database.")
    else:
        input_df = student_df[features]
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        st.success(f"🎯 Prediction: {'Dropout' if prediction == 1 else 'Not Dropout'}")
        st.info(f"📊 Dropout Probability: {probability:.2%}")

        st.markdown("### Student Feature Snapshot")
        st.dataframe(input_df.T)

        if prediction == 1:
            st.warning("⚠️ Student at risk of dropping out — computing explanations...")
            shap_values = compute_shap_values(input_df)
            store_prediction(student_id, prediction, shap_values)
            st.success("✅ SHAP explanations stored in the database.")

            # --- Top 10 SHAP Contributors Chart ---


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
            store_prediction(student_id, prediction)
            st.success("✅ Prediction stored without SHAP values (student not at risk).")
