import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import mysql.connector
import plotly.graph_objects as go

from joblib import load
from datetime import datetime
from joblib import load

# Define custom wrapper class before loading
class DropoutPredictor:
    def __init__(self, model, threshold, features):
        self.model = model
        self.threshold = threshold
        self.features = features

    def predict(self, X):
        X = X[self.features]
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        X = X[self.features]
        return self.model.predict_proba(X)

# Now safely load the model
model = load("dropout_model_rf.joblib")
features = model.features


# --- Load studentdata.csv to infer data types for manual input form ---
full_data = pd.read_csv("studentdata.csv")
feature_types = full_data.drop(columns=["Target", "Target_encoded","Student_ID"], errors='ignore').dtypes.to_dict()

#--- Database Configuration ---
DB_CONFIG = {
    "host": st.secrets["connections"]["mysql"]["host"],
    "port": st.secrets["connections"]["mysql"]["port"],
    "user": st.secrets["connections"]["mysql"]["user"],
    "password": st.secrets["connections"]["mysql"]["password"],
    "database": st.secrets["connections"]["mysql"]["database"]
}

# DB_CONFIG = {
#     "host": "localhost",  # or 127.0.0.1
#     "port": 3306,
#     "user": "root",
#     "password": "Sajin@77",
#     "database": "dropoutpreddb"
# }

# DB_CONFIG = {
#     "host": "localhost",  # or 127.0.0.1
#     "port": 3306,
#     "user": "root",
#     "password": "Password@2000",
#     "database": "dropoutpreddb"
# }



# --- Database Access ---
def get_student_data(student_id):
    conn = mysql.connector.connect(**DB_CONFIG)
    query = f"SELECT * FROM student_data WHERE Student_ID = {student_id}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --- SHAP Computation ---
def compute_shap_values(input_df):
    explainer = shap.TreeExplainer(model.model)  # Access the internal sklearn model
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
def store_prediction(student_id, prediction, probability, name, email, shap_values=None):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cols = ["Student_ID", "Prediction_Result", "Dropout_Probability", "Name", "Email"] + features
    columns_sql = ", ".join([f"`{col}`" for col in cols])  # Escape column names
    placeholders = ", ".join(["%s"] * len(cols))
    insert_sql = f"""
        INSERT INTO predictions ({columns_sql})
        VALUES ({placeholders})
    """

    # 🛠 FIX HERE
    if shap_values is not None and not isinstance(shap_values, (list, np.ndarray)):
        shap_values = [shap_values]  # wrap float to make iterable

    if shap_values is not None:
        values = [student_id, prediction, probability, name, email] + [
            float(v) if v is not None else None for v in shap_values
        ]
    else:
        values = [student_id, prediction, probability, name, email] + [None] * len(features)

    if len(values) != len(cols):
        raise ValueError(f"🚨 Mismatch: {len(values)} values for {len(cols)} columns")

    if shap_values is not None and len(features) != len(shap_values):
        raise ValueError(f"SHAP value count ({len(shap_values)}) doesn't match feature count ({len(features)})")

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
        name = student_df["Name"].iloc[0] if "Name" in student_df.columns else None
        email = student_df["Email"].iloc[0] if "Email" in student_df.columns else None
        prediction = int(model.predict(input_df)[0])
        probability = round(float(model.predict_proba(input_df)[0][1]) * 100, 2)

        st.success(f" Prediction: {'Dropout' if prediction == 1 else 'Not Dropout'}")
        st.info(f" Dropout Probability: {probability}%")


        if prediction == 1:
            st.warning(" Student at risk of dropping out — computing explanations...")
            shap_values = compute_shap_values(input_df)
            store_prediction(student_id, prediction,probability, name, email, shap_values)
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
            store_prediction(student_id, prediction,probability, name, email)
            st.success(" Prediction stored without SHAP values (student not at risk).")


#file upload section

#to check and update student_data
def upsert_student_data(row):
    """Insert or update student_data table with given row"""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    columns = list(row.index)
    placeholders = ", ".join(["%s"] * len(columns))
    col_sql = ", ".join([f"`{col}`" for col in columns])

    update_sql = ", ".join([f"`{col}`=VALUES(`{col}`)" for col in columns if col != "Student_ID"])

    sql = f"""
        INSERT INTO student_data ({col_sql})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_sql}
    """

    values = [row[col] for col in columns]
    cursor.execute(sql, values)
    conn.commit()
    cursor.close()
    conn.close()


#batch prediction function
def run_batch_predictions(df):
    try:
        input_df = df[model.features].copy()
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1] * 100

        results = []
        for i, row in df.iterrows():
            student_id = row["Student_ID"]
            name = row.get("Name", None)
            email = row.get("Email", None)

            # ✅ INSERT student_data entry exactly once per student
            upsert_student_data(row)
#removed itrow
            input_row = pd.DataFrame([row[features].values], columns=features)

            prediction = int(model.predict(input_row)[0])
            probability = round(float(model.predict_proba(input_row)[0][1]) * 100, 2)

            # SHAP values only for predicted dropouts
            if prediction == 1:
                explainer = shap.TreeExplainer(model.model)
                shap_values = explainer.shap_values(input_row[features])

                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_row = shap_values[1][0]  # Class 1
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    shap_row = shap_values[0, :, 1]
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                    shap_row = shap_values[0]
                else:
                    raise ValueError("❌ Unexpected SHAP format.")

                shap_row = shap_row.tolist()
            else:
                shap_row = [None] * len(features)

            store_prediction(student_id, prediction, probability, name, email, shap_row)

            results.append({
                "Student_ID": student_id,
                "Prediction_Result": "Dropout" if prediction == 1 else "Not Dropout",
                "Dropout_Probability": probability,
                **{f: shap_row[j] for j, f in enumerate(model.features)}
            })

        return pd.DataFrame(results)

    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
        return None
#


st.markdown("---")
st.header("📁 Upload File for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV or Excel file ", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            uploaded_df = pd.read_excel(uploaded_file)
        else:
            uploaded_df = pd.read_csv(uploaded_file)

        # Check if required columns are present
        if "Student_ID" not in uploaded_df.columns:
            st.error("Uploaded file must contain 'Student_ID'")
            st.stop()

        missing_features = [f for f in model.features if f not in uploaded_df.columns]
        if missing_features:
            st.error(f"Missing features: {', '.join(missing_features)}")
            st.stop()

        st.success(" File uploaded and validated!")
        st.dataframe(uploaded_df.head())

        if st.button(" Predict All"):
            with st.spinner("Running batch predictions..."):
                prediction_df = run_batch_predictions(uploaded_df)
            if prediction_df is not None:  # For prediction probabilities from batch prediction
                st.success("Predictions completed and stored in database.")

                # Show the results in the UI
                st.subheader("Prediction Results")
                st.dataframe(prediction_df[["Student_ID", "Prediction_Result", "Dropout_Probability"]].sort_values(
                    by="Dropout_Probability", ascending=False))

                # visualising shap values (Top Features Contributing to Dropouts):
                # Only keep SHAP columns
                shap_cols = [col for col in prediction_df.columns if
                             col not in ["Student_ID", "Prediction_Result", "Dropout_Probability"]]
                shap_df = prediction_df[shap_cols].dropna()

                # Mean absolute SHAP values
                mean_shap = shap_df.abs().mean().sort_values(ascending=False).head(10)

                # Prepare data for Plotly
                features = mean_shap.index.tolist()
                shap_values = mean_shap.values.tolist()
                colors = ['#AEC6CF'] * len(features)  # soft green for all bars

                # Create Plotly bar chart
                fig = go.Figure(go.Bar(
                    x=shap_values,
                    y=features,
                    orientation='h',
                    marker_color=colors
                ))

                fig.update_layout(
                    title="Top 10 Predictive Features (Mean Absolute SHAP Values)",
                    xaxis_title="Mean SHAP Value",
                    yaxis_title="Feature",
                    width=500,
                    height=600
                )

                fig.update_yaxes(autorange="reversed")  # Highest at top
                st.plotly_chart(fig)

                # # dropout vs non-dropout
                # st.subheader("Dropout vs Not Dropout Count")
                # Count values
                count_data = prediction_df["Prediction_Result"].value_counts()
                labels = count_data.index.tolist()
                values = count_data.values.tolist()

                # Define custom colors
                colors = ['#A8D5BA' if label.lower() == 'not dropout' else '#F4A6A6' for label in labels]

                # Create the bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=labels,
                        y=values,
                        marker_color=colors
                    )
                ])

                fig.update_layout(
                    title="Dropout vs Not Dropout Count",
                    xaxis_title="Prediction Result",
                    yaxis_title="Number of Students",
                    width=400,
                    height=600
                )
                st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")

