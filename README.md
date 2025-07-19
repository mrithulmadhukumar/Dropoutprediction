Student Dropout Prediction System
A predictive analytics solution designed to identify at-risk students early using behavioral, academic, mental health, and physiological indicators. This project aims to support timely, data-driven interventions that reduce student dropout rates and improve retention in higher education.

Overview
Student dropout is a persistent issue in global education, with over 40% of undergraduates failing to complete their degrees. This project addresses that challenge by integrating multi-source data and machine learning to build a holistic, interpretable dropout prediction system.

Problem Statement
"Spotting struggling students early is hard for universities, which leads to late support and more dropouts."

Although institutions collect data, they often lack real-time predictive tools to intervene early. Our solution bridges this gap with a practical, scalable system designed for academic advisors, counselors, and university administrators.

Project Objectives
Detect high-risk students early using machine learning

Enrich academic data with mental, behavioral, and health indicators

Present actionable insights through dashboards and explainable AI

Support timely and personalized intervention strategies

Datasets Used
Dataset	Source	Description
edudata	UCI Repository	Core academic, demographic, and financial data (4424 records)
mental_health_data	Survey (IIUM)	Self-reported mental health indicators: anxiety, depression, CGPA
performance_data	Survey (Bangladesh)	Academic behavior, scholarships, study time, transport, etc.
health_data	Simulated	Biosensor metrics: heart rate, blood pressure, sleep, stress, physical activity

Data Integration
We used K-Means clustering to integrate datasets without a common identifier. Each dataset was clustered independently, and semantic links were created using cluster assignments. Key steps:

Standardization and normalization using MinMaxScaler

Feature engineering (e.g., GPA normalization, performance ratio)

Enrichment of edudata with psychological and behavioral features

Synthetic ID generation (Name, Email, Student_ID) using SDV

Behavioral tagging (e.g., StudentWorker, Living_With_Family)

See: integration_workflow.png

Modeling Approach
Model: Random Forest (selected after comparing with SVM, Logistic Regression, KNN)

Class balancing: SMOTE for dropout vs graduate imbalance

Feature selection: RFE using SHAP values with cross-validation

Performance:

Accuracy: ~95%

Recall (Dropout): ~92%

ROC AUC: 0.973

Tech Stack
Python (pandas, scikit-learn, shap, imbalanced-learn)

Streamlit (Web app for advisors)

MySQL (Cloud-hosted student data and predictions)

Power BI (Institution-wide analytics dashboard)

SDV (Synthetic data generation)

Dashboard Features
Streamlit App
Individual prediction via Student ID (with SHAP explainability)

Batch upload with live predictions

Integration with MySQL for data and result persistence

Power BI Dashboards
University overview: dropout risk distribution, top risk factors

Student profile: GPA, attendance, dropout probability, risk trend

SDG Alignment
SDG 4: Quality Education – Enhancing personalized support

SDG 10: Reduced Inequalities – Identifying vulnerable students early


Team and Collaboration
This project was developed through agile collaboration using Trello, GitHub, Google Colab, and MS Teams. The team focused on real-world impact, transparency, and ethical AI practices throughout the process.

Business Value
Enables early, personalized intervention

Reduces institutional dropout and financial loss

Strengthens advisor-student engagement

Supports data-informed decision-making across the university

How to Use
Clone the repository:
git clone https://github.com/your-username/student-dropout-prediction.git

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run streamlit_app/app.py

View Power BI dashboards (exports available in dashboards/)

