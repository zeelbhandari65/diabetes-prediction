import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

df = load_data()

# Prepare the data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# Set up Streamlit
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.write("This app predicts whether a person is diabetic using medical data.")

# Sidebar inputs
st.sidebar.header("Enter Patient Data")

pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose Level", 0, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 0, 140, 70)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0.0, 900.0, 80.0)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 10, 100, 30)

# Create input data
input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [bp],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.error(f"🔴 The person is **diabetic** with {prob*100:.2f}% probability.")
    else:
        st.success(f"🟢 The person is **not diabetic** with {prob*100:.2f}% probability.")

# Expanders
with st.expander("📊 View Raw Dataset"):
    st.dataframe(df)

with st.expander("ℹ️ About this App"):
    st.markdown("""
    - Trained using Logistic Regression on Pima Indians Diabetes dataset.
    - Inputs are scaled using StandardScaler.
    - This app is for **educational/demo purposes** only.
    """)
