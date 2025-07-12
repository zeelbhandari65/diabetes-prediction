# 🩺 Diabetes Prediction Web App

A machine learning web application built with **Streamlit** to predict whether a person has diabetes based on medical input features. It uses a **Logistic Regression model** trained on the famous `diabetes.csv` dataset.

---

## 🚀 Live Demo

👉 https://diabetes-prediction-leuqhoclmvnkd2eehswuey.streamlit.app/
## 🧠 How It Works

- Dataset: `diabetes.csv`
- Model: Logistic Regression (`sklearn`)
- Features used:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age

User inputs these values, and the app predicts the likelihood of diabetes.

---

## 📁 Project Structure

diabetes-prediction/
├── app.py # Streamlit application
├── diabetes.csv # Input dataset
└── requirements.txt # Python dependencies
