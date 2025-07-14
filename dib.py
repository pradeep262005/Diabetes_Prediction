import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    df = pd.read_csv('diabetes.csv')
    for column in ['Glucose', 'BloodPressure', 'Insulin', 'BMI']:
        df[column] = df[column].replace(0, np.nan)
        df[column] = df[column].fillna(df[column].median())
        
    return df


df = load_data()


st.title("🩺 Diabetes Prediction App")


if st.checkbox("🔍 Show Raw Data"):
    st.write(df)

try:
    X = df.drop(columns=['SkinThickness', 'outcome'], axis=1)
    Y = df['outcome']
except KeyError:
    st.error("Check column names in your CSV file. 'outcome' or 'SkinThickness' might be missing.")


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = DecisionTreeClassifier()
model.fit(X_train_scaled, Y_train)

accuracy = accuracy_score(Y_test, model.predict(X_test_scaled))
st.write(f"✅ Model Accuracy: **{accuracy:.2f}**")

st.subheader("📝 Enter Your Medical Details")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("🎯 Predict"):
    input_data = np.array([[pregnancies, glucose, bp, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("🚨 The model predicts that you are likely **diabetic**.")
    else:
        st.success("✅ The model predicts that you are **not diabetic**.")

