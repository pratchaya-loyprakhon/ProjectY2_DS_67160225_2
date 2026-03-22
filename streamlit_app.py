import streamlit as st
import pandas as pd
import joblib

# โหลดโมเดลและตัวแปลง
model = joblib.load('salary_model.pkl')
encoders = joblib.load('encoders.pkl')
features_list = joblib.load('features_list.pkl')

st.title("Salary Prediction App")

# สร้างช่องกรอกข้อมูลตามคอลัมน์ที่มี
input_data = {}
input_data['Rating'] = st.slider("Rating", 1.0, 5.0, 4.0)

for col in ['Company Name', 'Job Title', 'Location']:
    # ดึงรายชื่อทั้งหมดในคอลัมน์นั้นมาทำเป็นตัวเลือก
    options = encoders[col].classes_
    selected = st.selectbox(f"Select {col}", options)
    input_data[col] = encoders[col].transform([selected])[0]

if st.button("Predict"):
    # เรียงลำดับคอลัมน์ให้ตรงกับตอน Train
    df_input = pd.DataFrame([input_data])[features_list]
    prediction = model.predict(df_input)
    st.header(f"Estimated Salary: ₹ {prediction[0]:,.2f}")
