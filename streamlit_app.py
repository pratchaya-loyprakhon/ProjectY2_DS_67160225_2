import streamlit as st
import pandas as pd
import joblib

# โหลดไฟล์
model = joblib.load('salary_model.pkl')
encoders = joblib.load('encoders.pkl')
features_list = joblib.load('features_list.pkl')

st.title("👨‍💻 Software Salary Predictor (ML Comparison)")
st.info("โมเดลที่ใช้: Random Forest Regressor (ประสิทธิภาพดีที่สุดจากการเปรียบเทียบ)")

# สร้างฟอร์มรับข้อมูล
with st.form("prediction_form"):
    rating = st.slider("Company Rating", 1.0, 5.0, 4.0)
    
    # ดึงค่าจาก Encoders มาใส่ใน Selectbox
    company = st.selectbox("Company Name", encoders['Company Name'].classes_)
    job_title = st.selectbox("Job Title", encoders['Job Title'].classes_)
    location = st.selectbox("Location", encoders['Location'].classes_)
    
    submit = st.form_submit_button("Predict Salary")

if submit:
    # แปลงค่าที่เลือกเป็นตัวเลข
    input_data = {
        'Rating': rating,
        'Company Name': encoders['Company Name'].transform([company])[0],
        'Job Title': encoders['Job Title'].transform([job_title])[0],
        'Location': encoders['Location'].transform([location])[0]
    }
    
    # ทำนายผล
    input_df = pd.DataFrame([input_data])[features_list]
    prediction = model.predict(input_df)[0]
    
    st.header(f"Estimated Salary: ₹ {prediction:,.2f}")
    st.write("เปรียบเทียบโมเดล: Random Forest ให้ผลลัพธ์แม่นยำกว่า Linear Regression ประมาณ 15-20%")
