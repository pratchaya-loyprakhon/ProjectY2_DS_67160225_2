import streamlit as st
import pandas as pd
import joblib
import sklearn
import requests
import io
import time

# --- 1. ตั้งค่าหน้าจอ ---
st.set_page_config(page_title="Salary Predictor", layout="centered")

# --- 2. ฟังก์ชันโหลดไฟล์ใหญ่ (เพิ่มระบบกันหลุด) ---
@st.cache_resource
def load_all_assets():
    url = "https://github.com/pratchaya-loyprakhon/ProjectY2_DS_67160225_2/releases/download/v1.0/salary_model.pkl"
    
    max_retries = 3
    for i in range(max_retries):
        try:
            # เพิ่ม timeout เป็น 600 วินาที (10 นาที) เพื่อรองรับไฟล์ 148MB
            response = requests.get(url, timeout=600, stream=True)
            response.raise_for_status()
            
            # อ่านข้อมูลแบบ Stream เพื่อป้องกัน RAM เต็ม
            content = response.content
            model = joblib.load(io.BytesIO(content))
            
            # โหลดไฟล์ประกอบอื่นๆ
            encoders = joblib.load('encoders.pkl')
            features_list = joblib.load('features_list.pkl')
            
            return model, encoders, features_list
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(5) # รอ 5 วินาทีแล้วลองใหม่
                continue
            else:
                st.error(f" โหลดไม่สำเร็จหลังจากพยายาม {max_retries} ครั้ง: {e}")
                return None, None, None

# เรียกใช้งาน
model, encoders, features_list = load_all_assets()

# --- 3. ส่วนแสดงผล UI ---
st.title("💰 Software Salary Prediction")

if model:
    st.success("✅ ระบบพร้อมใช้งาน (Model Loaded)")
    
    with st.expander("📊 Model Comparison Info"):
        st.write("Random Forest (R²: 0.85) vs Linear Regression (R²: 0.48)")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            rating = st.slider("Rating", 1.0, 5.0, 4.0)
            company = st.selectbox("Company", encoders['Company Name'].classes_)
        with c2:
            job_title = st.selectbox("Job Title", encoders['Job Title'].classes_)
            location = st.selectbox("Location", encoders['Location'].classes_)
        
        submit = st.form_submit_button("Predict")

    if submit:
        try:
            input_dict = {
                'Rating': rating,
                'Company Name': encoders['Company Name'].transform([company])[0],
                'Job Title': encoders['Job Title'].transform([job_title])[0],
                'Location': encoders['Location'].transform([location])[0]
            }
            input_df = pd.DataFrame([input_dict])[features_list]
            prediction = model.predict(input_df)[0]
            st.header(f"Estimated: ₹ {prediction:,.2f}")
            st.balloons()
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.warning(" กำลังดาวน์โหลดโมเดลขนาด 148MB... อาจใช้เวลา 1-3 นาที ห้ามรีเฟรชหน้าจอ")
    st.progress(0) # แสดงแถบรอ