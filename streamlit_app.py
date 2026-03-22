import streamlit as st
import pandas as pd
import joblib
import sklearn
import requests
import io
import time

# --- 1. ตั้งค่า Theme สีดำ-เขียว (Custom CSS) ---
st.set_page_config(page_title="Salary Predictor Pro", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #000000;
        color: #00FF00;
    }
    .stButton>button {
        background-color: #00FF00;
        color: black;
        border-radius: 5px;
        font-weight: bold;
    }
    .stSelectbox, .stSlider {
        color: #00FF00 !important;
    }
    h1, h2, h3 {
        color: #00FF00 !important;
        font-family: 'Courier New', Courier, monospace;
    }
    /* ปรับแต่งกรอบ Input */
    div[data-baseweb="select"] > div {
        background-color: #1a1a1a !important;
        color: #00FF00 !important;
        border: 1px solid #00FF00;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ฟังก์ชันโหลด Model และ Assets (ใช้ Cache) ---
@st.cache_resource
def load_assets():
    # เปลี่ยน URL เป็นลิงก์ Release ของคุณ (อันนี้คือตัวอย่าง)
    url = "https://github.com/pratchaya-loyprakhon/ProjectY2_DS_67160225_2/releases/download/v2.0/salary_model.pkl"
    try:
        response = requests.get(url, timeout=300)
        model = joblib.load(io.BytesIO(response.content))
        encoders = joblib.load('encoders.pkl')
        features_list = joblib.load('features_list.pkl')
        return model, encoders, features_list
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

model, encoders, features_list = load_assets()

# --- 3. ส่วนการแสดงผล ---
st.title("SALARY PREDICTOR v2.0")
st.subheader("Database: salary_dataset_with_extra_feature")

if model:
    with st.form("hacker_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            rating = st.slider("Company Rating", 1.0, 5.0, 4.0)
            company = st.selectbox("Company Name", encoders['Company Name'].classes_)
            job_role = st.selectbox("Job Role", encoders['Job Role'].classes_) # เพิ่ม Job Role
            
        with col2:
            job_title = st.selectbox("Job Title", encoders['Job Title'].classes_)
            location = st.selectbox("Location", encoders['Location'].classes_)
            emp_status = st.selectbox("Employment Status", encoders['Employment Status'].classes_) # เพิ่ม Status
            
        submit = st.form_submit_button("RUN PREDICTION")

    if submit:
        try:
            # เตรียมข้อมูล (ต้องมั่นใจว่าชื่อ Key ตรงกับที่ใช้ Train)
            input_dict = {
                'Rating': rating,
                'Company Name': encoders['Company Name'].transform([company])[0],
                'Job Title': encoders['Job Title'].transform([job_title])[0],
                'Location': encoders['Location'].transform([location])[0],
                'Job Role': encoders['Job Role'].transform([job_role])[0],
                'Employment Status': encoders['Employment Status'].transform([emp_status])[0]
            }
            
            input_df = pd.DataFrame([input_dict])[features_list]
            prediction = model.predict(input_df)[0]
            
            st.write("---")
            st.markdown(f"<h2 style='text-align: center;'>ESTIMATED SALARY:</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; font-size: 50px;'>₹ {prediction:,.2f}</h1>", unsafe_allow_html=True)
            st.balloons()
            
        except Exception as e:
            st.error(f"Prediction Failed: {e}")
            st.info("ตรวจสอบว่าคอลัมน์ใน features_list ตรงกับข้อมูลที่กรอกหรือไม่")

else:
    st.warning("SYSTEM INITIALIZING... PLEASE WAIT")