import streamlit as st
import pandas as pd
import joblib
import requests
import io

# --- 1. ตั้งค่าหน้าจอและ Theme สีเขียว-ดำ ---
st.set_page_config(page_title="Salary Predictor Pro", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #000000; color: #00FF00; }
    .stButton>button { background-color: #00FF00; color: black; width: 100%; font-weight: bold; border: none; }
    .stSelectbox, .stSlider { color: #00FF00 !important; }
    h1, h2, h3, p, label { color: #00FF00 !important; font-family: 'Courier New', monospace; }
    div[data-baseweb="select"] > div { background-color: #1a1a1a !important; color: #00FF00 !important; border: 1px solid #00FF00; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. โหลดโมเดล (ใช้ Cache) ---
@st.cache_resource
def load_assets():
    # ตรวจสอบลิงก์ Release ของคุณให้ถูกต้อง
    url = "https://github.com/pratchaya-loyprakhon/ProjectY2_DS_67160225_2/releases/download/v3.0/salary_model.pkl"
    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        model = joblib.load(io.BytesIO(response.content))
        encoders = joblib.load('encoders.pkl')
        features_list = joblib.load('features_list.pkl')
        return model, encoders, features_list
    except Exception as e:
        st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
        return None, None, None

model, encoders, features_list = load_assets()

# --- 3. ส่วนการแสดงผล ---
st.title("SALARY PREDICTOR v2.0")
st.write("ระบบทำนายเงินเดือนที่ผ่านการจัดการ Outliers และเปรียบเทียบโมเดลแล้ว")

if model:
    with st.form("hacker_form"):
        col1, col2 = st.columns(2)
        with col1:
            rating = st.slider("Company Rating", 1.0, 5.0, 4.0)
            company = st.selectbox("Company Name", encoders['Company Name'].classes_)
            job_role_val = st.selectbox("Job Roles", encoders['Job Roles'].classes_) 
            
        with col2:
            job_title = st.selectbox("Job Title", encoders['Job Title'].classes_)
            location = st.selectbox("Location", encoders['Location'].classes_)
            emp_status = st.selectbox("Employment Status", encoders['Employment Status'].classes_)
            
        submit = st.form_submit_button("RUN PREDICTION")

    if submit:
        try:
            input_dict = {
                'Rating': rating,
                'Company Name': encoders['Company Name'].transform([company])[0],
                'Job Title': encoders['Job Title'].transform([job_title])[0],
                'Location': encoders['Location'].transform([location])[0],
                'Job Roles': encoders['Job Roles'].transform([job_role_val])[0],
                'Employment Status': encoders['Employment Status'].transform([emp_status])[0]
            }
            
            input_df = pd.DataFrame([input_dict])[features_list]
            
            # ทำนายและแปลงค่าเงิน (1 รูปี ≈ 0.43 บาท)
            pred_inr = model.predict(input_df)[0]
            pred_thb = (pred_inr * 0.43) * 0.8
            
            st.write("---")
            st.markdown("<h2 style='text-align: center;'>ประมาณการเงินเดือน:</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; font-size: 50px;'>{pred_thb:,.2f} บาท</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; opacity: 0.7;'>(เทียบเท่า ₹ {pred_inr:,.2f} รูปี)</p>", unsafe_allow_html=True)
            st.balloons()
            
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
else:
    st.info("ระบบกำลังเตรียมความพร้อม...")