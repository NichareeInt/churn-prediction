import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ต้องอยู่บรรทัดแรกสุดก่อนคำสั่ง st. อื่นทุกอัน
st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="wide")

# โหลด model
@st.cache_resource
def load_models():
    with open('models/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, feature_names = load_models()

# ---- UI ----
st.title("📡 Telco Customer Churn Predictor")
st.markdown("กรอกข้อมูลลูกค้าเพื่อทำนายความเสี่ยง churn")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("ข้อมูลส่วนตัว")
    gender        = st.selectbox("เพศ", ["Male", "Female"])
    senior        = st.selectbox("ผู้สูงอายุ (65+)", ["No", "Yes"])
    partner       = st.selectbox("มีคู่", ["No", "Yes"])
    dependents    = st.selectbox("มีผู้พึ่งพา", ["No", "Yes"])
    tenure        = st.slider("อายุการใช้งาน (เดือน)", 0, 72, 12)

with col2:
    st.subheader("บริการที่ใช้")
    phone         = st.selectbox("บริการโทรศัพท์", ["No", "Yes"])
    multiple      = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet      = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security      = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup        = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("สัญญาและการชำระ")
    contract      = st.selectbox("ประเภทสัญญา", ["Month-to-month", "One year", "Two year"])
    paperless     = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment       = st.selectbox("วิธีชำระ", ["Electronic check", "Mailed check",
                                               "Bank transfer (automatic)",
                                               "Credit card (automatic)"])
    monthly       = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0)
    total         = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

# สร้าง input DataFrame
input_data = {
    'gender': 1 if gender == 'Male' else 0,
    'SeniorCitizen': 1 if senior == 'Yes' else 0,
    'Partner': 1 if partner == 'Yes' else 0,
    'Dependents': 1 if dependents == 'Yes' else 0,
    'tenure': tenure,
    'PhoneService': 1 if phone == 'Yes' else 0,
    'PaperlessBilling': 1 if paperless == 'Yes' else 0,
    'MonthlyCharges': monthly,
    'TotalCharges': total,
}

# One-hot encode ค่าที่เลือก
for col in ['MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'Contract', 'PaymentMethod']:
    val_map = {
        'MultipleLines': multiple,
        'InternetService': internet,
        'OnlineSecurity': security,
        'OnlineBackup': backup,
        'Contract': contract,
        'PaymentMethod': payment
    }
    options = {
        'MultipleLines': ['No phone service', 'Yes'],
        'InternetService': ['Fiber optic', 'No'],
        'OnlineSecurity': ['No internet service', 'Yes'],
        'OnlineBackup': ['No internet service', 'Yes'],
        'Contract': ['One year', 'Two year'],
        'PaymentMethod': ['Credit card (automatic)',
                          'Electronic check', 'Mailed check']
    }
    if col in options:
        for opt in options[col]:
            key = f"{col}_{opt}"
            input_data[key] = 1 if val_map[col] == opt else 0

# สร้าง DataFrame และเติม missing columns
input_df = pd.DataFrame([input_data])
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

# ทำนาย
st.divider()
if st.button("🔍 ทำนายความเสี่ยง Churn", type="primary", use_container_width=True):
    input_sc   = scaler.transform(input_df)
    prob       = model.predict_proba(input_sc)[0][1]
    prediction = model.predict(input_sc)[0]

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        if prediction == 1:
            st.error(f"⚠️ ความเสี่ยง Churn: **{prob*100:.1f}%**")
            st.markdown("ลูกค้ารายนี้มีความเสี่ยงสูงที่จะยกเลิกบริการ")
        else:
            st.success(f"✅ ความเสี่ยง Churn: **{prob*100:.1f}%**")
            st.markdown("ลูกค้ารายนี้น่าจะยังคงใช้บริการต่อ")

    with col_res2:
        st.metric("Churn Probability", f"{prob*100:.1f}%")
        st.progress(float(prob))

    # Business Recommendation
    st.subheader("💡 Business Recommendation")
    if prob >= 0.7:
        st.warning("**ความเสี่ยงสูง** — ควรโทรติดต่อทันทีและเสนอ retention package")
    elif prob >= 0.4:
        st.info("**ความเสี่ยงปานกลาง** — ส่ง personalized email และ loyalty points")
    else:
        st.success("**ความเสี่ยงต่ำ** — ติดตามตามปกติ ไม่จำเป็นต้องแทรกแซงพิเศษ")