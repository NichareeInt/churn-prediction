# 📡 Telco Customer Churn Predictor

ทำนายความเสี่ยงที่ลูกค้าจะยกเลิกบริการ Telco โดยใช้ Machine Learning
พร้อม Business Recommendation สำหรับทีม Marketing

## 🚀 Live Demo
👉 [Try the App](https://churn-prediction-cjq6qns8ennwenwasqilkb.streamlit.app/)

## 🎯 Business Problem
บริษัท Telco สูญเสียลูกค้าจาก churn ซึ่งมีต้นทุนสูงกว่าการหาลูกค้าใหม่ 5-7 เท่า
โปรเจกต์นี้สร้างโมเดลทำนายล่วงหน้าว่าลูกค้าคนไหนมีความเสี่ยงจะยกเลิกบริการ
เพื่อให้ทีม retention แทรกแซงได้ทันเวลา

## 📊 Dataset
- **แหล่งที่มา:** Kaggle — Telco Customer Churn
- **ขนาด:** 7,043 ลูกค้า, 21 features
- **Churn rate:** 26.5%

## 🔍 Methodology
1. EDA & Data Cleaning
2. Feature Engineering & Encoding
3. เปรียบเทียบ 3 Models: Logistic Regression, Random Forest, XGBoost
4. SHAP Values อธิบาย feature importance
5. Deploy เป็น Web App ด้วย Streamlit

## 📈 Model Performance
| Model | AUC-ROC |
|-------|---------|
| Logistic Regression | 0.836 |
| Random Forest | 0.814 |
| XGBoost | 0.820 |

**เลือกใช้ Logistic Regression** เพราะ AUC สูงสุดและอธิบายผลได้ง่ายกว่า

## 💡 Key Insights
- ลูกค้าสัญญา Month-to-month
