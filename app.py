import streamlit as st
import pandas as pd
import joblib

# تصميم احترافي
st.set_page_config(page_title="Mobile Intelligence 2026", layout="wide", initial_sidebar_state="expanded")

# إضافة CSS مخصص لجعل الموقع يبدو كأنه تطبيق مدفوع
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .prediction-box { padding: 20px; border-radius: 10px; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# استدعاء الملفات
@st.cache_resource
def load_all():
    df = pd.read_csv('mobile_data_cleaned_2026.csv')
    model = joblib.load('mobile_model.pkl')
    return df, model

df, model = load_all()

# هيدر الموقع
st.title("🚀 Mobile Intelligence Hub")
st.markdown("---")

# تقسيم الصفحة (يسار للتوقع - يمين للإحصائيات)
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("🔮 Price Predictor")
    with st.container():
        ram = st.select_slider("RAM Capacity (GB)", options=[1, 2, 4, 6, 8, 12, 16], value=8)
        battery = st.slider("Battery (mAh)", 1000, 7000, 4500)
        camera = st.number_input("Main Camera (MP)", 2, 200, 48)
        weight = st.number_input("Device Weight (g)", 100, 500, 190)
        
        if st.button("Calculate Market Value"):
            res = model.predict([[battery, ram, weight, camera]])
            st.success(f"Estimated Value: €{res[0]:.2f}")

with right_col:
    st.subheader("📊 Market Insights")
    # إضافة رسوم بيانية توضح توزيع الأسعار في الـ 1943 موبايل
    st.bar_chart(df['brand'].value_counts().head(10))
