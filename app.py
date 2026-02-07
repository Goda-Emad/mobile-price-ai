import streamlit as st
import pandas as pd
import joblib
import altair as alt

# 1. إعدادات الصفحة
st.set_page_config(page_title="Mobile Price AI", page_icon="📱", layout="wide")

# 2. CSS للواجهة
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                    url("https://images.unsplash.com/photo-1616348436168-de43ad0db179?auto=format&fit=crop&q=80&w=2000");
        background-size: cover;
        background-position: center;
        color: white;
    }
    .main-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        width: 100%;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# 3. تحميل الموديل والداتا
@st.cache_data
def load_data():
    return pd.read_csv('mobile_data_cleaned_2026.csv')

@st.cache_resource
def load_model():
    return joblib.load('mobile_model.pkl')

try:
    data = load_data()
    model = load_model()

    # العنوان الرئيسي
    st.title("📱 AI Mobile Valuation Hub")
    st.markdown("### Predict market value based on 1,943 analyzed devices")
    st.write("---")

    # تقسيم الشاشة لعمودين
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("🔧 Technical Specs")
        ram = st.slider("RAM (GB)", 1, 64, 8, help="Amount of RAM in GB, affects performance")
        battery = st.slider("Battery (mAh)", 1000, 7000, 4500, help="Battery capacity in milliampere-hours")
        camera = st.slider("Main Camera (MP)", 2, 200, 50, help="Main camera resolution in megapixels")
        weight = st.number_input("Weight (grams)", 100, 500, 190, help="Device weight in grams")

        if weight < 100 or weight > 500:
            st.warning("Please enter a realistic weight!")

        predict_btn = st.button("Calculate Market Value")

    with col2:
        st.subheader("📊 Market Insights")
        # عرض رسم بياني متطور
        top_brands = data['Brand'].value_counts().head(10).reset_index()
        top_brands.columns = ['Brand', 'Count']
        chart = alt.Chart(top_brands).mark_bar(color="#4CAF50").encode(
            x=alt.X('Brand', sort='-y'),
            y='Count',
            tooltip=['Brand', 'Count']
        ).properties(width=500, height=400)
        st.altair_chart(chart, use_container_width=True)

    # مساحة لعرض النتيجة أسفل الأعمدة
    result_placeholder = st.empty()

    if predict_btn:
        input_data = pd.DataFrame([[ram, battery, camera, weight]], 
                                  columns=['RAM', 'Battery', 'Camera', 'Weight'])
        prediction = model.predict(input_data)[0]
        result_placeholder.success(f"### Estimated Value: €{prediction:,.2f}")
        result_placeholder.info("This price is based on the 2026 market trends learned by the AI.")

except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.info("Make sure 'mobile_model.pkl' and 'mobile_data_cleaned_2026.csv' are in the same folder.")

# 4. Footer
st.write("---")
st.markdown("Developed by [Goda Emad](https://www.linkedin.com/in/goda-emad/) | 2026 AI Portfolio")
