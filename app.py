import streamlit as st
import pandas as pd
import joblib
import altair as alt
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ================== 1️⃣ إعدادات الصفحة ==================
st.set_page_config(page_title="AI Mobile Price Hub", page_icon="📱", layout="wide")

# ================== 2️⃣ CSS للواجهة (Light Mode) ==================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(255,255,255,0.8), rgba(255,255,255,0.8)),
                url("https://images.unsplash.com/photo-1616348436168-de43ad0db179?auto=format&fit=crop&q=80&w=2000");
    background-size: cover;
    background-position: center;
    color: black;
}
h1,h2,h3 { color:black !important; }
.stSlider label, .stNumberInput label, .stSelectbox label { 
    color:black !important; font-weight:bold; 
}
.stButton>button {
    background-color: #4CAF50; color: white; border-radius:20px; width:100%; border:none; transition:0.3s;
}
.stButton>button:hover { background-color:#45a049; transform: scale(1.02); }
</style>
""", unsafe_allow_html=True)

# ================== 3️⃣ تحميل البيانات والموديل ==================
@st.cache_data
def load_data():
    return pd.read_csv("mobile_data_cleaned_2026.csv")

@st.cache_resource
def load_model():
    return joblib.load("mobile_model.pkl")

data = load_data()
model = load_model()
features = model.feature_names_in_

# ================== 4️⃣ LabelEncoders آمن ==================
def safe_label_encoder(column, value):
    le = LabelEncoder()
    le.fit(data[column])
    if value not in le.classes_:
        le.classes_ = np.append(le.classes_, value)
    return le.transform([value])[0]

# ================== 5️⃣ واجهة المستخدم ==================
st.title("📱 AI Mobile Valuation Hub")
st.markdown("### Predict market value based on 1,943 analyzed devices")
st.write("---")
st.info("""
**Why this model is important:**  
1️⃣ The Brain: Stores patterns from 1,943 devices.  
2️⃣ Serialization (.pkl): Fast loading and cloud-ready.  
3️⃣ Prediction Engine: Converts specs into a market price instantly.
""")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("🔧 Technical Specs")
    ram = st.slider("RAM (GB)", 1, 64, 8)
    battery = st.slider("Battery (mAh)", 1000, 7000, 4500)
    camera = st.slider("Main Camera (MP)", 2, 200, 50)
    weight = st.number_input("Weight (grams)", 100, 500, 190)
    
    brand = st.selectbox("Brand", sorted(data['brand'].dropna().unique()))
    os_choice = st.selectbox("Operating System", sorted(data['OS'].dropna().unique()))
    chipset = st.selectbox("Chipset", sorted(data['Chipset'].dropna().unique()))
    
    predict_btn = st.button("Calculate Market Value")

with col2:
    st.subheader("📊 Market Insights")
    top_brands = data['brand'].value_counts().head(10).reset_index()
    top_brands.columns = ['Brand','Count']
    chart = alt.Chart(top_brands).mark_bar(color="#4CAF50").encode(
        x=alt.X('Brand', sort='-y'),
        y='Count',
        tooltip=['Brand','Count']
    ).interactive().properties(width=500, height=400)
    st.altair_chart(chart, use_container_width=True)

result_placeholder = st.empty()

# ================== 6️⃣ Prediction + أقرب صورة ==================
if predict_btn:
    # تجهيز البيانات للـ prediction
    input_dict = {
        'RAM_GB': ram,
        'battery_mAh': battery,
        'primary_camera_MP': camera,
        'weight_g': weight,
        'brand': safe_label_encoder('brand', brand),
        'OS': safe_label_encoder('OS', os_choice),
        'Chipset': safe_label_encoder('Chipset', chipset)
    }
    
    input_data = pd.DataFrame([{f: input_dict[f] for f in features}])
    prediction = model.predict(input_data)[0]
    
    # عمودين: سعر + صورة
    col_price, col_image = st.columns([1,1])
    
    with col_price:
        st.success(f"### Estimated Value: €{prediction:,.2f}")
        st.code(f"{prediction:.2f} €", language='text')  # copy price
        st.info("Price based on 2026 market trends learned by the AI.")
    
    with col_image:
        # ===== أقرب صورة ممكنة =====
        subset = data[
            (data['brand'] == brand) &
            (data['OS'] == os_choice) &
            (data['Chipset'] == chipset) &
            data['img_url'].notna()
        ]
        if not subset.empty:
            subset['distance'] = (
                abs(subset['RAM_GB'] - ram) +
                abs(subset['battery_mAh'] - battery)/1000 +
                abs(subset['primary_camera_MP'] - camera)/10 +
                abs(subset['weight_g'] - weight)/50
            )
            best_match = subset.loc[subset['distance'].idxmin(), 'img_url']
            st.image(best_match, use_column_width=True)
        else:
            st.image("https://via.placeholder.com/250x400.png?text=No+Image", use_column_width=True)

# ================== 7️⃣ Footer ==================
st.write("---")
st.markdown("""
Developed by **Goda Emad** |  
[GitHub](https://github.com/Goda-Emad) | [LinkedIn](https://www.linkedin.com/in/goda-emad/) | 2026 AI Portfolio
""")
