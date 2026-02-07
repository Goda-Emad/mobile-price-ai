import streamlit as st
import pandas as pd
import joblib
import altair as alt
from sklearn.preprocessing import LabelEncoder

# 1️⃣ إعدادات الصفحة
st.set_page_config(page_title="Mobile Price AI", page_icon="📱", layout="wide")

# 2️⃣ CSS للواجهة والخطوط واللون الأبيض
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                    url("https://images.unsplash.com/photo-1616348436168-de43ad0db179?auto=format&fit=crop&q=80&w=2000");
        background-size: cover;
        background-position: center;
        color: white;
    }
    h1,h2,h3 { color:white !important; }
    .stSlider label, .stNumberInput label, .stSelectbox label { 
        color:white !important; font-weight:bold; 
    }
    .stButton>button {
        background-color: #4CAF50; color: white; border-radius:20px; width:100%; border:none; transition:0.3s;
    }
    .stButton>button:hover { background-color:#45a049; transform: scale(1.02); }
    </style>
""", unsafe_allow_html=True)

# 3️⃣ تحميل البيانات والموديل
@st.cache_data
def load_data():
    return pd.read_csv('mobile_data_cleaned_2026.csv')

@st.cache_resource
def load_model():
    return joblib.load('mobile_model.pkl')

try:
    data = load_data()
    model = load_model()
    features = model.feature_names_in_

    # LabelEncoders لكل categorical column
    le_brand = LabelEncoder().fit(data['brand'])
    le_os = LabelEncoder().fit(data['OS'])
    le_chipset = LabelEncoder().fit(data['Chipset'])

    # Debug Info للتأكد من Feature Names
    st.write("### Debug Info: Feature Names Check")
    st.write("Feature names in the model:", features)
    st.write("Columns in DataFrame:", data.columns.tolist())
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        st.warning("⚠️ These features are missing in the DataFrame: " + ", ".join(missing_features))
    else:
        st.success("✅ All feature names are present in the DataFrame.")

    # العنوان الرئيسي
    st.title("📱 AI Mobile Valuation Hub")
    st.markdown("### Predict market value based on 1,943 analyzed devices")
    st.write("---")

    # Info Box عن أهمية الموديل
    st.info("""
    **Why this model is important:**  
    1️⃣ **The Brain**: The model stores patterns from 1,943 devices, learning the rules connecting specs to price.  
    2️⃣ **Serialization (.pkl)**: Enables fast loading and easy transfer to cloud without retraining.  
    3️⃣ **Prediction Engine**: Converts user inputs like RAM, Battery, Camera, and Weight into a market price instantly.
    """)

    # تقسيم الشاشة لعمودين
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("🔧 Technical Specs")
        ram = st.slider("RAM (GB)", 1, 64, 8)
        battery = st.slider("Battery (mAh)", 1000, 7000, 4500)
        camera = st.slider("Main Camera (MP)", 2, 200, 50)
        weight = st.number_input("Weight (grams)", 100, 500, 190)

        # Dropdowns إضافية
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

    if predict_btn:
        # تحويل الـ categorical inputs للـ labels
        input_dict = {
            'RAM_GB': ram,
            'battery_mAh': battery,
            'primary_camera_MP': camera,
            'weight_g': weight,
            'brand': le_brand.transform([brand])[0],
            'OS': le_os.transform([os_choice])[0],
            'Chipset': le_chipset.transform([chipset])[0]
        }

        # ترتيب الأعمدة حسب ما اتدرب عليه الموديل
        input_data = pd.DataFrame([{f: input_dict[f] for f in features}])
        prediction = model.predict(input_data)[0]

        # إنشاء عمودين: واحد للسعر والتاني للصورة
        col_price, col_image = st.columns([1,1])

        with col_price:
            st.success(f"### Estimated Value: €{prediction:,.2f}")
            # زر نسخ السعر
            st.code(f"{prediction:.2f} €", language='text')
            st.info("Price based on 2026 market trends learned by the AI.")

        with col_image:
            # افتراضياً بنجيب أول صورة من الـ data matching specs (لو موجود)
            matching_img = data.loc[
                (data['RAM_GB']==ram) &
                (data['battery_mAh']==battery) &
                (data['primary_camera_MP']==camera) &
                (data['weight_g']==weight),
                'img_url'
            ]
            if not matching_img.empty:
                st.image(matching_img.values[0], use_column_width=True)
            else:
                st.image("https://via.placeholder.com/250x400.png?text=No+Image", use_column_width=True)

except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.info("Make sure 'mobile_model.pkl' and 'mobile_data_cleaned_2026.csv' are in the same folder.")

# Footer باسمك وروابط GitHub وLinkedIn
st.write("---")
st.markdown("""
Developed by **Goda Emad** |  
[GitHub](https://github.com/Goda-Emad) | [LinkedIn](https://www.linkedin.com/in/goda-emad/) | 2026 AI Portfolio
""")
