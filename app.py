import streamlit as st
import pandas as pd

# 1. إعدادات الصفحة (تظهر في تبويب المتصفح)
st.set_page_config(
    page_title="Mobile Price Predictor 2026",
    page_icon="📱",
    layout="wide"
)

# 2. دالة تحميل البيانات مع خاصية الـ Cache لتسريع الموقع
@st.cache_data
def load_data():
    # تأكد أن ملف الـ CSV مرفوع في نفس الفولدر على GitHub
    df = pd.read_csv('mobile_data_cleaned_2026.csv')
    # تنظيف أسماء الأعمدة من أي مسافات زائدة
    df.columns = df.columns.str.strip()
    return df

try:
    df = load_data()

    # --- الشريط الجانبي (Sidebar) ---
    st.sidebar.header("🔍 Search & Filter")
    
    # فلتر البحث بالاسم
    search_query = st.sidebar.text_input("Search Mobile Name", "")
    
    # فلتر البراند
    brands = ["All Brands"] + sorted(df['brand'].unique().tolist())
    selected_brand = st.sidebar.selectbox("Select Brand", brands)
    
    # --- منطق الفلترة ---
    filtered_df = df.copy()
    if selected_brand != "All Brands":
        filtered_df = filtered_df[filtered_df['brand'] == selected_brand]
    
    if search_query:
        filtered_df = filtered_df[filtered_df['model'].str.contains(search_query, case=False, na=False)]

    # --- الواجهة الرئيسية ---
    st.title("📱 Mobile Discovery Dashboard")
    st.markdown(f"Currently exploring **{len(filtered_df)}** devices")

    # عرض النتائج في شكل كروت (Grid)
    # سنعرض أول 40 نتيجة فقط لتحسين الأداء ومنع التهنيج
    display_limit = 40
    results_to_show = filtered_df.head(display_limit)

    if not results_to_show.empty:
        cols = st.columns(4) # تقسيم الشاشة لـ 4 أعمدة
        for i, (index, row) in enumerate(results_to_show.iterrows()):
            with cols[i % 4]:
                # عرض الصورة
                st.image(row['img_url'], use_container_width=True)
                # اسم الموبايل وسعره
                st.subheader(f"{row['brand']} {row['model']}")
                st.write(f"💰 **Price:** {row['approx_price_EUR']} EUR")
                # مواصفات إضافية في شكل كابشن
                st.caption(f"🔋 {row['battery_mAh']} mAh | 🧠 {row['RAM_GB']}GB RAM")
                st.divider()
    else:
        st.warning("No devices found matching your criteria. Try adjusting the filters!")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure 'mobile_data_cleaned_2026.csv' is uploaded to your GitHub repository.")

# --- تذييل الصفحة ---
st.markdown("---")
st.caption("Developed by Goda Emad | Data Science Project 2026")
