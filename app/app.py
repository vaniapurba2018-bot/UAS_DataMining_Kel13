import streamlit as st

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="CardioSense | Deteksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a4a 0%, #0d2137 100%);
        border-right: 2px solid #2196F3;
    }
    
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1a3a4a !important;
        text-align: center;
    }
    
    h2 {
        color: #1a3a4a !important;
        font-weight: 700 !important;
        border-left: 4px solid #2196F3;
        padding-left: 15px;
    }
    
    h3 {
        color: #1565C0 !important;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(33, 150, 243, 0.2);
    }
    .metric-value {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1565C0;
    }
    .metric-label {
        color: #666;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .info-box {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 16px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        color: #333;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1565C0, #2196F3);
        color: white;
        font-weight: 600;
        padding: 12px 35px;
        border-radius: 50px;
        border: none;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(33, 150, 243, 0.5);
    }
    
    .gradient-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #2196F3, transparent);
        margin: 20px 0;
    }
    
    .result-box-success {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 2px solid #4CAF50;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
    }
    .result-box-danger {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border: 2px solid #f44336;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #999;
        font-size: 0.8rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HERO SECTION
# ============================================
st.markdown("""
<div style="text-align: center; padding: 30px 0;">
    <h1>❤️ CardioSense</h1>
    <p style="font-size: 1.2rem; color: #555;">
        Sistem Deteksi Dini Risiko Penyakit Jantung<br>
        <span style="color: #1565C0; font-weight: 600;">K-Means Clustering</span> & 
        <span style="color: #2196F3; font-weight: 600;">XGBoost Classification</span>
    </p>
</div>
<div class="gradient-divider"></div>
""", unsafe_allow_html=True)

# ============================================
# METRIC CARDS
# ============================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">93%</div>
        <div class="metric-label">🎯 Akurasi Model</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">1,025</div>
        <div class="metric-label">📊 Total Data</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">13</div>
        <div class="metric-label">🔬 Fitur Klinis</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">2</div>
        <div class="metric-label">🧠 Metode AI</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# WELCOME SECTION
# ============================================
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("""
    <div class="info-box">
        <h3>🩺 Selamat Datang di CardioSense</h3>
        <p style="font-size: 1.05rem; color: #444;">
            Platform cerdas untuk mendeteksi risiko penyakit jantung secara dini 
            menggunakan teknologi <strong>Artificial Intelligence</strong> dan 
            <strong>Machine Learning</strong>.
        </p>
        <p style="color: #666;">
            Masukkan data klinis Anda, dan sistem akan menganalisis 
            risiko penyakit jantung dalam hitungan detik.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box" style="text-align: center;">
        <h3>👥 Tim Pengembang</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px;">👨‍💻</td>
                <td style="padding: 10px;"><strong>Nama Anggota 1</strong></td>
                <td style="padding: 10px; color: #999;">NIM 001</td>
            </tr>
            <tr>
                <td style="padding: 10px;">👩‍💻</td>
                <td style="padding: 10px;"><strong>Nama Anggota 2</strong></td>
                <td style="padding: 10px; color: #999;">NIM 002</td>
            </tr>
        </table>
        <p style="margin-top: 15px; color: #999; font-size: 0.8rem;">
            📚 Mata Kuliah Data Mining<br>
            🏫 Semester Genap 2025/2026
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# FEATURES
# ============================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("## ✨ Fitur Unggulan")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2.5rem;">🔬</div>
        <h4>Analisis Multi-Metode</h4>
        <p style="color: #666; font-size: 0.9rem;">
            Clustering + Classification untuk hasil akurat
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2.5rem;">📊</div>
        <h4>Visualisasi Interaktif</h4>
        <p style="color: #666; font-size: 0.9rem;">
            Grafik dinamis untuk eksplorasi data
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size: 2.5rem;">⚡</div>
        <h4>Prediksi Real-Time</h4>
        <p style="color: #666; font-size: 0.9rem;">
            Hasil instan dengan akurasi tinggi
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# HOW IT WORKS
# ============================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("## 🔄 Bagaimana Cara Kerjanya?")

col1, col2, col3, col4 = st.columns(4)

for i, (emoji, title, desc) in enumerate([
    ("1️⃣", "Input Data", "Masukkan 13 parameter klinis"),
    ("2️⃣", "Preprocessing", "Data dinormalisasi"),
    ("3️⃣", "AI Analysis", "XGBoost + K-Means bekerja"),
    ("4️⃣", "Hasil", "Prediksi + Rekomendasi")
]):
    with [col1, col2, col3, col4][i]:
        st.markdown(f"""
        <div style="text-align: center; background: #fff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <div style="font-size: 2.5rem;">{emoji}</div>
            <h4 style="color: #1a3a4a;">{title}</h4>
            <p style="color: #666; font-size: 0.85rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="gradient-divider"></div>
<div class="footer">
    <p>© 2026 CardioSense | Data Mining UAS Project</p>
    <p>Dibangun dengan ❤️ menggunakan Streamlit, Scikit-learn, dan XGBoost</p>
</div>
""", unsafe_allow_html=True)
