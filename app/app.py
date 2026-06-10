import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_option_menu import option_menu
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Sistem Analisis Penyakit Jantung",
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
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 50%, #FF8E8E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        animation: fadeIn 1s ease-in;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2C3E50;
        margin-top: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #FF4B4B;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        animation: slideIn 0.8s ease-out;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        border-color: #667eea;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #f5576c 0%, #ff6b6b 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        animation: pulse 2s infinite;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.8rem 2.5rem;
        border-radius: 30px;
        border: none;
        transition: all 0.3s ease;
        font-size: 1.1rem;
        letter-spacing: 1px;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5);
    }
    
    .glass-effect {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-top: 3rem;
    }
    
    .team-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #667eea30;
        transition: all 0.3s ease;
    }
    
    .team-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border-color: #667eea;
    }
    
    div[data-testid="stExpander"] {
        border: 2px solid #667eea20;
        border-radius: 15px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)
# ============================================
# KONFIGURASI PATH - AUTO DETECT (PASTI BERHASIL)
# ============================================
import os
from pathlib import Path
import streamlit as st

# Fungsi untuk mencari file heart.csv di seluruh struktur project
def find_heart_csv():
    """Mencari file heart.csv di berbagai kemungkinan lokasi"""
    
    # Daftar semua kemungkinan path yang akan dicoba
    possible_paths = []
    
    # 1. Path berdasarkan lokasi file script saat ini
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    
    # Coba berbagai level direktori
    for level in range(4):  # naik 0-3 level
        base = current_dir
        for _ in range(level):
            base = base.parent
        
        possible_paths.append(base / "dataset" / "heart.csv")
        possible_paths.append(base / "data" / "heart.csv")
        possible_paths.append(base / "heart.csv")
        possible_paths.append(base / "app" / "dataset" / "heart.csv")
    
    # 2. Path absolut yang umum di Streamlit Cloud
    possible_paths.extend([
        Path("/mount/src/uas_datamining_kell3/dataset/heart.csv"),
        Path("/mount/src/UAS_DataMining_Kel13/dataset/heart.csv"),
        Path("/app/dataset/heart.csv"),
        Path("/home/appuser/dataset/heart.csv"),
    ])
    
    # 3. Path dari current working directory
    cwd = Path.cwd()
    possible_paths.extend([
        cwd / "dataset" / "heart.csv",
        cwd / "heart.csv",
        cwd / "app" / "dataset" / "heart.csv",
    ])
    
    # 4. Cari semua file heart.csv di seluruh project (paling lambat tapi paling akurat)
    try:
        for root, dirs, files in os.walk(cwd):
            if "heart.csv" in files:
                possible_paths.append(Path(root) / "heart.csv")
                break  # cukup satu saja
    except:
        pass
    
    # Coba semua path
    for path in possible_paths:
        try:
            if path.exists():
                return path
        except:
            continue
    
    return None

# Cari file dataset
DATA_PATH = find_heart_csv()

# Tampilkan debug info (HAPUS BARIS INI SETELAH BERHASIL)
st.sidebar.markdown("### 🔍 Debug Info")
st.sidebar.markdown(f"**Current working directory:** `{Path.cwd()}`")
st.sidebar.markdown(f"**__file__ location:** `{Path(__file__).resolve()}`")
st.sidebar.markdown(f"**DATA_PATH found:** `{DATA_PATH if DATA_PATH else 'NOT FOUND'}`")

if DATA_PATH is None:
    st.error("""
    ❌ **CRITICAL ERROR: File heart.csv tidak ditemukan!**
    
    ### Langkah yang harus dilakukan:
    
    1️⃣ **Upload file heart.csv langsung ke root folder**
       - Buka repository GitHub Anda
       - Upload heart.csv ke folder utama (bukan di dalam app/)
    
    2️⃣ **Atau gunakan URL dataset eksternal** (tambahkan kode di bawah)
    
    3️⃣ **Periksa nama file** - pastikan namanya tepat `heart.csv` (case sensitive)
    
    ### Sementara, saya akan menggunakan dataset dari URL:
    """)
    
    # FALLBACK TERAKHIR: Gunakan URL langsung
    import urllib.request
    try:
        # Download dataset dari URL public
        url = "https://raw.githubusercontent.com/datasets/heart-disease/main/data/cleveland.csv"
        os.makedirs("dataset", exist_ok=True)
        urllib.request.urlretrieve(url, "dataset/heart.csv")
        DATA_PATH = Path("dataset/heart.csv")
        st.success("✅ Berhasil mendownload dataset dari URL sebagai fallback!")
    except Exception as e:
        st.error(f"Gagal download dataset: {e}")
        st.stop()

# Path untuk model
BASE_DIR = DATA_PATH.parent.parent if DATA_PATH.parent.name == "dataset" else DATA_PATH.parent
MODEL_PATH = BASE_DIR / "model" / "heart_model.pkl"
KMEANS_PATH = BASE_DIR / "model" / "kmeans_model.pkl"

# Buat folder model jika belum ada
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

st.sidebar.success(f"✅ Dataset ditemukan di: `{DATA_PATH}`")
# ============================================
# KONSTANTA
# ============================================
FITUR = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
         "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

DESKRIPSI_FITUR = {
    "age": "Usia pasien (tahun)",
    "sex": "Jenis kelamin (1: Laki-laki, 0: Perempuan)",
    "cp": "Tipe nyeri dada (0-3)",
    "trestbps": "Tekanan darah istirahat (mm Hg)",
    "chol": "Kolesterol serum (mg/dl)",
    "fbs": "Gula darah puasa > 120 mg/dl (1: Ya, 0: Tidak)",
    "restecg": "Hasil EKG istirahat (0-2)",
    "thalach": "Denyut jantung maksimum",
    "exang": "Angina akibat olahraga (1: Ya, 0: Tidak)",
    "oldpeak": "ST depression akibat olahraga",
    "slope": "Kemiringan segmen ST (0-2)",
    "ca": "Jumlah pembuluh besar (0-4)",
    "thal": "Thalassemia (0-3)"
}

TIPE_NYERI_DADA = {
    0: "Angina Tipikal",
    1: "Angina Atipikal", 
    2: "Nyeri Non-Anginal",
    3: "Asimtomatik"
}

# ============================================
# FUNGSI LOAD DATA - DENGAN ERROR HANDLING
# ============================================
@st.cache_data
def load_data():
    try:
        # Pastikan DATA_PATH tersedia
        if DATA_PATH is None:
            st.error("DATA_PATH is None!")
            return pd.DataFrame()
        
        # Cek apakah file benar-benar ada
        if not DATA_PATH.exists():
            st.error(f"File tidak ditemukan di: {DATA_PATH}")
            return pd.DataFrame()
        
        # Baca file CSV
        df = pd.read_csv(DATA_PATH)
        
        # Validasi: pastikan kolom 'target' ada
        if 'target' not in df.columns:
            st.warning("Kolom 'target' tidak ditemukan. Cek file CSV.")
            st.write("Kolom yang ada:", df.columns.tolist())
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        st.error(f"Error detail: {type(e).__name__}: {str(e)}")
        return pd.DataFrame()

# ============================================
# FORM INPUT PASIEN
# ============================================
def form_input_pasien():
    st.markdown('<p class="sub-header">📋 Data Klinis Pasien</p>', unsafe_allow_html=True)
    
    with st.expander("📝 Isi Data Pasien", expanded=True):
        tab1, tab2, tab3 = st.tabs(["👤 Demografi & Tanda Vital", "🫀 Indikator Jantung", "🔬 Hasil Lab"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                usia = st.slider("🎂 Usia (tahun)", 20, 90, 52)
                jenis_kelamin = st.radio("⚥ Jenis Kelamin", [0, 1], 
                              format_func=lambda x: "👩 Perempuan" if x == 0 else "👨 Laki-laki",
                              horizontal=True)
                nyeri_dada = st.selectbox("💔 Tipe Nyeri Dada", [0, 1, 2, 3], 
                                format_func=lambda x: f"{x} - {TIPE_NYERI_DADA[x]}")
            with col2:
                tekanan_darah = st.slider("🩸 Tekanan Darah Istirahat (mmHg)", 80, 220, 125)
                kolesterol = st.slider("🧪 Kolesterol Serum (mg/dl)", 100, 600, 212)
                gula_darah = st.radio("🩸 Gula Darah Puasa > 120 mg/dl", [0, 1],
                             format_func=lambda x: "✅ Normal" if x == 0 else "⚠️ Tinggi",
                             horizontal=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                ekg = st.selectbox("📊 Hasil EKG Istirahat", [0, 1, 2],
                                 format_func=lambda x: ["Normal", "Kelainan ST-T", "Hipertrofi Ventrikel Kiri"][x])
                detak_jantung = st.slider("💓 Detak Jantung Maksimum", 60, 230, 168)
                angina = st.radio("🏃 Angina Akibat Olahraga", [0, 1],
                               format_func=lambda x: "✅ Tidak" if x == 0 else "⚠️ Ya",
                               horizontal=True)
            with col2:
                st_depresi = st.slider("📉 Depresi ST Akibat Olahraga", 0.0, 7.0, 1.0, 0.1)
                kemiringan_st = st.selectbox("📈 Kemiringan Segmen ST", [0, 1, 2],
                                   format_func=lambda x: ["Menanjak", "Datar", "Menurun"][x])
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                pembuluh = st.selectbox("🔬 Jumlah Pembuluh Darah Besar", [0, 1, 2, 3, 4])
            with col2:
                thalassemia = st.selectbox("🧬 Thalassemia", [0, 1, 2, 3],
                                  format_func=lambda x: ["Normal", "Cacat Tetap", "Cacat Reversibel", "Tidak Diketahui"][x])
    
    return pd.DataFrame([[usia, jenis_kelamin, nyeri_dada, tekanan_darah, kolesterol, gula_darah, ekg, 
                         detak_jantung, angina, st_depresi, kemiringan_st, pembuluh, thalassemia]], 
                       columns=FITUR)

# ============================================
# HALAMAN: BERANDA
# ============================================
def halaman_beranda():
    st.markdown('<p class="main-header">❤️ Sistem Analisis Penyakit Jantung</p>', unsafe_allow_html=True)
    
    # Informasi Proyek
    st.markdown("""
    <div class='card'>
        <h2 style='margin-bottom: 1rem;'>🎓 Proyek UAS Data Mining</h2>
        <p style='font-size: 1.1rem; line-height: 1.8;'>
            Aplikasi ini merupakan implementasi proyek UAS mata kuliah Data Mining yang menerapkan 
            metodologi <strong>CRISP-DM</strong> untuk analisis dan prediksi penyakit jantung 
            menggunakan berbagai algoritma Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Anggota Kelompok
    st.markdown("### 👥 Anggota Kelompok")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='team-card'>
            <h3 style='color: #667eea;'>👩‍🎓 Vania Setyorini</h3>
            <p style='font-size: 1.2rem; font-weight: 600; color: #2C3E50;'>NIM: 24051214064</p>
            <p style='color: #7f8c8d;'>Mahasiswa Data Mining</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='team-card'>
            <h3 style='color: #764ba2;'>👩‍🎓 Rozalinda Titalia Putri</h3>
            <p style='font-size: 1.2rem; font-weight: 600; color: #2C3E50;'>NIM: 24051214069</p>
            <p style='color: #7f8c8d;'>Mahasiswa Data Mining</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Fitur Utama
    st.markdown("### 🚀 Fitur Utama Aplikasi")
    
    fitur_col1, fitur_col2, fitur_col3 = st.columns(3)
    
    with fitur_col1:
        st.markdown("""
        <div class='metric-card'>
            <span style='font-size: 2.5rem;'>🤖</span>
            <h4>Model Ensemble</h4>
            <p>Menggabungkan 4 algoritma ML untuk akurasi optimal</p>
            <small style='color: #667eea;'>Regresi Logistik | Random Forest | GBM | SVM</small>
        </div>
        """, unsafe_allow_html=True)
    
    with fitur_col2:
        st.markdown("""
        <div class='metric-card'>
            <span style='font-size: 2.5rem;'>🧠</span>
            <h4>Explainable AI</h4>
            <p>Analisis SHAP & LIME untuk transparansi prediksi</p>
            <small style='color: #764ba2;'>Memahami faktor risiko utama</small>
        </div>
        """, unsafe_allow_html=True)
    
    with fitur_col3:
        st.markdown("""
        <div class='metric-card'>
            <span style='font-size: 2.5rem;'>📊</span>
            <h4>Dashboard Interaktif</h4>
            <p>Visualisasi data dan performa model real-time</p>
            <small style='color: #f5576c;'>Analisis komprehensif</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Metodologi
    st.markdown("### 📋 Metodologi CRISP-DM")
    
    metod_col1, metod_col2, metod_col3, metod_col4, metod_col5, metod_col6 = st.columns(6)
    
    tahapan = [
        ("📋", "Pemahaman\nBisnis"),
        ("📊", "Pemahaman\nData"),
        ("🔧", "Persiapan\nData"),
        ("🤖", "Pemodelan"),
        ("📈", "Evaluasi"),
        ("🚀", "Deployment")
    ]
    
    for col, (icon, tahap) in zip([metod_col1, metod_col2, metod_col3, metod_col4, metod_col5, metod_col6], tahapan):
        with col:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem;'>
                <div style='font-size: 2rem;'>{icon}</div>
                <p style='font-weight: 600; font-size: 0.8rem; color: #2C3E50; white-space: pre-line;'>{tahap}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# HALAMAN: DATASET
# ============================================
def halaman_dataset():
    df = load_data()
    if df.empty:
        st.error("Dataset tidak ditemukan!")
        return
    
    st.markdown('<p class="main-header">📊 Analisis Dataset</p>', unsafe_allow_html=True)
    
    # Metrik Dataset
    col1, col2, col3, col4, col5 = st.columns(5)
    data_metrik = [
        ("📈", "Total Data", df.shape[0], "#667eea"),
        ("🔢", "Fitur", df.shape[1] - 1, "#764ba2"),
        ("✅", "Data Lengkap", df.shape[0] - int(df.isna().sum().sum()), "#4facfe"),
        ("🔄", "Duplikasi", int(df.duplicated().sum()), "#f093fb"),
        ("🎯", "Kelas Target", df['target'].nunique(), "#f5576c")
    ]
    
    for col, (icon, label, value, color) in zip([col1, col2, col3, col4, col5], data_metrik):
        with col:
            st.markdown(f"""
                <div class='metric-card' style='border-left: 4px solid {color};'>
                    <div style='font-size: 2.5rem;'>{icon}</div>
                    <h3 style='margin: 0.5rem 0; color: #2C3E50; font-size: 1.8rem;'>{value}</h3>
                    <p style='color: #7f8c8d; margin: 0; font-weight: 600;'>{label}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Eksplorasi Data", "📊 Statistik", "🔍 Distribusi", "🧬 Korelasi"])
    
    with tab1:
        st.subheader("Eksplorasi Data Interaktif")
        col1, col2 = st.columns([1, 3])
        with col1:
            filter_target = st.multiselect("Target", options=[0, 1], default=[0, 1],
                                          format_func=lambda x: "Tidak Sakit" if x == 0 else "Sakit Jantung")
            range_usia = st.slider("Rentang Usia", 20, 90, (20, 90))
        with col2:
            df_filtered = df[(df['target'].isin(filter_target)) & 
                           (df['age'].between(range_usia[0], range_usia[1]))]
            st.dataframe(df_filtered.head(20), use_container_width=True)
            st.caption(f"Menampilkan {len(df_filtered)} data terfilter")
        
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Unduh Data Terfilter",
            data=csv,
            file_name=f"data_jantung_filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Analisis Statistik")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Statistik Deskriptif")
            st.dataframe(df.describe(), use_container_width=True)
        with col2:
            st.markdown("#### Distribusi Target")
            target_dist = df['target'].value_counts()
            fig = make_subplots(rows=1, cols=2, specs=[[{'type':'pie'}, {'type':'bar'}]])
            fig.add_trace(go.Pie(labels=['Tidak Sakit', 'Sakit Jantung'], values=target_dist.values,
                                hole=0.4, marker_colors=['#4facfe', '#f5576c']), row=1, col=1)
            fig.add_trace(go.Bar(x=['Tidak Sakit', 'Sakit Jantung'], y=target_dist.values,
                                marker_color=['#4facfe', '#f5576c'],
                                text=target_dist.values, textposition='auto'), row=1, col=2)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Analisis Distribusi Fitur")
        fitur_plot = st.selectbox("Pilih Fitur", FITUR)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=fitur_plot, color='target', marginal='box', barmode='overlay',
                             color_discrete_map={0: '#4facfe', 1: '#f5576c'},
                             title=f'Distribusi {fitur_plot} berdasarkan Target')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, y=fitur_plot, x='target', color='target',
                        color_discrete_map={0: '#4facfe', 1: '#f5576c'},
                        title=f'Box Plot {fitur_plot}')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Analisis Korelasi & Feature Engineering")
        col1, col2 = st.columns(2)
        with col1:
            corr_matrix = df.corr()
            fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                          color_continuous_scale='RdBu_r', title='Matriks Korelasi Fitur')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            X = df[FITUR]
            y = df['target']
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importance_df = pd.DataFrame({
                'fitur': FITUR,
                'tingkat_kepentingan': rf.feature_importances_
            }).sort_values('tingkat_kepentingan', ascending=True)
            fig = px.bar(importance_df.tail(10), x='tingkat_kepentingan', y='fitur',
                        orientation='h', title='10 Fitur Terpenting')
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# HALAMAN: PREDIKSI
# ============================================
def halaman_prediksi():
    st.markdown('<p class="main-header">🔮 Prediksi Penyakit Jantung</p>', unsafe_allow_html=True)
    
    result = train_advanced_models()
    if result[0] is None:
        st.error("Gagal melatih model. Periksa dataset.")
        return
    
    advanced_models, scaler, X_test, y_test, X_train, y_train = result
    user_df = form_input_pasien()
    
    if st.button("🔍 Jalankan Analisis", use_container_width=True):
        with st.spinner("🔄 Menjalankan pipeline AI..."):
            user_scaled = scaler.transform(user_df)
            
            prediksi = {}
            probabilitas = {}
            for nama, model in advanced_models.items():
                prediksi[nama] = model.predict(user_scaled)[0]
                if hasattr(model, 'predict_proba'):
                    probabilitas[nama] = model.predict_proba(user_scaled)[0][1]
            
            st.markdown("---")
            
            ensemble_pred = prediksi['Ensemble']
            ensemble_prob = probabilitas['Ensemble']
            
            if ensemble_pred == 1:
                st.markdown(f"""
                    <div class='risk-high'>
                        <h2>⚠️ RISIKO TINGGI TERDETEKSI</h2>
                        <h3>Probabilitas Model Ensemble: {ensemble_prob:.1%}</h3>
                        <p>Beberapa model menunjukkan risiko jantung yang tinggi</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='risk-low'>
                        <h2>✅ RISIKO RENDAH</h2>
                        <h3>Probabilitas Model Ensemble: {ensemble_prob:.1%}</h3>
                        <p>Model menunjukkan profil jantung normal</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Perbandingan Model
            st.markdown("### 📊 Perbandingan Performa Model")
            data_perbandingan = []
            for nama in advanced_models.keys():
                if nama in probabilitas:
                    data_perbandingan.append({
                        'Model': nama,
                        'Probabilitas Risiko': probabilitas[nama],
                        'Prediksi': 'Risiko Tinggi' if prediksi[nama] == 1 else 'Risiko Rendah'
                    })
            
            df_comp = pd.DataFrame(data_perbandingan)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(df_comp, x='Model', y='Probabilitas Risiko',
                           color='Prediksi',
                           color_discrete_map={'Risiko Tinggi': '#f5576c', 'Risiko Rendah': '#4facfe'},
                           title='Perbandingan Prediksi Model')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                kesepakatan = sum(1 for p in prediksi.values() if p == ensemble_pred)
                total = len(prediksi)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=kesepakatan,
                    title={'text': f"Kesepakatan Model ({kesepakatan}/{total})"},
                    gauge={'axis': {'range': [0, total]}, 'bar': {'color': "#667eea"}}
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Profil Pasien
            st.markdown("### 👤 Profil Klinis Pasien")
            profil_col1, profil_col2, profil_col3 = st.columns(3)
            with profil_col1:
                st.metric("Usia", f"{user_df['age'].values[0]} tahun")
                st.metric("Tekanan Darah", f"{user_df['trestbps'].values[0]} mmHg")
                st.metric("Detak Jantung Maks", f"{user_df['thalach'].values[0]} bpm")
            with profil_col2:
                st.metric("Kolesterol", f"{user_df['chol'].values[0]} mg/dl")
                st.metric("Gula Darah", "Tinggi" if user_df['fbs'].values[0] == 1 else "Normal")
                st.metric("Depresi ST", f"{user_df['oldpeak'].values[0]}")
            with profil_col3:
                st.metric("Nyeri Dada", TIPE_NYERI_DADA[user_df['cp'].values[0]])
                st.metric("Angina Olahraga", "Ya" if user_df['exang'].values[0] == 1 else "Tidak")
                st.metric("Pembuluh Besar", f"{user_df['ca'].values[0]}")
            
            # Metrik Model
            st.markdown("### 📈 Metrik Performa Model")
            ensemble = advanced_models['Ensemble']
            y_pred = ensemble.predict(X_test)
            
            col1, col2 = st.columns(2)
            with col1:
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True, aspect='auto',
                              x=['Tidak Sakit', 'Sakit Jantung'], y=['Tidak Sakit', 'Sakit Jantung'],
                              color_continuous_scale='RdBu_r',
                              title='Confusion Matrix - Model Ensemble')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                       name=f'ROC (AUC = {roc_auc:.3f})'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                       name='Acak', line=dict(dash='dash')))
                fig.update_layout(title='Kurva ROC - Model Ensemble',
                                xaxis_title='False Positive Rate',
                                yaxis_title='True Positive Rate', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.warning("""
            ⚠️ **Disclaimer:** Aplikasi ini hanya untuk tujuan pembelajaran Data Mining. 
            Hasil prediksi tidak dapat dijadikan sebagai diagnosis medis. 
            Silakan konsultasikan dengan tenaga medis profesional.
            """)

# ============================================
# HALAMAN: EXPLAINABLE AI
# ============================================
def halaman_xai():
    st.markdown('<p class="main-header">🧠 Explainable AI (XAI)</p>', unsafe_allow_html=True)
    
    explainer, shap_values, X_test, rf_model = get_shap_data()
    
    st.markdown("""
    <div class='card'>
        <h3>🔍 Memahami Keputusan AI</h3>
        <p>Jelajahi bagaimana model AI membuat prediksi menggunakan teknik SHAP dan LIME. 
        Analisis ini membantu memahami faktor-faktor yang paling mempengaruhi prediksi.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📊 Tingkat Kepentingan Fitur", "🎯 Analisis LIME"])
    
    with tab1:
        st.subheader("Tingkat Kepentingan Fitur Global (SHAP)")
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Fitur': FITUR,
            'Rata-rata |SHAP|': mean_abs_shap
        }).sort_values('Rata-rata |SHAP|', ascending=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(importance_df.tail(10), x='Rata-rata |SHAP|', y='Fitur',
                        orientation='h', title='10 Fitur Terpenting (SHAP)',
                        color='Rata-rata |SHAP|', color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            rf_importance = pd.DataFrame({
                'Fitur': FITUR,
                'Tingkat Kepentingan': rf_model.feature_importances_
            }).sort_values('Tingkat Kepentingan', ascending=True)
            
            fig = px.bar(rf_importance.tail(10), x='Tingkat Kepentingan', y='Fitur',
                        orientation='h', title='10 Fitur Terpenting (Random Forest)',
                        color='Tingkat Kepentingan', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap SHAP
        st.subheader("Heatmap Nilai SHAP (Sampel)")
        n_samples = min(30, shap_values.shape[0])
        sample_idx = np.random.choice(shap_values.shape[0], n_samples, replace=False)
        
        fig = px.imshow(
            shap_values[sample_idx].T,
            labels=dict(x="Indeks Sampel", y="Fitur", color="Nilai SHAP"),
            y=FITUR,
            title=f" Heatmap Nilai SHAP ({n_samples} sampel)",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Penjelasan Prediksi Individu (LIME)")
        
        instance_idx = st.slider("Pilih Pasien", 0, len(X_test)-1, 0)
        
        try:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_test.values,
                feature_names=FITUR,
                class_names=['Tidak Sakit', 'Sakit Jantung'],
                mode='classification',
                discretize_continuous=True,
                random_state=42
            )
            
            exp = lime_explainer.explain_instance(
                X_test.iloc[instance_idx].values,
                rf_model.predict_proba,
                num_features=10
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Kontribusi Fitur (LIME)")
                try:
                    fig = exp.as_pyplot_figure()
                    st.pyplot(fig)
                    plt.close()
                except:
                    st.markdown("**Kontribusi Fitur:**")
                    for fitur, bobot in exp.as_list():
                        warna = "#f5576c" if bobot > 0 else "#4facfe"
                        st.markdown(f"""
                        <div style='background: {warna}10; padding: 0.5rem; border-radius: 5px; 
                                  margin: 0.3rem 0; border-left: 3px solid {warna};'>
                            <span style='font-weight: 600;'>{fitur}</span>: {bobot:.4f}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                prediksi = rf_model.predict(X_test.iloc[instance_idx:instance_idx+1])[0]
                probabilitas = rf_model.predict_proba(X_test.iloc[instance_idx:instance_idx+1])[0]
                
                st.markdown(f"""
                <div class='{"risk-high" if prediksi == 1 else "risk-low"}'>
                    <h3>Pasien #{instance_idx}</h3>
                    <h4>Prediksi: {'RISIKO TINGGI' if prediksi == 1 else 'RISIKO RENDAH'}</h4>
                    <p>Probabilitas Sakit Jantung: {probabilitas[1]:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Nilai Klinis Pasien")
                fitur_pasien = X_test.iloc[instance_idx]
                df_fitur = pd.DataFrame({
                    'Fitur': FITUR,
                    'Nilai': fitur_pasien.values
                })
                st.dataframe(df_fitur, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error LIME: {e}")
            st.info("Menampilkan prediksi tanpa penjelasan LIME")
            prediksi = rf_model.predict(X_test.iloc[instance_idx:instance_idx+1])[0]
            probabilitas = rf_model.predict_proba(X_test.iloc[instance_idx:instance_idx+1])[0]
            
            st.markdown(f"""
            <div class='{"risk-high" if prediksi == 1 else "risk-low"}'>
                <h3>Pasien #{instance_idx}</h3>
                <h4>Prediksi: {'RISIKO TINGGI' if prediksi == 1 else 'RISIKO RENDAH'}</h4>
                <p>Probabilitas Sakit Jantung: {probabilitas[1]:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# HALAMAN: DASHBOARD
# ============================================
def halaman_dashboard():
    st.markdown('<p class="main-header">📊 Dashboard Analitik</p>', unsafe_allow_html=True)
    
    df = load_data()
    if df.empty:
        st.error("Dataset tidak ditemukan!")
        return
    
    # Dashboard Performa Model
    st.markdown("### 🤖 Dashboard Performa Model")
    
    X = df[FITUR]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Regresi Logistik': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
    }
    
    data_metrik = []
    for nama, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        data_metrik.append({
            'Model': nama,
            'Rata-rata CV': cv_scores.mean(),
            'Std CV': cv_scores.std(),
            'Akurasi Test': accuracy_score(y_test, y_pred)
        })
    
    df_metrik = pd.DataFrame(data_metrik)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.bar(df_metrik, x='Model', y='Rata-rata CV', error_y='Std CV',
                    color='Model', title='Performa Cross-Validation 5-Fold')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(df_metrik, x='Model', y='Akurasi Test',
                    color='Model', title='Akurasi Data Test')
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        st.dataframe(df_metrik.style.highlight_max(subset=['Rata-rata CV', 'Akurasi Test']), 
                    use_container_width=True)
    
    # Analitik Real-time
    st.markdown("### 🔄 Analitik Data Real-time")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Risiko Berdasarkan Kelompok Usia")
        df['kelompok_usia'] = pd.cut(df['age'], bins=[20, 40, 50, 60, 90], 
                                     labels=['20-40', '40-50', '50-60', '60+'])
        risiko_per_usia = df.groupby('kelompok_usia', observed=False)['target'].mean() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=risiko_per_usia.index, y=risiko_per_usia.values,
                                mode='lines+markers',
                                line=dict(color='#667eea', width=3),
                                marker=dict(size=12)))
        fig.update_layout(title='Persentase Risiko per Kelompok Usia', 
                         yaxis_title='Persentase Risiko (%)', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("#### Kolesterol vs Detak Jantung Maksimum")
        fig = px.scatter(df, x='chol', y='thalach', color='target',
                        size='age', hover_data=['cp', 'oldpeak'],
                        color_discrete_map={0: '#4facfe', 1: '#f5576c'},
                        title='Kolesterol vs Detak Jantung (Ukuran: Usia)')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# HALAMAN: TENTANG
# ============================================
def halaman_tentang():
    st.markdown('<p class="main-header">ℹ️ Tentang Proyek</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h2>🎓 Proyek UAS Data Mining</h2>
            <p>Aplikasi ini dikembangkan sebagai proyek akhir mata kuliah Data Mining dengan 
            menerapkan metodologi CRISP-DM untuk analisis dan prediksi penyakit jantung.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 👥 Anggota Kelompok")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class='team-card'>
                <h3 style='color: #667eea;'>👩‍🎓 Vania Setyorini</h3>
                <p style='font-size: 1.1rem; font-weight: 600;'>NIM: 24051214064</p>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class='team-card'>
                <h3 style='color: #764ba2;'>👩‍🎓 Rozalinda Titalia Putri</h3>
                <p style='font-size: 1.1rem; font-weight: 600;'>NIM: 24051214069</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🚀 Teknologi yang Digunakan")
        
        tech_tab1, tech_tab2, tech_tab3 = st.tabs(["🤖 Model ML", "🧠 XAI", "📊 Visualisasi"])
        
        with tech_tab1:
            st.markdown("""
            #### Model Machine Learning:
            - **Regresi Logistik** - Klasifikasi linear dasar
            - **Random Forest** - Model ensemble berbasis pohon keputusan
            - **Gradient Boosting** - Pembelajaran ensemble sekuensial
            - **Support Vector Machine** - Klasifikasi berbasis kernel
            - **Voting Ensemble** - Kombinasi prediksi dari semua model
            
            #### Optimasi Model:
            - Cross-validation untuk evaluasi yang robust
            - Feature importance analysis
            - Perbandingan dan pemilihan model terbaik
            """)
        
        with tech_tab2:
            st.markdown("""
            #### Teknik Explainable AI:
            - **SHAP Analysis** - Tingkat kepentingan fitur global & lokal
            - **LIME Explanations** - Interpretasi tingkat individu
            - **Analisis Dampak Fitur** - Memahami keputusan model
            
            #### Manfaat:
            - Memahami keputusan model AI
            - Membangun kepercayaan pada prediksi
            - Identifikasi faktor risiko utama
            - Dukungan keputusan klinis
            """)
        
        with tech_tab3:
            st.markdown("""
            #### Tools Visualisasi:
            - **Streamlit** - Framework web interaktif
            - **Plotly** - Grafik interaktif
            - **Matplotlib** - Visualisasi statis
            - **Scikit-learn** - Pipeline machine learning
            - **SHAP & LIME** - Library explainable AI
            """)
    
    with col2:
        st.markdown("""
        <div class='glass-effect'>
            <h3 style='color: #2C3E50; text-align: center;'>📊 Informasi Dataset</h3>
            <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;'>
                <p><strong>Sumber:</strong> UCI Machine Learning Repository</p>
                <p><strong>Jumlah Data:</strong> 303 pasien</p>
                <p><strong>Jumlah Fitur:</strong> 13 atribut klinis</p>
                <p><strong>Target:</strong> Ada/tidaknya penyakit jantung</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📖 Deskripsi Fitur")
        df_deskripsi = pd.DataFrame({
            "Fitur": list(DESKRIPSI_FITUR.keys()),
            "Keterangan": list(DESKRIPSI_FITUR.values())
        })
        st.dataframe(df_deskripsi, use_container_width=True, hide_index=True)
        
        st.warning("""
        ⚠️ **Disclaimer Penting:** Aplikasi ini dikembangkan untuk tujuan edukasi sebagai 
        proyek UAS Data Mining. Prediksi dan analisis yang diberikan tidak dapat dijadikan 
        sebagai diagnosis medis. Selalu konsultasikan dengan tenaga medis profesional.
        """)

# ============================================
# APLIKASI UTAMA
# ============================================
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea, #764ba2); 
                       border-radius: 15px; color: white; margin-bottom: 1rem;'>
                <h1 style='font-size: 2rem; margin: 0;'>❤️</h1>
                <h2 style='margin: 0.5rem 0;'>HeartAI</h2>
                <p style='font-size: 0.8rem; opacity: 0.9;'>Proyek UAS Data Mining</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigasi
        menu = option_menu(
            menu_title=None,
            options=["Beranda", "Dataset", "Prediksi", "XAI Analysis", "Dashboard", "Tentang"],
            icons=["house", "database", "robot", "brain", "speedometer2", "info-circle"],
            default_index=0,
            styles={
                "container": {"padding": "0!important"},
                "icon": {"color": "#667eea", "font-size": "1.2rem"},
                "nav-link": {"font-size": "0.95rem", "text-align": "left", "margin": "0.3rem 0", "border-radius": "10px"},
                "nav-link-selected": {"background": "linear-gradient(135deg, #667eea, #764ba2)", "color": "white", "font-weight": "600"},
            }
        )
        
        st.markdown("---")
        st.markdown("### 🟢 Status Sistem")
        st.progress(0.95, text="Model: Dimuat 95%")
        st.progress(1.0, text="Database: Online")
        
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; font-size: 0.8rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;'>
                <p style='margin: 0; color: #7f8c8d;'>HeartAI v2.0</p>
                <p style='margin: 0; color: #7f8c8d;'>CRISP-DM Framework</p>
                <p style='margin: 0; color: #667eea;'>© 2024 Proyek UAS Data Mining</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Routing Halaman
    if menu == "Beranda":
        halaman_beranda()
    elif menu == "Dataset":
        halaman_dataset()
    elif menu == "Prediksi":
        halaman_prediksi()
    elif menu == "XAI Analysis":
        halaman_xai()
    elif menu == "Dashboard":
        halaman_dashboard()
    else:
        halaman_tentang()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div class='footer'>
            <h3>❤️ HeartAI - Sistem Prediksi Penyakit Jantung</h3>
            <p style='margin: 0; opacity: 0.9;'>Proyek UAS Data Mining</p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;'>
                👩‍🎓 Vania Setyorini (24051214064) & Rozalinda Titalia Putri (24051214069)
            </p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>
                ⚠️ Hanya untuk tujuan pembelajaran. Bukan alat diagnosis medis.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
