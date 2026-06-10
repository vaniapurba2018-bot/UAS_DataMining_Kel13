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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
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
    page_title="HeartWise - Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS PREMIUM
# ============================================
st.markdown("""
<style>
    /* Font & Global */
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap');
    
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #e9edf2 100%);
    }
    
    /* Header Utama */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #DC2626, #EF4444, #F87171);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1F2937;
        margin: 1.5rem 0 1rem 0;
        padding-left: 0.75rem;
        border-left: 5px solid #EF4444;
    }
    
    /* Card Premium */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
    }
    
    /* Metric Card */
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #f0f0f0;
        transition: all 0.2s;
    }
    
    .metric-card:hover {
        border-color: #EF4444;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.1);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F2937;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    
    /* Risk Indicator */
    .risk-high {
        background: linear-gradient(135deg, #DC2626, #EF4444);
        border-radius: 20px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        animation: pulse 1.5s infinite;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10B981, #059669);
        border-radius: 20px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        font-weight: 600;
        border-radius: 40px;
        padding: 0.6rem 1.5rem;
        border: none;
        transition: all 0.3s;
        width: 100%;
        font-size: 1rem;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1F2937 0%, #111827 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #F3F4F6;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        background: white;
        border-radius: 16px;
        font-size: 0.8rem;
        color: #6B7280;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #EF4444, transparent);
        margin: 1rem 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 40px;
        padding: 8px 24px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: #EF4444 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# KONFIGURASI PATH
# ============================================
DATA_PATH = None
possible_paths = [
    Path.cwd() / "dataset" / "heart.csv",
    Path(__file__).parent.parent / "dataset" / "heart.csv",
    Path("/mount/src/uas_datamining_kel13/dataset/heart.csv"),
]

for path in possible_paths:
    if path.exists():
        DATA_PATH = path
        break

if DATA_PATH is None or not DATA_PATH.exists():
    import urllib.request
    os.makedirs("dataset", exist_ok=True)
    url = "https://raw.githubusercontent.com/datasets/heart-disease/main/data/cleveland.csv"
    urllib.request.urlretrieve(url, "dataset/heart.csv")
    DATA_PATH = Path("dataset/heart.csv")

# ============================================
# KONSTANTA
# ============================================
FITUR = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
         "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

TIPE_NYERI_DADA = {
    0: "Angina Tipikal",
    1: "Angina Atipikal", 
    2: "Nyeri Non-Anginal",
    3: "Asimtomatik"
}

# ============================================
# FUNGSI LOAD DATA
# ============================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        
        if 'target' not in df.columns:
            column_names = FITUR + ['target']
            df = pd.read_csv(DATA_PATH, names=column_names, na_values='?')
            df = df.dropna()
            df['target'] = (df['target'] > 0).astype(int)
        
        if len(df.columns) > 14:
            df = df[FITUR + ['target']]
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# ============================================
# TRAIN ADVANCED MODELS
# ============================================
@st.cache_resource
def train_advanced_models():
    df = load_data()
    if df.empty:
        return None, None, None, None, None, None
    
    X = df[FITUR]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Regresi Logistik': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
    
    ensemble = VotingClassifier(estimators=[
        ('lr', trained_models['Regresi Logistik']),
        ('rf', trained_models['Random Forest']),
        ('gb', trained_models['Gradient Boosting']),
        ('svm', trained_models['SVM'])
    ], voting='soft')
    ensemble.fit(X_train_scaled, y_train)
    trained_models['Ensemble'] = ensemble
    
    return trained_models, scaler, X_test_scaled, y_test, X_train_scaled, y_train

# ============================================
# GET SHAP DATA
# ============================================
@st.cache_resource
def get_shap_data():
    df = load_data()
    if df.empty:
        return None, None, None, None
    
    X = df[FITUR]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    shap_values = np.zeros((len(X_test), len(FITUR)))
    for i, col in enumerate(FITUR):
        shap_values[:, i] = rf_model.feature_importances_[i] * (X_test[col].values - X_train[col].mean())
    
    return None, shap_values, X_test, rf_model

# ============================================
# FORM INPUT PASIEN (BEAUTIFUL)
# ============================================
def form_input_pasien():
    st.markdown('<div class="sub-header">📋 Data Klinis Pasien</div>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("#### 👤 Demografi")
            usia = st.slider("Usia (tahun)", 20, 90, 52, help="Usia pasien dalam tahun")
            jenis_kelamin = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "👩 Perempuan" if x == 0 else "👨 Laki-laki")
            nyeri_dada = st.selectbox("Tipe Nyeri Dada", [0, 1, 2, 3], format_func=lambda x: f"{x} - {TIPE_NYERI_DADA[x]}")
            tekanan_darah = st.slider("Tekanan Darah Istirahat (mmHg)", 80, 220, 125)
            kolesterol = st.slider("Kolesterol Serum (mg/dl)", 100, 600, 212)
        
        with col2:
            st.markdown("#### 🫀 Indikator Jantung")
            gula_darah = st.selectbox("Gula Darah Puasa > 120", [0, 1], format_func=lambda x: "✅ Normal" if x == 0 else "⚠️ Tinggi")
            ekg = st.selectbox("Hasil EKG Istirahat", [0, 1, 2], format_func=lambda x: ["Normal", "Kelainan ST-T", "Hipertrofi Ventrikel Kiri"][x])
            detak_jantung = st.slider("Detak Jantung Maksimum (bpm)", 60, 230, 168)
            angina = st.selectbox("Angina Akibat Olahraga", [0, 1], format_func=lambda x: "✅ Tidak" if x == 0 else "⚠️ Ya")
            st_depresi = st.slider("Depresi ST Akibat Olahraga", 0.0, 7.0, 1.0, 0.1)
            kemiringan_st = st.selectbox("Kemiringan Segmen ST", [0, 1, 2], format_func=lambda x: ["Menanjak", "Datar", "Menurun"][x])
            pembuluh = st.selectbox("Jumlah Pembuluh Darah Besar", [0, 1, 2, 3, 4])
            thalassemia = st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: ["Normal", "Cacat Tetap", "Cacat Reversibel", "Tidak Diketahui"][x])
    
    return pd.DataFrame([[usia, jenis_kelamin, nyeri_dada, tekanan_darah, kolesterol, gula_darah, ekg, 
                         detak_jantung, angina, st_depresi, kemiringan_st, pembuluh, thalassemia]], 
                       columns=FITUR)

# ============================================
# HALAMAN BERANDA
# ============================================
def halaman_beranda():
    st.markdown('<div class="main-header">❤️ HeartWise</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6B7280; margin-bottom: 2rem;">Sistem Prediksi Penyakit Jantung berbasis AI</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h2 style="color: #1F2937; margin-bottom: 1rem;">🎓 Proyek UAS Data Mining</h2>
            <p style="color: #4B5563; line-height: 1.6;">
                Aplikasi ini mengimplementasikan metodologi <strong>CRISP-DM</strong> untuk menganalisis 
                dan memprediksi penyakit jantung menggunakan berbagai algoritma Machine Learning.
            </p>
            <div class="divider"></div>
            <p style="color: #4B5563;">✨ <strong>Fitur Unggulan:</strong> Ensemble Learning (4 Algoritma), 
            Explainable AI, Dashboard Interaktif</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center;">
            <h3 style="color: #1F2937;">👥 Anggota</h3>
            <p style="margin: 0.5rem 0;"><strong>Vania Setyorini</strong><br>24051214064</p>
            <p style="margin: 0.5rem 0;"><strong>Rozalinda Titalia Putri</strong><br>24051214069</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistik Cepat
    st.markdown('<div class="sub-header">📊 Statistik Dataset</div>', unsafe_allow_html=True)
    
    df = load_data()
    if not df.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[0]}</div>
                <div class="metric-label">Total Pasien</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[1]-1}</div>
                <div class="metric-label">Fitur Klinis</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            sakit = (df['target'] == 1).sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{sakit}</div>
                <div class="metric-label">Pasien Sakit Jantung</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            sehat = (df['target'] == 0).sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{sehat}</div>
                <div class="metric-label">Pasien Sehat</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            persen = (sakit / df.shape[0]) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{persen:.1f}%</div>
                <div class="metric-label">Prevalensi</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# HALAMAN DATASET
# ============================================
def halaman_dataset():
    df = load_data()
    if df.empty:
        st.error("Dataset tidak ditemukan!")
        return
    
    st.markdown('<div class="main-header">📊 Eksplorasi Dataset</div>', unsafe_allow_html=True)
    
    # Preview
    tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📈 Statistik", "📊 Distribusi"])
    
    with tab1:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "heart_disease_data.csv", "text/csv")
    
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='age', color='target', nbins=30, 
                               title='Distribusi Usia', barmode='overlay',
                               color_discrete_map={0: '#10B981', 1: '#EF4444'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            target_counts = df['target'].value_counts()
            fig = px.pie(values=target_counts.values, names=['Sehat', 'Sakit Jantung'],
                        title='Proporsi Target', color_discrete_sequence=['#10B981', '#EF4444'])
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# HALAMAN PREDIKSI
# ============================================
def halaman_prediksi():
    st.markdown('<div class="main-header">🔮 Prediksi Penyakit Jantung</div>', unsafe_allow_html=True)
    
    result = train_advanced_models()
    if result[0] is None:
        st.error("Gagal melatih model. Periksa dataset.")
        return
    
    advanced_models, scaler, X_test, y_test, X_train, y_train = result
    
    with st.container():
        user_df = form_input_pasien()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Jalankan Analisis AI", use_container_width=True):
            with st.spinner("Memproses data dengan 4 algoritma AI..."):
                user_scaled = scaler.transform(user_df)
                
                ensemble_pred = advanced_models['Ensemble'].predict(user_scaled)[0]
                ensemble_prob = advanced_models['Ensemble'].predict_proba(user_scaled)[0][1]
                
                if ensemble_pred == 1:
                    st.markdown(f"""
                    <div class="risk-high">
                        <h2>⚠️ RISIKO TINGGI</h2>
                        <h3>Probabilitas: {ensemble_prob:.1%}</h3>
                        <p>Model AI mendeteksi indikasi penyakit jantung. Segera konsultasikan ke dokter.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-low">
                        <h2>✅ RISIKO RENDAH</h2>
                        <h3>Probabilitas: {ensemble_prob:.1%}</h3>
                        <p>Profil jantung Anda tergolong normal. Tetap jaga pola hidup sehat!</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detail Prediksi per Model
                st.markdown("#### 🤖 Prediksi per Model")
                prediksi_detail = []
                for name, model in advanced_models.items():
                    pred = model.predict(user_scaled)[0]
                    prob = model.predict_proba(user_scaled)[0][1] if hasattr(model, 'predict_proba') else 0
                    prediksi_detail.append({
                        'Model': name,
                        'Prediksi': '⚠️ Risiko Tinggi' if pred == 1 else '✅ Risiko Rendah',
                        'Probabilitas': f"{prob:.1%}"
                    })
                st.dataframe(pd.DataFrame(prediksi_detail), use_container_width=True, hide_index=True)

# ============================================
# HALAMAN XAI
# ============================================
def halaman_xai():
    st.markdown('<div class="main-header">🧠 Explainable AI</div>', unsafe_allow_html=True)
    
    explainer, shap_values, X_test, rf_model = get_shap_data()
    
    if shap_values is None:
        st.error("Gagal memuat data XAI")
        return
    
    importance_df = pd.DataFrame({
        'Fitur': FITUR,
        'Pentingnya': np.abs(shap_values).mean(axis=0)
    }).sort_values('Pentingnya', ascending=True)
    
    fig = px.bar(importance_df, x='Pentingnya', y='Fitur', orientation='h',
                 title='Top 10 Faktor Risiko Penyakit Jantung',
                 color='Pentingnya', color_continuous_scale='Reds',
                 labels={'Pentingnya': 'Tingkat Pengaruh', 'Fitur': 'Fitur Klinis'})
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Insight:** Fitur dengan nilai tertinggi memiliki pengaruh terbesar terhadap prediksi penyakit jantung. Fokus pada faktor-faktor ini untuk pencegahan.")

# ============================================
# HALAMAN DASHBOARD
# ============================================
def halaman_dashboard():
    st.markdown('<div class="main-header">📊 Dashboard Analitik</div>', unsafe_allow_html=True)
    
    df = load_data()
    if df.empty:
        st.error("Dataset tidak ditemukan!")
        return
    
    # Matriks Korelasi
    st.markdown("#### 🔗 Matriks Korelasi Fitur")
    corr = df[FITUR + ['target']].corr()
    fig = px.imshow(corr, text_auto='.2f', aspect='auto', 
                    color_continuous_scale='RdBu_r', 
                    title='Korelasi antar Fitur Klinis')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analisis Usia
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Risiko Berdasarkan Usia")
        df['kelompok_usia'] = pd.cut(df['age'], bins=[20, 40, 50, 60, 80], labels=['20-40', '40-50', '50-60', '60-80'])
        risiko_usia = df.groupby('kelompok_usia', observed=False)['target'].mean() * 100
        fig = px.bar(x=risiko_usia.index, y=risiko_usia.values, 
                     title='Persentase Risiko per Kelompok Usia',
                     color=risiko_usia.values, color_continuous_scale='Reds',
                     labels={'x': 'Kelompok Usia', 'y': 'Risiko (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🔬 Kolesterol vs Detak Jantung")
        fig = px.scatter(df, x='chol', y='thalach', color='target', 
                        size='age', hover_data=['cp', 'oldpeak'],
                        title='Hubungan Kolesterol & Detak Jantung',
                        labels={'chol': 'Kolesterol (mg/dl)', 'thalach': 'Detak Jantung Maks (bpm)'},
                        color_discrete_map={0: '#10B981', 1: '#EF4444'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# HALAMAN TENTANG
# ============================================
def halaman_tentang():
    st.markdown('<div class="main-header">ℹ️ Tentang HeartWise</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h2 style="color: #1F2937;">🎯 Misi Kami</h2>
        <p>Memberikan akses mudah untuk deteksi dini risiko penyakit jantung menggunakan teknologi AI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>🤖 Algoritma yang Digunakan</h3>
            <ul>
                <li><strong>Regresi Logistik</strong> - Baseline model</li>
                <li><strong>Random Forest</strong> - Ensemble pohon keputusan</li>
                <li><strong>Gradient Boosting</strong> - Boosting algorithm</li>
                <li><strong>SVM</strong> - Support Vector Machine</li>
                <li><strong>Voting Ensemble</strong> - Kombinasi ke-4 model</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>📊 Sumber Data</h3>
            <p><strong>Dataset:</strong> UCI Machine Learning Repository<br>
            <strong>Heart Disease Dataset</strong><br>
            <strong>Jumlah:</strong> 303 pasien<br>
            <strong>Fitur:</strong> 13 atribut klinis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.warning("""
    ⚠️ **Disclaimer Penting:** Aplikasi ini dikembangkan untuk tujuan pembelajaran 
    sebagai proyek UAS Data Mining. Hasil prediksi tidak dapat dijadikan sebagai 
    diagnosis medis profesional. Selalu konsultasikan dengan tenaga medis.
    """)

# ============================================
# MAIN
# ============================================
def main():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 2.5rem; margin: 0;">❤️</h1>
            <h2 style="margin: 0.5rem 0; color: white;">HeartWise</h2>
            <p style="color: #9CA3AF; font-size: 0.8rem;">Powered by AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        menu = option_menu(
            menu_title=None,
            options=["Beranda", "Dataset", "Prediksi", "XAI Analysis", "Dashboard", "Tentang"],
            icons=["house-fill", "table", "robot", "graph-up", "speedometer2", "info-circle"],
            default_index=0,
            styles={
                "nav-link": {
                    "font-size": "1rem", 
                    "margin": "0.3rem 0",
                    "border-radius": "12px",
                    "color": "#F3F4F6",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, #EF4444, #DC2626)",
                    "color": "white",
                    "font-weight": "600",
                },
            }
        )
        
        st.markdown("---")
        st.markdown("<p style='text-align: center; color: #6B7280; font-size: 0.7rem;'>© 2024 HeartWise<br>Proyek UAS Data Mining</p>", unsafe_allow_html=True)
    
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
    st.markdown("""
    <div class="footer">
        ❤️ HeartWise - Sistem Prediksi Penyakit Jantung | Proyek UAS Data Mining<br>
        Vania Setyorini (24051214064) & Rozalinda Titalia Putri (24051214069)
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
