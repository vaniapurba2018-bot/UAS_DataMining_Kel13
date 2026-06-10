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
    layout="wide"
)

# ============================================
# CUSTOM CSS SEDERHANA TAPI MENARIK
# ============================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: #f8f9fa;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #DC2626;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1F2937;
        margin: 1rem 0 1rem 0;
        border-left: 4px solid #DC2626;
        padding-left: 1rem;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
    }
    
    .metric-box {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-top: 3px solid #DC2626;
    }
    
    .metric-number {
        font-size: 2rem;
        font-weight: 700;
        color: #DC2626;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
    }
    
    /* Risk indicators */
    .risk-high {
        background: #FEE2E2;
        border: 2px solid #DC2626;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .risk-high h2 {
        color: #DC2626;
        margin: 0;
    }
    
    .risk-low {
        background: #D1FAE5;
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .risk-low h2 {
        color: #059669;
        margin: 0;
    }
    
    /* Button */
    .stButton > button {
        background-color: #DC2626;
        color: white;
        font-weight: 600;
        border-radius: 30px;
        padding: 0.5rem 1.5rem;
        border: none;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #B91C1C;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        background: white;
        border-radius: 12px;
        font-size: 0.8rem;
        color: #6B7280;
        border: 1px solid #e5e7eb;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #1F2937;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #F3F4F6;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 30px;
        padding: 8px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #DC2626 !important;
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
# FORM INPUT PASIEN
# ============================================
def form_input_pasien():
    st.markdown('<div class="sub-header">📋 Data Klinis Pasien</div>', unsafe_allow_html=True)
    
    with st.expander("📝 Isi Data Pasien", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            usia = st.slider("🎂 Usia (tahun)", 20, 90, 52)
            jenis_kelamin = st.selectbox("⚥ Jenis Kelamin", [0, 1], format_func=lambda x: "👩 Perempuan" if x == 0 else "👨 Laki-laki")
            nyeri_dada = st.selectbox("💔 Tipe Nyeri Dada", [0, 1, 2, 3], format_func=lambda x: f"{x} - {TIPE_NYERI_DADA[x]}")
            tekanan_darah = st.slider("🩸 Tekanan Darah (mmHg)", 80, 220, 125)
            kolesterol = st.slider("🧪 Kolesterol (mg/dl)", 100, 600, 212)
        
        with col2:
            gula_darah = st.selectbox("🩸 Gula Darah Puasa >120", [0, 1], format_func=lambda x: "✅ Normal" if x == 0 else "⚠️ Tinggi")
            ekg = st.selectbox("📊 Hasil EKG", [0, 1, 2], format_func=lambda x: ["Normal", "Kelainan ST-T", "Hipertrofi"][x])
            detak_jantung = st.slider("💓 Detak Jantung Maks", 60, 230, 168)
            angina = st.selectbox("🏃 Angina Olahraga", [0, 1], format_func=lambda x: "✅ Tidak" if x == 0 else "⚠️ Ya")
            st_depresi = st.slider("📉 Depresi ST", 0.0, 7.0, 1.0, 0.1)
            kemiringan_st = st.selectbox("📈 Kemiringan ST", [0, 1, 2], format_func=lambda x: ["Menanjak", "Datar", "Menurun"][x])
            pembuluh = st.selectbox("🔬 Pembuluh Besar", [0, 1, 2, 3, 4])
            thalassemia = st.selectbox("🧬 Thalassemia", [0, 1, 2, 3], format_func=lambda x: ["Normal", "Cacat Tetap", "Cacat Reversibel", "Tidak Diketahui"][x])
    
    return pd.DataFrame([[usia, jenis_kelamin, nyeri_dada, tekanan_darah, kolesterol, gula_darah, ekg, 
                         detak_jantung, angina, st_depresi, kemiringan_st, pembuluh, thalassemia]], 
                       columns=FITUR)

# ============================================
# HALAMAN BERANDA
# ============================================
def halaman_beranda():
    st.markdown('<div class="main-header">❤️ HeartWise</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #6B7280; margin-bottom: 2rem;'>Sistem Prediksi Penyakit Jantung berbasis Machine Learning</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #1F2937;">🎓 Proyek UAS Data Mining</h3>
            <p style="color: #4B5563;">Aplikasi ini mengimplementasikan metodologi <strong>CRISP-DM</strong> 
            untuk menganalisis dan memprediksi penyakit jantung menggunakan 4 algoritma Machine Learning 
            yang digabung dalam <strong>Voting Ensemble</strong>.</p>
            <hr>
            <p><strong>✨ Fitur Unggulan:</strong> Ensemble Learning (4 Algoritma), Explainable AI, Dashboard Interaktif</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <h3 style="color: #1F2937;">👥 Anggota Kelompok</h3>
            <p><strong>Vania Setyorini</strong><br>24051214064</p>
            <p><strong>Rozalinda Titalia Putri</strong><br>24051214069</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Statistik
    st.markdown('<div class="sub-header">📊 Statistik Dataset</div>', unsafe_allow_html=True)
    
    df = load_data()
    if not df.empty:
        sakit = (df['target'] == 1).sum()
        sehat = (df['target'] == 0).sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number">{df.shape[0]}</div>
                <div class="metric-label">Total Pasien</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number">{df.shape[1]-1}</div>
                <div class="metric-label">Fitur Klinis</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number">{sakit}</div>
                <div class="metric-label">Sakit Jantung</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number">{sehat}</div>
                <div class="metric-label">Sehat</div>
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
    
    tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📈 Statistik", "📊 Visualisasi"])
    
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
                               title='Distribusi Usia Berdasarkan Target',
                               color_discrete_map={0: '#10B981', 1: '#DC2626'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            target_counts = df['target'].value_counts()
            fig = px.pie(values=target_counts.values, names=['Sehat', 'Sakit Jantung'],
                        title='Proporsi Target', color_discrete_sequence=['#10B981', '#DC2626'])
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
    user_df = form_input_pasien()
    
    if st.button("🔍 Jalankan Analisis", use_container_width=True):
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
            
            # Detail per model
            st.markdown("#### 🤖 Hasil dari 4 Model AI")
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
        'Tingkat Kepentingan': np.abs(shap_values).mean(axis=0)
    }).sort_values('Tingkat Kepentingan', ascending=True)
    
    fig = px.bar(importance_df, x='Tingkat Kepentingan', y='Fitur', orientation='h',
                 title='🔝 10 Faktor Risiko Utama Penyakit Jantung',
                 color='Tingkat Kepentingan', color_continuous_scale='Reds',
                 labels={'Tingkat Kepentingan': 'Pengaruh terhadap Prediksi', 'Fitur': 'Fitur Klinis'})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Insight:** Semakin panjang bar, semakin besar pengaruh fitur tersebut terhadap prediksi penyakit jantung. Fokus pada faktor-faktor ini untuk pencegahan.")

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
    st.markdown("#### 🔗 Matriks Korelasi Antar Fitur")
    corr = df[FITUR + ['target']].corr()
    fig = px.imshow(corr, text_auto='.2f', aspect='auto', 
                    color_continuous_scale='RdBu_r', 
                    title='Korelasi Antar Fitur Klinis')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analisis Usia
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Risiko Berdasarkan Kelompok Usia")
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
                        size='age', hover_data=['cp'],
                        title='Hubungan Kolesterol & Detak Jantung Maksimum',
                        labels={'chol': 'Kolesterol (mg/dl)', 'thalach': 'Detak Jantung Maks (bpm)'},
                        color_discrete_map={0: '#10B981', 1: '#DC2626'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# HALAMAN TENTANG
# ============================================
def halaman_tentang():
    st.markdown('<div class="main-header">ℹ️ Tentang HeartWise</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 Algoritma Machine Learning</h3>
            <ul>
                <li><strong>Regresi Logistik</strong> - Baseline model</li>
                <li><strong>Random Forest</strong> - Ensemble pohon keputusan</li>
                <li><strong>Gradient Boosting</strong> - Boosting algorithm</li>
                <li><strong>Support Vector Machine (SVM)</strong> - Klasifikasi kernel</li>
                <li><strong>Voting Ensemble</strong> - Kombinasi ke-4 model untuk akurasi terbaik</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Sumber Data</h3>
            <p><strong>Dataset:</strong> Heart Disease Dataset<br>
            <strong>Sumber:</strong> UCI Machine Learning Repository<br>
            <strong>Jumlah:</strong> 303 pasien<br>
            <strong>Fitur:</strong> 13 atribut klinis<br>
            <strong>Target:</strong> Ada/tidaknya penyakit jantung</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>👥 Anggota Kelompok</h3>
        <p><strong>Vania Setyorini</strong> (24051214064)<br>
        <strong>Rozalinda Titalia Putri</strong> (24051214069)</p>
        <hr>
        <p><strong>📚 Mata Kuliah:</strong> Data Mining<br>
        <strong>🎓 Semester:</strong> Genap 2024/2025<br>
        <strong>🏫 Universitas:</strong> PGRI Madiun</p>
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
        <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
            <h1 style="font-size: 3rem; margin: 0;">❤️</h1>
            <h2 style="margin: 0; color: white;">HeartWise</h2>
            <p style="color: #9CA3AF; font-size: 0.8rem;">Powered by AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        menu = option_menu(
            menu_title=None,
            options=["Beranda", "Dataset", "Prediksi", "XAI Analysis", "Dashboard", "Tentang"],
            icons=["house", "table", "robot", "brain", "speedometer2", "info-circle"],
            default_index=0,
            styles={
                "container": {"padding": "0!important"},
                "nav-link": {
                    "font-size": "1rem",
                    "margin": "0.3rem 0",
                    "border-radius": "10px",
                    "color": "#D1D5DB",
                },
                "nav-link-selected": {
                    "background-color": "#DC2626",
                    "color": "white",
                    "font-weight": "600",
                },
            }
        )
        
        st.markdown("---")
        st.markdown("<p style='text-align: center; color: #6B7280; font-size: 0.7rem;'>© 2024 HeartWise<br>UAS Data Mining</p>", unsafe_allow_html=True)
    
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
        ❤️ <strong>HeartWise</strong> - Sistem Prediksi Penyakit Jantung | Proyek UAS Data Mining<br>
        Vania Setyorini (24051214064) & Rozalinda Titalia Putri (24051214069)
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
