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
    page_title="Sistem Analisis Penyakit Jantung",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# KONFIGURASI PATH - SEDERHANA
# ============================================
# Cari file dataset
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

# Jika tidak ditemukan, download
if DATA_PATH is None or not DATA_PATH.exists():
    import urllib.request
    os.makedirs("dataset", exist_ok=True)
    url = "https://raw.githubusercontent.com/datasets/heart-disease/main/data/cleveland.csv"
    urllib.request.urlretrieve(url, "dataset/heart.csv")
    DATA_PATH = Path("dataset/heart.csv")
    st.sidebar.success("Dataset downloaded!")

st.sidebar.info(f"Dataset: {DATA_PATH}")

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
        
        # Cek apakah kolom sudah sesuai
        if 'target' not in df.columns:
            # Dataset Cleveland tanpa header
            column_names = FITUR + ['target']
            df = pd.read_csv(DATA_PATH, names=column_names, na_values='?')
            df = df.dropna()
            df['target'] = (df['target'] > 0).astype(int)
        
        # Pastikan hanya 13 fitur + target
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
# GET SHAP DATA (versi sederhana)
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
    
    # Mock SHAP values (karena shap library bermasalah)
    shap_values = np.zeros((len(X_test), len(FITUR)))
    for i, col in enumerate(FITUR):
        shap_values[:, i] = rf_model.feature_importances_[i] * (X_test[col].values - X_train[col].mean())
    
    return None, shap_values, X_test, rf_model

# ============================================
# FORM INPUT PASIEN
# ============================================
def form_input_pasien():
    st.markdown("### 📋 Data Klinis Pasien")
    
    col1, col2 = st.columns(2)
    with col1:
        usia = st.slider("Usia (tahun)", 20, 90, 52)
        jenis_kelamin = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
        nyeri_dada = st.selectbox("Tipe Nyeri Dada", [0, 1, 2, 3], format_func=lambda x: TIPE_NYERI_DADA[x])
        tekanan_darah = st.slider("Tekanan Darah (mmHg)", 80, 220, 125)
        kolesterol = st.slider("Kolesterol (mg/dl)", 100, 600, 212)
    
    with col2:
        gula_darah = st.selectbox("Gula Darah Puasa >120", [0, 1], format_func=lambda x: "Normal" if x == 0 else "Tinggi")
        ekg = st.selectbox("Hasil EKG", [0, 1, 2], format_func=lambda x: ["Normal", "Kelainan ST-T", "Hipertrofi"][x])
        detak_jantung = st.slider("Detak Jantung Maks", 60, 230, 168)
        angina = st.selectbox("Angina Olahraga", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        st_depresi = st.slider("Depresi ST", 0.0, 7.0, 1.0, 0.1)
        kemiringan_st = st.selectbox("Kemiringan ST", [0, 1, 2], format_func=lambda x: ["Menanjak", "Datar", "Menurun"][x])
        pembuluh = st.selectbox("Pembuluh Besar", [0, 1, 2, 3, 4])
        thalassemia = st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: ["Normal", "Cacat Tetap", "Cacat Reversibel", "Tidak Diketahui"][x])
    
    return pd.DataFrame([[usia, jenis_kelamin, nyeri_dada, tekanan_darah, kolesterol, gula_darah, ekg, 
                         detak_jantung, angina, st_depresi, kemiringan_st, pembuluh, thalassemia]], 
                       columns=FITUR)

# ============================================
# HALAMAN BERANDA
# ============================================
def halaman_beranda():
    st.markdown("# ❤️ Sistem Analisis Penyakit Jantung")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea, #764ba2); padding: 2rem; border-radius: 20px; color: white;'>
        <h2>🎓 Proyek UAS Data Mining</h2>
        <p>Aplikasi ini menerapkan metodologi CRISP-DM untuk analisis dan prediksi penyakit jantung.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 👩‍🎓 Vania Setyorini")
        st.markdown("NIM: 24051214064")
    with col2:
        st.markdown("### 👩‍🎓 Rozalinda Titalia Putri")
        st.markdown("NIM: 24051214069")

# ============================================
# HALAMAN DATASET
# ============================================
def halaman_dataset():
    df = load_data()
    if df.empty:
        st.error("Dataset tidak ditemukan!")
        return
    
    st.markdown("# 📊 Analisis Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data", df.shape[0])
    with col2:
        st.metric("Jumlah Fitur", df.shape[1] - 1)
    with col3:
        st.metric("Kelas Target", df['target'].nunique())
    with col4:
        st.metric("Pasien Sakit", (df['target'] == 1).sum())
    
    st.dataframe(df.head(20), use_container_width=True)

# ============================================
# HALAMAN PREDIKSI
# ============================================
def halaman_prediksi():
    st.markdown("# 🔮 Prediksi Penyakit Jantung")
    
    result = train_advanced_models()
    if result[0] is None:
        st.error("Gagal melatih model. Periksa dataset.")
        return
    
    advanced_models, scaler, X_test, y_test, X_train, y_train = result
    user_df = form_input_pasien()
    
    if st.button("Jalankan Analisis", type="primary", use_container_width=True):
        with st.spinner("Menjalankan AI..."):
            user_scaled = scaler.transform(user_df)
            
            ensemble_pred = advanced_models['Ensemble'].predict(user_scaled)[0]
            ensemble_prob = advanced_models['Ensemble'].predict_proba(user_scaled)[0][1]
            
            if ensemble_pred == 1:
                st.error(f"⚠️ RISIKO TINGGI - Probabilitas: {ensemble_prob:.1%}")
            else:
                st.success(f"✅ RISIKO RENDAH - Probabilitas: {ensemble_prob:.1%}")
            
            # Profil pasien
            st.markdown("### Profil Pasien")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Usia", f"{user_df['age'].values[0]} tahun")
                st.metric("Tekanan Darah", f"{user_df['trestbps'].values[0]} mmHg")
            with col2:
                st.metric("Kolesterol", f"{user_df['chol'].values[0]} mg/dl")
                st.metric("Detak Jantung", f"{user_df['thalach'].values[0]} bpm")
            with col3:
                st.metric("Nyeri Dada", TIPE_NYERI_DADA[user_df['cp'].values[0]])
                st.metric("Depresi ST", user_df['oldpeak'].values[0])

# ============================================
# HALAMAN XAI
# ============================================
def halaman_xai():
    st.markdown("# 🧠 Explainable AI")
    
    explainer, shap_values, X_test, rf_model = get_shap_data()
    
    if shap_values is None:
        st.error("Gagal memuat data XAI")
        return
    
    # Feature Importance
    importance_df = pd.DataFrame({
        'Fitur': FITUR,
        'Importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(importance_df.tail(10), x='Importance', y='Fitur', orientation='h',
                 title='10 Fitur Terpenting', color='Importance', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# HALAMAN DASHBOARD
# ============================================
def halaman_dashboard():
    st.markdown("# 📊 Dashboard")
    
    df = load_data()
    if df.empty:
        st.error("Dataset tidak ditemukan!")
        return
    
    # Distribusi usia vs target
    fig = px.histogram(df, x='age', color='target', nbins=30,
                       title='Distribusi Usia Berdasarkan Target',
                       color_discrete_map={0: 'blue', 1: 'red'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Korelasi
    corr = df[FITUR + ['target']].corr()
    fig = px.imshow(corr, text_auto='.2f', title='Matriks Korelasi')
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# HALAMAN TENTANG
# ============================================
def halaman_tentang():
    st.markdown("# ℹ️ Tentang Proyek")
    
    st.markdown("""
    ### HeartAI - Sistem Prediksi Penyakit Jantung
    
    Aplikasi ini dikembangkan untuk memenuhi Proyek UAS mata kuliah Data Mining.
    
    **Anggota Kelompok:**
    - Vania Setyorini (24051214064)
    - Rozalinda Titalia Putri (24051214069)
    
    **Teknologi:**
    - Streamlit untuk web framework
    - Scikit-learn untuk machine learning
    - Plotly untuk visualisasi
    
    ⚠️ **Disclaimer:** Aplikasi ini hanya untuk tujuan pembelajaran. Bukan alat diagnosis medis.
    """)

# ============================================
# MAIN
# ============================================
def main():
    with st.sidebar:
        st.markdown("# ❤️ HeartAI")
        st.markdown("Proyek UAS Data Mining")
        st.markdown("---")
        
        menu = option_menu(
            menu_title=None,
            options=["Beranda", "Dataset", "Prediksi", "XAI Analysis", "Dashboard", "Tentang"],
            icons=["house", "database", "robot", "brain", "speedometer2", "info-circle"],
            default_index=0,
            styles={
                "nav-link": {"font-size": "1rem", "margin": "0.3rem 0"},
                "nav-link-selected": {"background": "linear-gradient(135deg, #667eea, #764ba2)"},
            }
        )
    
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

if __name__ == "__main__":
    main()
