# UAS Data Mining - Prediksi Risiko Penyakit Jantung

Proyek ini menggunakan dataset Heart Disease untuk membangun solusi Data Mining dengan dua metode:

1. **Classification**: Logistic Regression dan Random Forest untuk memprediksi risiko penyakit jantung.
2. **Clustering**: K-Means untuk mengelompokkan profil pasien berdasarkan karakteristik klinis.

Model terbaik yang diimplementasikan pada aplikasi Streamlit adalah **Logistic Regression**, karena performanya stabil pada validasi deduplikasi dan lebih mudah diinterpretasikan.

## Struktur Folder

```text
UAS_DataMining_HeartDisease
├── dataset/heart.csv.csv
├── notebook/analysis.ipynb
├── model/heart_model.pkl
├── model/kmeans_model.pkl
├── app/app.py
├── app/assets
├── laporan/laporan_heart_disease.pdf
├── requirements.txt
└── README.md
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Catatan Dataset

Dataset memiliki 1.025 record dan 14 atribut. Tidak terdapat missing value. Namun terdapat duplikasi baris yang cukup banyak, sehingga laporan menyertakan catatan bahwa evaluasi pada data mentah dapat menghasilkan skor terlalu tinggi. Validasi tambahan pada data deduplikasi digunakan sebagai pembanding.
