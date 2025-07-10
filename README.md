# Klasifikasi Vegetasi dari Citra Sentinel-2

Aplikasi web untuk ekstraksi dan klasifikasi informasi vegetasi menggunakan Band B4 (Red) dan B8 (NIR) dari citra Sentinel-2 dengan algoritma K-Means clustering.

## Fitur Utama

- 🛰️ **Input Data Citra Satelit**: Upload file B4 (Red) dan B8 (NIR) atau gunakan data sampel
- 🌱 **Perhitungan NDVI**: Normalized Difference Vegetation Index untuk analisis vegetasi
- 🔄 **Klasifikasi K-Means**: Clustering otomatis dengan 4 klaster (Air, Bangunan/Jalan, Lahan Terbuka, Vegetasi)
- 📊 **Visualisasi Interaktif**: Grafik dan peta hasil klasifikasi menggunakan Plotly
- 📈 **Evaluasi Kualitas**: Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index

## 🎨 Fitur Optimasi

- **Konfigurasi Streamlit**: File `.streamlit/config.toml` untuk pengaturan optimal
- **Tema Kustom**: Tema gelap dengan warna hijau untuk aplikasi vegetasi
- **Performa**: Konfigurasi cache dan optimasi loading
- **Responsif**: Tampilan yang optimal di berbagai ukuran layar
- **Clean UI**: Menghilangkan branding "Made with Streamlit" dan elemen UI yang tidak perlu
- **Custom Styling**: File `style.css` untuk styling yang lebih profesional

## Instalasi Lokal

1. Clone atau download repository ini
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```bash
   streamlit run vegetasi.py
   ```

## Deploy ke Streamlit Cloud

### Persiapan Repository

1. **Upload ke GitHub**:
   - Buat repository baru di GitHub
   - Upload semua file (`vegetasi.py`, `requirements.txt`, `README.md`)
   - Pastikan repository bersifat public atau private dengan akses yang sesuai

### Deploy ke Streamlit Cloud

1. **Akses Streamlit Cloud**:
   - Kunjungi [share.streamlit.io](https://share.streamlit.io)
   - Login dengan akun GitHub Anda

2. **Deploy Aplikasi**:
   - Klik "New app"
   - Pilih repository GitHub Anda
   - Pilih branch (biasanya `main` atau `master`)
   - Pilih file utama: `vegetasi.py`
   - Klik "Deploy!"

3. **Konfigurasi Tambahan** (jika diperlukan):
   - Streamlit akan otomatis membaca `requirements.txt`
   - Proses deployment biasanya memakan waktu 2-5 menit

### URL Aplikasi

Setelah berhasil deploy, aplikasi akan tersedia di:
```
https://[username]-[repository-name]-[branch]-[random-string].streamlit.app
```

### Tips Deployment

1. **Optimasi Requirements**:
   - Pastikan hanya library yang diperlukan ada di `requirements.txt`
   - Gunakan versi yang kompatibel

2. **Ukuran File**:
   - Hindari file data yang terlalu besar (>100MB)
   - Gunakan data sampel untuk demo

3. **Error Handling**:
   - Tambahkan try-catch untuk upload file
   - Validasi input pengguna

4. **Performance**:
   - Gunakan `@st.cache_data` untuk fungsi yang memproses data besar
   - Optimasi visualisasi untuk performa yang lebih baik

## Struktur Aplikasi

```
vegetasi/
├── vegetasi.py          # Aplikasi utama Streamlit
├── requirements.txt     # Dependencies Python
└── README.md           # Dokumentasi
```

## Teknologi yang Digunakan

- **Streamlit**: Framework web app
- **Scikit-learn**: Machine learning (K-Means)
- **NumPy & Pandas**: Manipulasi data
- **Plotly**: Visualisasi interaktif
- **Rasterio**: Pemrosesan data geospasial
- **SciPy**: Komputasi ilmiah

## Kontribusi

Silakan buat issue atau pull request untuk perbaikan dan pengembangan aplikasi ini.

## Lisensi

MIT License - silakan gunakan untuk keperluan akademis dan penelitian.