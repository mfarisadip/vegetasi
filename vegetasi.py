import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import rasterio
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Klasifikasi Vegetasi Sentinel-2",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Klasifikasi Vegetasi dari Citra Sentinel-2")
st.markdown("**Ekstraksi dan Klasifikasi Informasi Vegetasi menggunakan Band B4 (Red) dan B8 (NIR)**")

# Sidebar untuk parameter
st.sidebar.header("Parameter Klasifikasi")
st.sidebar.markdown("**Klaster yang akan dibentuk:**")
st.sidebar.markdown("- C1: Air")
st.sidebar.markdown("- C2: Bangunan/Jalan") 
st.sidebar.markdown("- C3: Lahan Terbuka")
st.sidebar.markdown("- C4: Vegetasi")

# Fixed number of clusters
n_clusters = 4
max_iter = st.sidebar.slider("Iterasi Maksimum", 50, 500, 300)
random_state = st.sidebar.number_input("Random State", 0, 100, 42)

class VegetationClassifier:
    def __init__(self, n_clusters=4, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.features = None
        self.labels = None
        self.cluster_names = {
            0: "Air",
            1: "Bangunan/Jalan", 
            2: "Lahan Terbuka",
            3: "Vegetasi"
        }
        
    def extract_features(self, red_band, nir_band):
        """
        Ekstraksi fitur: Red, NIR, dan NDVI
        """
        st.info("🔄 Tahap 3: Ekstraksi Fitur Citra (Red, NIR, NDVI)")
        
        # Pastikan data dalam format float untuk menghindari pembagian integer
        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        
        # Hitung NDVI dengan penanganan pembagian nol
        denominator = nir + red
        ndvi = np.where(denominator != 0, (nir - red) / denominator, 0)
        
        # Flatten arrays untuk clustering
        red_flat = red.flatten()
        nir_flat = nir.flatten()
        ndvi_flat = ndvi.flatten()
        
        # Gabungkan fitur
        features = np.column_stack([red_flat, nir_flat, ndvi_flat])
        
        # Hapus nilai NaN dan infinite
        valid_mask = np.isfinite(features).all(axis=1)
        features = features[valid_mask]
        
        self.features = features
        self.original_shape = red.shape
        self.valid_mask = valid_mask
        
        return features, ndvi
    
    def calculate_euclidean_distance(self, data, centroids):
        """
        Tahap 4: Perhitungan Jarak Euclidean
        """
        distances = []
        for centroid in centroids:
            dist = np.array([euclidean(point, centroid) for point in data])
            distances.append(dist)
        return np.array(distances).T
    
    def apply_kmeans_clustering(self):
        """
        Tahap 5: Penerapan Algoritma K-Means dengan 4 klaster tetap
        """
        st.info(f"🔄 Tahap 5: Penerapan Algoritma K-Means dengan {self.n_clusters} klaster")
        
        # Inisialisasi K-Means dengan 4 klaster
        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, 
                       random_state=self.random_state, n_init=10)
        
        # Fit model dan dapatkan labels
        labels = kmeans.fit_predict(self.features)
        centroids = kmeans.cluster_centers_
        
        # Analisis karakteristik setiap cluster untuk labeling yang tepat
        cluster_characteristics = []
        for i in range(self.n_clusters):
            cluster_mask = labels == i
            cluster_data = self.features[cluster_mask]
            
            ndvi_mean = np.mean(cluster_data[:, 2])  # NDVI
            red_mean = np.mean(cluster_data[:, 0])   # Red
            nir_mean = np.mean(cluster_data[:, 1])   # NIR
            
            # Hitung rasio NIR/Red untuk membantu identifikasi
            nir_red_ratio = nir_mean / (red_mean + 1e-8)
            
            cluster_characteristics.append({
                'cluster_id': i,
                'ndvi_mean': ndvi_mean,
                'red_mean': red_mean,
                'nir_mean': nir_mean,
                'nir_red_ratio': nir_red_ratio,
                'size': np.sum(cluster_mask)
            })
        
        # Sorting berdasarkan multiple criteria untuk labeling yang lebih akurat
        cluster_characteristics.sort(key=lambda x: (x['ndvi_mean'], x['nir_red_ratio']))
        
        # Buat mapping berdasarkan karakteristik yang lebih detail
        label_mapping = {}
        sorted_centroids = np.zeros_like(centroids)
        
        for new_label, char in enumerate(cluster_characteristics):
            old_label = char['cluster_id']
            label_mapping[old_label] = new_label
            sorted_centroids[new_label] = centroids[old_label]
            
            # Debug info untuk memahami karakteristik
            st.write(f"Cluster {new_label}: NDVI={char['ndvi_mean']:.3f}, NIR/Red={char['nir_red_ratio']:.3f}")
        
        # Relabel berdasarkan mapping yang sudah dibuat
        new_labels = np.array([label_mapping[label] for label in labels])
        self.labels = new_labels
        
        # Update model dengan centroids yang sudah diurutkan
        self.model = kmeans
        
        # Tahap 7: Update Centroid berdasarkan jarak Euclidean
        st.info("🔄 Tahap 7: Update Centroid berdasarkan Jarak Euclidean")
        
        return self.labels, sorted_centroids
    
    def evaluate_clustering_quality(self):
        """
        Tahap 6: Evaluasi kualitas clustering dengan berbagai metrik
        """
        st.info("🔄 Tahap 6: Evaluasi Kualitas Clustering")
        
        if self.labels is not None and self.features is not None:
            # Silhouette Score (rata-rata)
            silhouette_avg = silhouette_score(self.features, self.labels)
            
            # Silhouette Score per sampel
            silhouette_samples_scores = silhouette_samples(self.features, self.labels)
            
            # Calinski-Harabasz Index (Variance Ratio Criterion)
            calinski_harabasz = calinski_harabasz_score(self.features, self.labels)
            
            # Davies-Bouldin Index
            davies_bouldin = davies_bouldin_score(self.features, self.labels)
            
            # Simpan hasil evaluasi
            self.evaluation_results = {
                'silhouette_avg': silhouette_avg,
                'silhouette_samples': silhouette_samples_scores,
                'calinski_harabasz': calinski_harabasz,
                'davies_bouldin': davies_bouldin
            }
            
            return self.evaluation_results
        return None
    
    def create_classification_map(self):
        """
        Tahap 8: Klasifikasi Akhir - Membuat peta segmentasi
        """
        st.info("🔄 Tahap 8: Klasifikasi Akhir - Visualisasi Peta Segmentasi")
        
        # Buat array kosong dengan shape asli
        classification_map = np.full(self.original_shape[0] * self.original_shape[1], -1)
        
        # Isi dengan hasil klasifikasi
        classification_map[self.valid_mask] = self.labels
        
        # Reshape ke dimensi asli
        classification_map = classification_map.reshape(self.original_shape)
        
        return classification_map

def load_sample_data():
    """
    Membuat data sampel Sentinel-2 untuk demonstrasi
    """
    np.random.seed(42)
    height, width = 100, 100
    
    # Simulasi data B4 (Red) dan B8 (NIR)
    # Area vegetasi (NDVI tinggi)
    veg_mask = np.zeros((height, width), dtype=bool)
    veg_mask[20:40, 20:60] = True  # Area hutan
    veg_mask[60:80, 10:50] = True  # Area pertanian
    
    # Area non-vegetasi (NDVI rendah)
    urban_mask = np.zeros((height, width), dtype=bool)
    urban_mask[10:30, 70:90] = True  # Area urban
    
    # Area air (NDVI sangat rendah)
    water_mask = np.zeros((height, width), dtype=bool)
    water_mask[70:90, 70:90] = True  # Area air
    
    # Generate data
    b4_red = np.random.uniform(0.05, 0.15, (height, width))  # Red band
    b8_nir = np.random.uniform(0.10, 0.25, (height, width))  # NIR band
    
    # Adjust values untuk area vegetasi (NIR tinggi, Red rendah)
    b4_red[veg_mask] = np.random.uniform(0.03, 0.08, np.sum(veg_mask))
    b8_nir[veg_mask] = np.random.uniform(0.25, 0.45, np.sum(veg_mask))
    
    # Adjust values untuk area urban (NIR dan Red sedang)
    b4_red[urban_mask] = np.random.uniform(0.12, 0.18, np.sum(urban_mask))
    b8_nir[urban_mask] = np.random.uniform(0.15, 0.22, np.sum(urban_mask))
    
    # Adjust values untuk area air (NIR dan Red rendah)
    b4_red[water_mask] = np.random.uniform(0.02, 0.05, np.sum(water_mask))
    b8_nir[water_mask] = np.random.uniform(0.01, 0.03, np.sum(water_mask))
    
    return b4_red, b8_nir

def plot_cluster_characteristics(centroids, cluster_names):
    """
    Visualisasi karakteristik setiap klaster
    """
    df_centroids = pd.DataFrame(centroids, columns=['Red', 'NIR', 'NDVI'])
    df_centroids['Cluster'] = [f"C{i+1}: {cluster_names[i]}" for i in range(len(centroids))]
    
    # Bar chart untuk setiap fitur
    fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=('Red Band', 'NIR Band', 'NDVI'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]])
    
    colors = ['blue', 'red', 'orange', 'green']
    
    for i, (feature, title) in enumerate([('Red', 'Red Band'), ('NIR', 'NIR Band'), ('NDVI', 'NDVI')]):
        fig.add_trace(
            go.Bar(x=df_centroids['Cluster'], y=df_centroids[feature], 
                   name=title, marker_color=colors[i % len(colors)], showlegend=False),
            row=1, col=i+1
        )
    
    fig.update_layout(height=400, title_text="Karakteristik Centroid Setiap Klaster")
    return fig

def plot_silhouette_analysis(features, labels, cluster_names):
    """
    Visualisasi analisis silhouette score per klaster
    """
    silhouette_avg = silhouette_score(features, labels)
    sample_silhouette_values = silhouette_samples(features, labels)
    
    # Buat subplot untuk silhouette plot
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Silhouette Plot per Klaster', 'Distribusi Silhouette Score'),
                       specs=[[{"secondary_y": False}, {"secondary_y": False}]])
    
    colors = ['#0066CC', '#CC0000', '#FF9900', '#00CC00']
    y_lower = 10
    
    # Plot silhouette untuk setiap klaster
    for i in range(len(np.unique(labels))):
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        # Buat array y untuk plotting
        y_values = np.arange(y_lower, y_upper)
        
        fig.add_trace(
            go.Scatter(
                x=cluster_silhouette_values,
                y=y_values,
                mode='lines',
                fill='tozeroy',
                fillcolor=colors[i],
                line=dict(color=colors[i], width=0),
                name=f'C{i+1}: {cluster_names[i]}',
                showlegend=True
            ),
            row=1, col=1
        )
        
        y_lower = y_upper + 10
    
    # Tambahkan garis rata-rata silhouette score
    fig.add_vline(x=silhouette_avg, line_dash="dash", line_color="red", 
                  annotation_text=f"Rata-rata: {silhouette_avg:.3f}", row=1, col=1)
    
    # Histogram distribusi silhouette score
    fig.add_trace(
        go.Histogram(
            x=sample_silhouette_values,
            nbinsx=30,
            name='Distribusi Silhouette',
            marker_color='lightblue',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Tambahkan garis rata-rata pada histogram
    fig.add_vline(x=silhouette_avg, line_dash="dash", line_color="red", 
                  annotation_text=f"Rata-rata: {silhouette_avg:.3f}", row=1, col=2)
    
    fig.update_layout(
        height=500,
        title_text=f"Analisis Silhouette Score (Rata-rata: {silhouette_avg:.4f})"
    )
    
    fig.update_xaxes(title_text="Silhouette Score", row=1, col=1)
    fig.update_yaxes(title_text="Indeks Sampel", row=1, col=1)
    fig.update_xaxes(title_text="Silhouette Score", row=1, col=2)
    fig.update_yaxes(title_text="Frekuensi", row=1, col=2)
    
    return fig

def plot_clustering_metrics_comparison(evaluation_results):
    """
    Visualisasi perbandingan berbagai metrik evaluasi clustering
    """
    metrics = {
        'Silhouette Score': evaluation_results['silhouette_avg'],
        'Calinski-Harabasz Index': evaluation_results['calinski_harabasz'],
        'Davies-Bouldin Index': evaluation_results['davies_bouldin']
    }
    
    # Normalisasi untuk visualisasi (Davies-Bouldin dibalik karena semakin kecil semakin baik)
    normalized_metrics = {
        'Silhouette Score': evaluation_results['silhouette_avg'],  # Range: -1 to 1
        'Calinski-Harabasz (Normalized)': min(evaluation_results['calinski_harabasz'] / 1000, 1),  # Normalisasi
        'Davies-Bouldin (Inverted)': 1 / (1 + evaluation_results['davies_bouldin'])  # Invert karena lower is better
    }
    
    fig = go.Figure()
    
    # Bar chart untuk metrik
    fig.add_trace(go.Bar(
        x=list(normalized_metrics.keys()),
        y=list(normalized_metrics.values()),
        marker_color=['green', 'blue', 'orange'],
        text=[f"{v:.4f}" for v in normalized_metrics.values()],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Perbandingan Metrik Evaluasi Clustering",
        xaxis_title="Metrik Evaluasi",
        yaxis_title="Nilai (Dinormalisasi)",
        height=400
    )
    
    return fig, metrics

def main():
    # Tahap 1: Inisialisasi Proses
    st.info("🚀 Tahap 1: Inisialisasi Proses - Memulai ekstraksi dan klasifikasi vegetasi")
    
    # Upload file atau gunakan data sampel
    st.subheader("📁 Input Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_sample = st.checkbox("Gunakan Data Sampel", value=True)
        
    with col2:
        if not use_sample:
            st.info("Upload file B4 (Red) dan B8 (NIR) dalam format TIFF/GeoTIFF")
            b4_file = st.file_uploader("Upload Band B4 (Red)", type=['tif', 'tiff'])
            b8_file = st.file_uploader("Upload Band B8 (NIR)", type=['tif', 'tiff'])
    
    if use_sample or (not use_sample and 'b4_file' in locals() and 'b8_file' in locals() and b4_file and b8_file):
        
        # Tahap 2: Masukkan Data Citra Satelit
        st.info("📡 Tahap 2: Input Data Citra Satelit Sentinel-2")
        
        if use_sample:
            st.success("Menggunakan data sampel Sentinel-2")
            b4_red, b8_nir = load_sample_data()
        else:
            # Load data dari file upload
            try:
                with rasterio.open(b4_file) as src:
                    b4_red = src.read(1)
                with rasterio.open(b8_file) as src:
                    b8_nir = src.read(1)
                st.success("Data berhasil dimuat dari file upload")
            except Exception as e:
                st.error(f"Error loading files: {str(e)}")
                return
        
        # Tampilkan informasi data
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dimensi Data", f"{b4_red.shape[0]} x {b4_red.shape[1]}")
        with col2:
            st.metric("B4 Range", f"{b4_red.min():.3f} - {b4_red.max():.3f}")
        with col3:
            st.metric("B8 Range", f"{b8_nir.min():.3f} - {b8_nir.max():.3f}")
        
        # Visualisasi band asli
        st.subheader("📊 Visualisasi Band Asli")
        fig_bands = make_subplots(rows=1, cols=2, 
                                 subplot_titles=('Band B4 (Red)', 'Band B8 (NIR)'))
        
        fig_bands.add_trace(go.Heatmap(z=b4_red, colorscale='Reds', name='B4'), row=1, col=1)
        fig_bands.add_trace(go.Heatmap(z=b8_nir, colorscale='Greens', name='B8'), row=1, col=2)
        
        fig_bands.update_layout(height=400, title_text="Band Spektral Sentinel-2")
        st.plotly_chart(fig_bands, use_container_width=True)
        
        # Inisialisasi classifier
        classifier = VegetationClassifier(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
        
        # Ekstraksi fitur
        features, ndvi = classifier.extract_features(b4_red, b8_nir)
        
        # Tampilkan NDVI
        st.subheader("🌱 Normalized Difference Vegetation Index (NDVI)")
        fig_ndvi = go.Figure(data=go.Heatmap(z=ndvi, colorscale='RdYlGn', 
                                           zmin=-1, zmax=1))
        fig_ndvi.update_layout(title='NDVI (Normalized Difference Vegetation Index)',
                              height=400)
        st.plotly_chart(fig_ndvi, use_container_width=True)
        
        # Statistik NDVI
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("NDVI Min", f"{ndvi.min():.3f}")
        with col2:
            st.metric("NDVI Max", f"{ndvi.max():.3f}")
        with col3:
            st.metric("NDVI Mean", f"{ndvi.mean():.3f}")
        with col4:
            st.metric("NDVI Std", f"{ndvi.std():.3f}")
        
        # Tombol untuk memulai klasifikasi
        if st.button("🔄 Mulai Proses Klasifikasi", type="primary"):
            
            # Progress tracking
            progress_container = st.container()
            
            with progress_container:
                # Tahap 5 & 7: Clustering dengan 4 klaster tetap
                with st.spinner('Menerapkan algoritma K-Means dengan 4 klaster...'):
                    labels, centroids = classifier.apply_kmeans_clustering()
                
                # Tahap 6: Evaluasi kualitas clustering
                with st.spinner('Mengevaluasi kualitas clustering...'):
                    evaluation_results = classifier.evaluate_clustering_quality()
                
                if evaluation_results:
                    silhouette_avg = evaluation_results['silhouette_avg']
                    st.success(f"✅ Clustering selesai dengan Silhouette Score: {silhouette_avg:.4f}")
                else:
                    st.warning("⚠️ Evaluasi clustering tidak berhasil.")
                
                # Tahap 8: Klasifikasi Akhir
                with st.spinner('Membuat peta klasifikasi...'):
                    classification_map = classifier.create_classification_map()
                
                st.success("✅ Klasifikasi selesai!")
                
                # Hasil Klasifikasi
                st.subheader("🗺️ Hasil Klasifikasi Vegetasi")
                
                # Peta klasifikasi dengan custom colorscale
                custom_colors = ['#0066CC', '#CC0000', '#FF9900', '#00CC00']  # Air, Bangunan, Lahan Terbuka, Vegetasi
                
                fig_classification = go.Figure(data=go.Heatmap(
                    z=classification_map, 
                    colorscale=[[0, custom_colors[0]], [0.33, custom_colors[1]], 
                               [0.66, custom_colors[2]], [1, custom_colors[3]]],
                    showscale=True,
                    colorbar=dict(
                        title="Klaster",
                        tickmode="array",
                        tickvals=[0, 1, 2, 3],
                        ticktext=["C1: Air", "C2: Bangunan/Jalan", "C3: Lahan Terbuka", "C4: Vegetasi"]
                    )
                ))
                fig_classification.update_layout(
                    title='Peta Klasifikasi Tutupan Lahan (4 Klaster)',
                    height=500
                )
                st.plotly_chart(fig_classification, use_container_width=True)
                
                # Karakteristik klaster
                st.subheader("📊 Karakteristik Setiap Klaster")
                fig_char = plot_cluster_characteristics(centroids, classifier.cluster_names)
                st.plotly_chart(fig_char, use_container_width=True)
                
                # Evaluasi kualitas clustering yang komprehensif
                st.subheader("📊 Evaluasi Kualitas Clustering")
                
                # Dapatkan hasil evaluasi lengkap
                evaluation_results = classifier.evaluation_results
                
                if evaluation_results:
                    # Tampilkan metrik utama
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Silhouette Score", f"{evaluation_results['silhouette_avg']:.4f}")
                        st.caption("Range: -1 (buruk) hingga 1 (sangat baik)")
                    
                    with col2:
                        st.metric("Calinski-Harabasz Index", f"{evaluation_results['calinski_harabasz']:.2f}")
                        st.caption("Semakin tinggi semakin baik")
                    
                    with col3:
                        st.metric("Davies-Bouldin Index", f"{evaluation_results['davies_bouldin']:.4f}")
                        st.caption("Semakin rendah semakin baik")
                    
                    # Visualisasi perbandingan metrik
                    st.subheader("📈 Perbandingan Metrik Evaluasi")
                    metrics_fig, raw_metrics = plot_clustering_metrics_comparison(evaluation_results)
                    st.plotly_chart(metrics_fig, use_container_width=True)
                    
                    # Tampilkan nilai mentah metrik
                    with st.expander("📋 Nilai Mentah Metrik Evaluasi"):
                        metrics_df = pd.DataFrame(list(raw_metrics.items()), columns=['Metrik', 'Nilai'])
                        st.dataframe(metrics_df, use_container_width=True)
                    
                    # Analisis Silhouette Score Detail
                    st.subheader("🎯 Analisis Silhouette Score Detail")
                    
                    # Visualisasi silhouette analysis
                    if 'silhouette_samples' in evaluation_results:
                        silhouette_fig = plot_silhouette_analysis(classifier.features, labels, classifier.cluster_names)
                        st.plotly_chart(silhouette_fig, use_container_width=True)
                        
                        st.subheader("📈 Statistik Silhouette Score per Klaster")
                        silhouette_samples_data = evaluation_results['silhouette_samples']
                        
                        # Buat dataframe untuk analisis
                        df_silhouette = pd.DataFrame({
                            'Sample_Index': range(len(silhouette_samples_data)),
                            'Silhouette_Score': silhouette_samples_data,
                            'Cluster': labels
                        })
                        
                        # Statistik per klaster
                        cluster_stats = df_silhouette.groupby('Cluster')['Silhouette_Score'].agg([
                            'mean', 'std', 'min', 'max', 'count'
                        ]).round(4)
                        cluster_stats.columns = ['Rata-rata', 'Std Dev', 'Min', 'Max', 'Jumlah Sampel']
                        cluster_stats.index = [f"C{i+1}: {classifier.cluster_names[i]}" for i in cluster_stats.index]
                        
                        st.dataframe(cluster_stats, use_container_width=True)
                        
                        # Analisis kualitas per klaster
                        st.subheader("🔍 Analisis Kualitas per Klaster")
                        for i, cluster_name in enumerate(classifier.cluster_names.values()):
                            cluster_silhouette = df_silhouette[df_silhouette['Cluster'] == i]['Silhouette_Score']
                            avg_score = cluster_silhouette.mean()
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                if avg_score > 0.6:
                                    st.success(f"**C{i+1}: {cluster_name}** - Silhouette rata-rata: {avg_score:.4f} (Sangat Baik)")
                                elif avg_score > 0.4:
                                    st.info(f"**C{i+1}: {cluster_name}** - Silhouette rata-rata: {avg_score:.4f} (Baik)")
                                elif avg_score > 0.2:
                                    st.warning(f"**C{i+1}: {cluster_name}** - Silhouette rata-rata: {avg_score:.4f} (Sedang)")
                                else:
                                    st.error(f"**C{i+1}: {cluster_name}** - Silhouette rata-rata: {avg_score:.4f} (Kurang Baik)")
                            
                            with col2:
                                st.metric("Std Dev", f"{cluster_silhouette.std():.4f}")
                        
                        # Identifikasi sampel dengan silhouette score rendah
                        low_silhouette_threshold = 0.1
                        low_silhouette_samples = df_silhouette[df_silhouette['Silhouette_Score'] < low_silhouette_threshold]
                        
                        st.subheader("⚠️ Deteksi Outlier dan Misclassification")
                        if len(low_silhouette_samples) > 0:
                            st.warning(f"Ditemukan {len(low_silhouette_samples)} sampel dengan silhouette score < {low_silhouette_threshold} yang mungkin merupakan outlier atau misclassified.")
                            
                            # Tampilkan distribusi sampel bermasalah per klaster
                            problem_distribution = low_silhouette_samples['Cluster'].value_counts().sort_index()
                            st.write("**Distribusi sampel bermasalah per klaster:**")
                            for cluster_id, count in problem_distribution.items():
                                percentage = (count / len(df_silhouette[df_silhouette['Cluster'] == cluster_id])) * 100
                                st.write(f"- C{cluster_id+1} ({classifier.cluster_names[cluster_id]}): {count} sampel ({percentage:.1f}% dari klaster)")
                            
                            # Tampilkan detail sampel bermasalah
                            with st.expander("📋 Detail Sampel Bermasalah"):
                                problem_details = low_silhouette_samples.copy()
                                problem_details['Cluster_Name'] = [classifier.cluster_names[c] for c in problem_details['Cluster']]
                                st.dataframe(problem_details[['Sample_Index', 'Cluster', 'Cluster_Name', 'Silhouette_Score']], use_container_width=True)
                        else:
                            st.success("✅ Semua sampel memiliki silhouette score yang baik (≥ 0.1). Tidak ada outlier yang terdeteksi.")
                        
                        # Interpretasi hasil evaluasi
                        st.subheader("🔍 Interpretasi Hasil Evaluasi")
                        silhouette_avg = evaluation_results['silhouette_avg']
                        
                        if silhouette_avg > 0.7:
                            st.success("🟢 **Clustering Sangat Baik**: Klaster terbentuk dengan sangat jelas dan terpisah dengan baik.")
                        elif silhouette_avg > 0.5:
                            st.info("🔵 **Clustering Baik**: Klaster terbentuk dengan cukup jelas, ada sedikit overlap.")
                        elif silhouette_avg > 0.25:
                            st.warning("🟡 **Clustering Sedang**: Klaster terbentuk namun ada overlap yang cukup signifikan.")
                        else:
                            st.error("🔴 **Clustering Kurang Baik**: Klaster tidak terbentuk dengan jelas, banyak overlap atau struktur klaster lemah.")
                        
                        # Rekomendasi berdasarkan hasil analisis
                        st.subheader("💡 Rekomendasi")
                        if silhouette_avg > 0.5:
                            st.success("✅ **Hasil clustering sudah baik.** Klaster terbentuk dengan jelas dan dapat digunakan untuk analisis lebih lanjut.")
                        elif silhouette_avg > 0.25:
                            st.info("ℹ️ **Hasil clustering cukup baik.** Pertimbangkan untuk:")
                            st.write("- Menyesuaikan jumlah klaster (k)")
                            st.write("- Melakukan preprocessing data tambahan")
                            st.write("- Menggunakan algoritma clustering lain")
                        else:
                            st.warning("⚠️ **Hasil clustering perlu diperbaiki.** Disarankan untuk:")
                            st.write("- Mengubah jumlah klaster (k)")
                            st.write("- Melakukan feature engineering")
                            st.write("- Menggunakan algoritma clustering yang berbeda")
                            st.write("- Melakukan outlier detection dan removal")
                else:
                    st.warning("⚠️ Data evaluasi clustering tidak tersedia.")
                
                # Statistik klasifikasi
                st.subheader("📊 Statistik Klasifikasi")
                
                # Hitung distribusi klaster
                unique_labels, counts = np.unique(labels, return_counts=True)
                percentages = (counts / len(labels)) * 100
                
                # Tabel statistik dengan nama klaster
                stats_df = pd.DataFrame({
                    'Klaster': [f'C{i+1}: {classifier.cluster_names[i]}' for i in unique_labels],
                    'Jumlah Piksel': counts,
                    'Persentase (%)': [f"{p:.2f}%" for p in percentages]
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(stats_df, use_container_width=True)
                
                with col2:
                    # Pie chart distribusi dengan warna custom
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=[f'C{i+1}: {classifier.cluster_names[i]}' for i in unique_labels],
                        values=counts,
                        hole=0.3,
                        marker_colors=[custom_colors[i] for i in unique_labels]
                    )])
                    fig_pie.update_layout(title="Distribusi Klaster Tutupan Lahan")
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Informasi centroid
                st.subheader("🎯 Centroid Klaster")
                centroid_df = pd.DataFrame(
                    centroids, 
                    columns=['Red', 'NIR', 'NDVI'],
                    index=[f'C{i+1}: {classifier.cluster_names[i]}' for i in range(len(centroids))]
                )
                st.dataframe(centroid_df.round(4), use_container_width=True)
                
                # Interpretasi hasil
                st.subheader("🔍 Interpretasi Hasil Klasifikasi")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Karakteristik Setiap Klaster:**")
                    st.markdown("- **C1 (Air)**: NDVI < 0.1, Rasio NIR/Red < 1.5")
                    st.markdown("- **C2 (Bangunan/Jalan)**: NDVI < 0.4, Rasio NIR/Red < 2.0")
                    st.markdown("- **C3 (Lahan Terbuka)**: NDVI -0.2 s/d 0.5, Rasio bervariasi")
                    st.markdown("- **C4 (Vegetasi)**: NDVI > 0.2, Rasio NIR/Red > 1.5")
                
                with col2:
                    st.markdown("**Validasi Hasil:**")
                    for i, (centroid, name) in enumerate(zip(centroids, classifier.cluster_names.values())):
                        ndvi_val = centroid[2]
                        red_val = centroid[0]
                        nir_val = centroid[1]
                        nir_red_ratio = nir_val / (red_val + 1e-8)
                        
                        # Validasi yang lebih fleksibel berdasarkan karakteristik lengkap
                        if name == "Air":
                            if ndvi_val < 0.1 and nir_red_ratio < 1.5:
                                st.success(f"✅ {name}: NDVI = {ndvi_val:.3f}, Rasio NIR/Red = {nir_red_ratio:.2f}")
                            else:
                                st.info(f"ℹ️ {name}: NDVI = {ndvi_val:.3f}, Rasio NIR/Red = {nir_red_ratio:.2f} (Karakteristik bervariasi)")
                        
                        elif name == "Bangunan/Jalan":
                            if ndvi_val < 0.4 and nir_red_ratio < 2.0:
                                st.success(f"✅ {name}: NDVI = {ndvi_val:.3f}, Rasio NIR/Red = {nir_red_ratio:.2f}")
                            else:
                                st.info(f"ℹ️ {name}: NDVI = {ndvi_val:.3f}, Rasio NIR/Red = {nir_red_ratio:.2f} (Karakteristik bervariasi)")
                        
                        elif name == "Lahan Terbuka":
                            if -0.2 <= ndvi_val <= 0.5:
                                st.success(f"✅ {name}: NDVI = {ndvi_val:.3f}, Rasio NIR/Red = {nir_red_ratio:.2f}")
                            else:
                                st.info(f"ℹ️ {name}: NDVI = {ndvi_val:.3f}, Rasio NIR/Red = {nir_red_ratio:.2f} (Karakteristik bervariasi)")
                        
                        elif name == "Vegetasi":
                            if ndvi_val > 0.2 and nir_red_ratio > 1.5:
                                st.success(f"✅ {name}: NDVI = {ndvi_val:.3f}, Rasio NIR/Red = {nir_red_ratio:.2f}")
                            else:
                                st.info(f"ℹ️ {name}: NDVI = {ndvi_val:.3f}, Rasio NIR/Red = {nir_red_ratio:.2f} (Karakteristik bervariasi)")
                    
                    st.markdown("---")
                    st.markdown("**Catatan:**")
                    st.markdown("- Data sintesis dapat memiliki karakteristik yang berbeda dari data riil")
                    st.markdown("- Clustering otomatis dapat menghasilkan pembagian yang tidak persis sesuai ekspektasi")
                    st.markdown("- Rasio NIR/Red membantu membedakan vegetasi dari non-vegetasi")
                
                # Scatter plot 3D fitur dengan klaster yang sudah diberi nama
                st.subheader("📈 Visualisasi 3D Fitur dan Klaster")
                
                # Sampling untuk visualisasi (jika data terlalu besar)
                n_samples = min(5000, len(classifier.features))
                sample_idx = np.random.choice(len(classifier.features), n_samples, replace=False)
                
                fig_3d = go.Figure(data=[go.Scatter3d(
                    x=classifier.features[sample_idx, 0],  # Red
                    y=classifier.features[sample_idx, 1],  # NIR
                    z=classifier.features[sample_idx, 2],  # NDVI
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=[custom_colors[l] for l in labels[sample_idx]],
                        showscale=False
                    ),
                    text=[f'C{l+1}: {classifier.cluster_names[l]}' for l in labels[sample_idx]],
                    hovertemplate='Red: %{x:.3f}<br>NIR: %{y:.3f}<br>NDVI: %{z:.3f}<br>%{text}<extra></extra>'
                )])
                
                # Tambahkan centroid sebagai titik besar
                fig_3d.add_trace(go.Scatter3d(
                    x=centroids[:, 0],
                    y=centroids[:, 1], 
                    z=centroids[:, 2],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='black',
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    text=[f'Centroid C{i+1}: {name}' for i, name in classifier.cluster_names.items()],
                    name='Centroids',
                    hovertemplate='%{text}<br>Red: %{x:.3f}<br>NIR: %{y:.3f}<br>NDVI: %{z:.3f}<extra></extra>'
                ))
                
                fig_3d.update_layout(
                    title='Distribusi 3D Fitur: Red, NIR, dan NDVI dengan Centroids',
                    scene=dict(
                        xaxis_title='Red Band',
                        yaxis_title='NIR Band',
                        zaxis_title='NDVI'
                    ),
                    height=600,
                    showlegend=True
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Manual Calculation Results - Separated Sections
                st.subheader("📋 Hasil Perhitungan Manual")
                
                # Helper functions for manual calculations
                def get_pixel_coordinates(classifier, b4_red):
                    """Get pixel coordinates for valid pixels"""
                    height, width = b4_red.shape
                    y_coords, x_coords = np.meshgrid(range(height), range(width), indexing='ij')
                    y_flat = y_coords.flatten()
                    x_flat = x_coords.flatten()
                    valid_y = y_flat[classifier.valid_mask]
                    valid_x = x_flat[classifier.valid_mask]
                    return valid_y, valid_x
                
                def calculate_distances_to_centroids(classifier, centroids):
                    """Calculate Euclidean distances to all centroids"""
                    distances_to_centroids = []
                    for i, centroid in enumerate(centroids):
                        distances = np.array([euclidean(point, centroid) for point in classifier.features])
                        distances_to_centroids.append(distances)
                    return distances_to_centroids
                
                # Generate base data
                with st.spinner('Memproses data perhitungan manual...'):
                    valid_y, valid_x = get_pixel_coordinates(classifier, b4_red)
                    distances_to_centroids = calculate_distances_to_centroids(classifier, centroids)
                
                # Section 1: Feature Extraction per Pixel
                st.markdown("### 1️⃣ Ekstraksi Fitur per Pixel")
                st.markdown("Tahap ekstraksi nilai Red, NIR, NDVI, dan rasio NIR/Red untuk setiap pixel")
                
                # Create feature extraction table
                feature_table = pd.DataFrame({
                    'No': range(1, len(classifier.features) + 1),
                    'Pixel_Y': valid_y,
                    'Pixel_X': valid_x,
                    'Koordinat': [f'({y},{x})' for y, x in zip(valid_y, valid_x)],
                    'Red_B4': classifier.features[:, 0].round(4),
                    'NIR_B8': classifier.features[:, 1].round(4),
                    'NDVI': classifier.features[:, 2].round(4),
                    'NIR_Red_Ratio': (classifier.features[:, 1] / (classifier.features[:, 0] + 1e-8)).round(4)
                })
                
                # Display options for feature table
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    feature_display_mode = st.selectbox("Mode Tampilan:", ["Sample Random", "Semua Data"], key="feature_mode")
                with col2:
                    if feature_display_mode == "Sample Random":
                        feature_sample_size = st.selectbox("Jumlah Sample:", [100, 250, 500, 1000], key="feature_sample")
                    else:
                        feature_show_rows = st.selectbox("Tampilkan baris:", [10, 25, 50, 100], key="feature_rows")
                with col3:
                    feature_filter_cluster = st.selectbox("Filter klaster:", ["Semua"] + [f"C{i+1}: {name}" for i, name in classifier.cluster_names.items()], key="feature_filter")
                with col4:
                    feature_sort_by = st.selectbox("Urutkan berdasarkan:", ["No", "NDVI", "Red_B4", "NIR_B8", "NIR_Red_Ratio"], key="feature_sort")
                
                # Apply filtering and display for feature table
                feature_filtered = feature_table.copy()
                
                # Add cluster info for filtering
                feature_filtered['Klaster_Terpilih'] = [f'C{label+1}' for label in labels]
                
                if feature_filter_cluster != "Semua":
                    cluster_num = feature_filter_cluster.split(":")[0]
                    feature_filtered = feature_filtered[feature_filtered['Klaster_Terpilih'] == cluster_num]
                
                if feature_display_mode == "Sample Random":
                    if len(feature_filtered) > feature_sample_size:
                        feature_filtered = feature_filtered.sample(n=feature_sample_size, random_state=42)
                else:
                    feature_filtered = feature_filtered.head(feature_show_rows)
                
                if feature_sort_by != "No":
                    feature_filtered = feature_filtered.sort_values(feature_sort_by, ascending=False if feature_sort_by in ["NDVI", "NIR_B8", "NIR_Red_Ratio"] else True)
                
                # Reset index and add sequential number
                feature_filtered = feature_filtered.reset_index(drop=True)
                feature_filtered['No'] = range(1, len(feature_filtered) + 1)
                
                # Remove cluster column for display
                feature_display = feature_filtered.drop('Klaster_Terpilih', axis=1)
                
                st.dataframe(feature_display, use_container_width=True, height=300)
                st.info(f"Menampilkan {len(feature_display)} dari {len(feature_table)} total pixel")
                
                # Section 2: Euclidean Distance Calculation
                st.markdown("### 2️⃣ Perhitungan Jarak Euclidean")
                st.markdown("Perhitungan jarak setiap pixel ke semua centroid klaster")
                
                # Create distance calculation table
                distance_table = pd.DataFrame({
                    'No': range(1, len(classifier.features) + 1),
                    'Koordinat': [f'({y},{x})' for y, x in zip(valid_y, valid_x)],
                    'Red_B4': classifier.features[:, 0].round(4),
                    'NIR_B8': classifier.features[:, 1].round(4),
                    'NDVI': classifier.features[:, 2].round(4),
                    'Jarak_ke_C1': distances_to_centroids[0].round(4),
                    'Jarak_ke_C2': distances_to_centroids[1].round(4),
                    'Jarak_ke_C3': distances_to_centroids[2].round(4),
                    'Jarak_ke_C4': distances_to_centroids[3].round(4),
                    'Jarak_Terdekat': np.min(distances_to_centroids, axis=0).round(4)
                })
                
                # Display options for distance table
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    distance_display_mode = st.selectbox("Mode Tampilan:", ["Sample Random", "Semua Data"], key="distance_mode")
                with col2:
                    if distance_display_mode == "Sample Random":
                        distance_sample_size = st.selectbox("Jumlah Sample:", [100, 250, 500, 1000], key="distance_sample")
                    else:
                        distance_show_rows = st.selectbox("Tampilkan baris:", [10, 25, 50, 100], key="distance_rows")
                with col3:
                    distance_filter_cluster = st.selectbox("Filter klaster:", ["Semua"] + [f"C{i+1}: {name}" for i, name in classifier.cluster_names.items()], key="distance_filter")
                with col4:
                    distance_sort_by = st.selectbox("Urutkan berdasarkan:", ["No", "Jarak_Terdekat", "Jarak_ke_C1", "Jarak_ke_C2", "Jarak_ke_C3", "Jarak_ke_C4"], key="distance_sort")
                
                # Apply filtering and display for distance table
                distance_filtered = distance_table.copy()
                
                # Add cluster info for filtering
                distance_filtered['Klaster_Terpilih'] = [f'C{label+1}' for label in labels]
                
                if distance_filter_cluster != "Semua":
                    cluster_num = distance_filter_cluster.split(":")[0]
                    distance_filtered = distance_filtered[distance_filtered['Klaster_Terpilih'] == cluster_num]
                
                if distance_display_mode == "Sample Random":
                    if len(distance_filtered) > distance_sample_size:
                        distance_filtered = distance_filtered.sample(n=distance_sample_size, random_state=42)
                else:
                    distance_filtered = distance_filtered.head(distance_show_rows)
                
                if distance_sort_by != "No":
                    distance_filtered = distance_filtered.sort_values(distance_sort_by, ascending=True)
                
                # Reset index and add sequential number
                distance_filtered = distance_filtered.reset_index(drop=True)
                distance_filtered['No'] = range(1, len(distance_filtered) + 1)
                
                # Remove cluster column for display
                distance_display = distance_filtered.drop('Klaster_Terpilih', axis=1)
                
                st.dataframe(distance_display, use_container_width=True, height=300)
                st.info(f"Menampilkan {len(distance_display)} dari {len(distance_table)} total pixel")
                
                # Section 3: Cluster Assignment
                st.markdown("### 3️⃣ Penentuan Klaster")
                st.markdown("Penentuan klaster berdasarkan jarak terdekat ke centroid")
                
                # Create cluster assignment table
                cluster_table = pd.DataFrame({
                    'No': range(1, len(classifier.features) + 1),
                    'Koordinat': [f'({y},{x})' for y, x in zip(valid_y, valid_x)],
                    'Jarak_ke_C1': distances_to_centroids[0].round(4),
                    'Jarak_ke_C2': distances_to_centroids[1].round(4),
                    'Jarak_ke_C3': distances_to_centroids[2].round(4),
                    'Jarak_ke_C4': distances_to_centroids[3].round(4),
                    'Jarak_Terdekat': np.min(distances_to_centroids, axis=0).round(4),
                    'Klaster_Terpilih': [f'C{label+1}' for label in labels],
                    'Nama_Klaster': [classifier.cluster_names[label] for label in labels]
                })
                
                # Display options for cluster table
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    cluster_display_mode = st.selectbox("Mode Tampilan:", ["Sample Random", "Semua Data"], key="cluster_mode")
                with col2:
                    if cluster_display_mode == "Sample Random":
                        cluster_sample_size = st.selectbox("Jumlah Sample:", [100, 250, 500, 1000], key="cluster_sample")
                    else:
                        cluster_show_rows = st.selectbox("Tampilkan baris:", [10, 25, 50, 100], key="cluster_rows")
                with col3:
                    cluster_filter_cluster = st.selectbox("Filter klaster:", ["Semua"] + [f"C{i+1}: {name}" for i, name in classifier.cluster_names.items()], key="cluster_filter")
                with col4:
                    cluster_sort_by = st.selectbox("Urutkan berdasarkan:", ["No", "Jarak_Terdekat", "Klaster_Terpilih"], key="cluster_sort")
                
                # Apply filtering and display for cluster table
                cluster_filtered = cluster_table.copy()
                
                if cluster_filter_cluster != "Semua":
                    cluster_num = cluster_filter_cluster.split(":")[0]
                    cluster_filtered = cluster_filtered[cluster_filtered['Klaster_Terpilih'] == cluster_num]
                
                if cluster_display_mode == "Sample Random":
                    if len(cluster_filtered) > cluster_sample_size:
                        cluster_filtered = cluster_filtered.sample(n=cluster_sample_size, random_state=42)
                else:
                    cluster_filtered = cluster_filtered.head(cluster_show_rows)
                
                if cluster_sort_by != "No":
                    cluster_filtered = cluster_filtered.sort_values(cluster_sort_by, ascending=True if cluster_sort_by == "Jarak_Terdekat" else False)
                
                # Reset index and add sequential number
                cluster_filtered = cluster_filtered.reset_index(drop=True)
                cluster_filtered['No'] = range(1, len(cluster_filtered) + 1)
                
                st.dataframe(cluster_filtered, use_container_width=True, height=300)
                st.info(f"Menampilkan {len(cluster_filtered)} dari {len(cluster_table)} total pixel")
                
                # Section 4: Result Interpretation
                st.markdown("### 4️⃣ Interpretasi Hasil")
                st.markdown("Analisis karakteristik dan validasi hasil klasifikasi")
                
                # Create interpretation table with all features
                interpretation_table = pd.DataFrame({
                    'No': range(1, len(classifier.features) + 1),
                    'Koordinat': [f'({y},{x})' for y, x in zip(valid_y, valid_x)],
                    'Red_B4': classifier.features[:, 0].round(4),
                    'NIR_B8': classifier.features[:, 1].round(4),
                    'NDVI': classifier.features[:, 2].round(4),
                    'NIR_Red_Ratio': (classifier.features[:, 1] / (classifier.features[:, 0] + 1e-8)).round(4),
                    'Klaster_Terpilih': [f'C{label+1}' for label in labels],
                    'Nama_Klaster': [classifier.cluster_names[label] for label in labels],
                    'Jarak_Terdekat': np.min(distances_to_centroids, axis=0).round(4)
                })
                
                # Add interpretation column
                def get_interpretation(row):
                    ndvi = row['NDVI']
                    nir_red_ratio = row['NIR_Red_Ratio']
                    cluster = row['Nama_Klaster']
                    
                    if cluster == "Air":
                        if ndvi < 0.1 and nir_red_ratio < 1.5:
                            return "✅ Sesuai karakteristik air"
                        else:
                            return "⚠️ Karakteristik tidak khas air"
                    elif cluster == "Bangunan/Jalan":
                        if ndvi < 0.4 and nir_red_ratio < 2.0:
                            return "✅ Sesuai karakteristik bangunan"
                        else:
                            return "⚠️ Karakteristik tidak khas bangunan"
                    elif cluster == "Lahan Terbuka":
                        if -0.2 <= ndvi <= 0.5:
                            return "✅ Sesuai karakteristik lahan terbuka"
                        else:
                            return "⚠️ Karakteristik tidak khas lahan terbuka"
                    elif cluster == "Vegetasi":
                        if ndvi > 0.2 and nir_red_ratio > 1.5:
                            return "✅ Sesuai karakteristik vegetasi"
                        else:
                            return "⚠️ Karakteristik tidak khas vegetasi"
                    return "❓ Tidak dapat diinterpretasi"
                
                interpretation_table['Interpretasi'] = interpretation_table.apply(get_interpretation, axis=1)
                
                # Display options for interpretation table
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    interp_display_mode = st.selectbox("Mode Tampilan:", ["Sample Random", "Semua Data"], key="interp_mode")
                with col2:
                    if interp_display_mode == "Sample Random":
                        interp_sample_size = st.selectbox("Jumlah Sample:", [100, 250, 500, 1000], key="interp_sample")
                    else:
                        interp_show_rows = st.selectbox("Tampilkan baris:", [10, 25, 50, 100], key="interp_rows")
                with col3:
                    interp_filter_cluster = st.selectbox("Filter klaster:", ["Semua"] + [f"C{i+1}: {name}" for i, name in classifier.cluster_names.items()], key="interp_filter")
                with col4:
                    interp_sort_by = st.selectbox("Urutkan berdasarkan:", ["No", "NDVI", "NIR_Red_Ratio", "Jarak_Terdekat"], key="interp_sort")
                
                # Apply filtering and display for interpretation table
                interp_filtered = interpretation_table.copy()
                
                if interp_filter_cluster != "Semua":
                    cluster_num = interp_filter_cluster.split(":")[0]
                    interp_filtered = interp_filtered[interp_filtered['Klaster_Terpilih'] == cluster_num]
                
                if interp_display_mode == "Sample Random":
                    if len(interp_filtered) > interp_sample_size:
                        interp_filtered = interp_filtered.sample(n=interp_sample_size, random_state=42)
                else:
                    interp_filtered = interp_filtered.head(interp_show_rows)
                
                if interp_sort_by != "No":
                    interp_filtered = interp_filtered.sort_values(interp_sort_by, ascending=False if interp_sort_by in ["NDVI", "NIR_Red_Ratio"] else True)
                
                # Reset index and add sequential number
                interp_filtered = interp_filtered.reset_index(drop=True)
                interp_filtered['No'] = range(1, len(interp_filtered) + 1)
                
                st.dataframe(interp_filtered, use_container_width=True, height=300)
                st.info(f"Menampilkan {len(interp_filtered)} dari {len(interpretation_table)} total pixel")
                
                # Create comprehensive manual table for export
                manual_table = pd.DataFrame({
                    'No': range(1, len(classifier.features) + 1),
                    'Pixel_Y': valid_y,
                    'Pixel_X': valid_x,
                    'Koordinat': [f'({y},{x})' for y, x in zip(valid_y, valid_x)],
                    'Red_B4': classifier.features[:, 0].round(4),
                    'NIR_B8': classifier.features[:, 1].round(4),
                    'NDVI': classifier.features[:, 2].round(4),
                    'NIR_Red_Ratio': (classifier.features[:, 1] / (classifier.features[:, 0] + 1e-8)).round(4),
                    'Jarak_ke_C1': distances_to_centroids[0].round(4),
                    'Jarak_ke_C2': distances_to_centroids[1].round(4),
                    'Jarak_ke_C3': distances_to_centroids[2].round(4),
                    'Jarak_ke_C4': distances_to_centroids[3].round(4),
                    'Jarak_Terdekat': np.min(distances_to_centroids, axis=0).round(4),
                    'Klaster_Terpilih': [f'C{label+1}' for label in labels],
                    'Nama_Klaster': [classifier.cluster_names[label] for label in labels]
                })
                
                st.success(f"✅ Semua tahap perhitungan manual selesai dengan {len(manual_table)} data pixel")
                
                # Summary statistics for the manual calculation
                st.subheader("📊 Ringkasan Perhitungan Manual")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Statistik Fitur:**")
                    feature_stats = pd.DataFrame({
                        'Fitur': ['Red (B4)', 'NIR (B8)', 'NDVI', 'NIR/Red Ratio'],
                        'Min': [
                            manual_table['Red_B4'].min(),
                            manual_table['NIR_B8'].min(), 
                            manual_table['NDVI'].min(),
                            manual_table['NIR_Red_Ratio'].min()
                        ],
                        'Max': [
                            manual_table['Red_B4'].max(),
                            manual_table['NIR_B8'].max(),
                            manual_table['NDVI'].max(), 
                            manual_table['NIR_Red_Ratio'].max()
                        ],
                        'Mean': [
                            manual_table['Red_B4'].mean(),
                            manual_table['NIR_B8'].mean(),
                            manual_table['NDVI'].mean(),
                            manual_table['NIR_Red_Ratio'].mean()
                        ],
                        'Std': [
                            manual_table['Red_B4'].std(),
                            manual_table['NIR_B8'].std(),
                            manual_table['NDVI'].std(),
                            manual_table['NIR_Red_Ratio'].std()
                        ]
                    })
                    st.dataframe(feature_stats.round(4), use_container_width=True)
                
                with col2:
                    st.markdown("**Distribusi Jarak ke Centroid:**")
                    distance_stats = pd.DataFrame({
                        'Centroid': ['C1: Air', 'C2: Bangunan/Jalan', 'C3: Lahan Terbuka', 'C4: Vegetasi'],
                        'Jarak_Min': [
                            manual_table['Jarak_ke_C1'].min(),
                            manual_table['Jarak_ke_C2'].min(),
                            manual_table['Jarak_ke_C3'].min(),
                            manual_table['Jarak_ke_C4'].min()
                        ],
                        'Jarak_Max': [
                            manual_table['Jarak_ke_C1'].max(),
                            manual_table['Jarak_ke_C2'].max(),
                            manual_table['Jarak_ke_C3'].max(),
                            manual_table['Jarak_ke_C4'].max()
                        ],
                        'Jarak_Mean': [
                            manual_table['Jarak_ke_C1'].mean(),
                            manual_table['Jarak_ke_C2'].mean(),
                            manual_table['Jarak_ke_C3'].mean(),
                            manual_table['Jarak_ke_C4'].mean()
                        ]
                    })
                    st.dataframe(distance_stats.round(4), use_container_width=True)
                
                # Export functionality
                st.subheader("💾 Export Data")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Export to CSV
                    csv_data = manual_table.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name="hasil_klasifikasi_vegetasi_manual.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Export centroid data
                    centroid_export = pd.DataFrame(
                        centroids,
                        columns=['Red_B4', 'NIR_B8', 'NDVI'],
                        index=[f'C{i+1}_{name}' for i, name in classifier.cluster_names.items()]
                    )
                    centroid_csv = centroid_export.to_csv()
                    st.download_button(
                        label="📥 Download Centroid CSV",
                        data=centroid_csv,
                        file_name="centroid_klaster.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    # Export summary statistics
                    summary_data = {
                        'Total_Pixel': len(manual_table),
                        'Silhouette_Score': evaluation_results['silhouette_avg'] if evaluation_results else 'N/A',
                        'Iterasi_KMeans': classifier.model.n_iter_,
                        'Random_State': random_state
                    }
                    summary_df = pd.DataFrame([summary_data])
                    summary_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary CSV",
                        data=summary_csv,
                        file_name="ringkasan_klasifikasi.csv",
                        mime="text/csv"
                    )
                
                # Detailed explanation of manual calculation
                st.subheader("📖 Penjelasan Perhitungan Manual")
                
                with st.expander("Klik untuk melihat detail perhitungan"):
                    st.markdown("""
                    **Tahapan Perhitungan Manual:**
                    
                    1. **Ekstraksi Fitur per Pixel:**
                       - Red (B4): Nilai reflektansi band merah
                       - NIR (B8): Nilai reflektansi band near-infrared
                       - NDVI: (NIR - Red) / (NIR + Red)
                       - NIR/Red Ratio: NIR / Red (indikator vegetasi)
                    
                    2. **Perhitungan Jarak Euclidean:**
                       - Untuk setiap pixel, hitung jarak ke semua centroid
                       - Rumus: √[(x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²]
                       - Dimana x=Red, y=NIR, z=NDVI
                    
                    3. **Penentuan Klaster:**
                       - Pixel diklasifikasikan ke klaster dengan jarak terdekat
                       - Jarak terkecil menentukan keanggotaan klaster
                    
                    4. **Interpretasi Hasil:**
                       - C1 (Air): NDVI rendah, NIR/Red ratio < 1.5
                       - C2 (Bangunan/Jalan): NDVI sedang, reflektansi seimbang
                       - C3 (Lahan Terbuka): NDVI bervariasi, karakteristik campuran
                       - C4 (Vegetasi): NDVI tinggi, NIR/Red ratio > 1.5
                    """)
                
                # Informasi evaluasi
                st.subheader("✅ Hasil Evaluasi")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if evaluation_results:
                        st.metric("Silhouette Score", f"{evaluation_results['silhouette_avg']:.4f}")
                    else:
                        st.metric("Silhouette Score", "N/A")
                with col2:
                    st.metric("Jumlah Klaster", n_clusters)
                with col3:
                    st.metric("Total Iterasi", classifier.model.n_iter_)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Klasifikasi Tutupan Lahan**")
st.sidebar.markdown("4 Klaster Utama:")
st.sidebar.markdown("🔵 Air | 🔴 Bangunan/Jalan")
st.sidebar.markdown("🟠 Lahan Terbuka | 🟢 Vegetasi")

if __name__ == "__main__":
    main()