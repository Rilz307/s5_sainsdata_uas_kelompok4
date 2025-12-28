import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Dashboard Adiwiyata & Sampah",
    page_icon="🌿",
    layout="wide"
)

# --- CSS SEDERHANA UNTUK MEMPERCANTIK MENU ---
st.markdown("""
<style>
    /* 1. Menghilangkan bulatan radio button standar */
    div.stRadio > div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    /* 2. Mengubah label radio button menjadi tombol full-width */
    div.stRadio > div[role="radiogroup"] > label {
        display: block;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 5px;
        border: 1px solid transparent;
        transition: all 0.3s;
        cursor: pointer;
    }

    /* 3. Efek Hover (Saat mouse diarahkan) - Dark Mode Friendly */
    div.stRadio > div[role="radiogroup"] > label:hover {
        background-color: rgba(255, 255, 255, 0.1); /* Putih transparan halus */
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* 4. Efek Selected (Saat menu dipilih) - Highlight Biru */
    div.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
        background-color: rgba(0, 123, 255, 0.2); /* Biru transparan */
        border-color: #007bff;
        color: #007bff !important;
        font-weight: bold;
    }
    
    /* SAYA MENGHAPUS BAGIAN [data-testid="stSidebar"] AGAR BACKGROUND MENGIKUTI TEMA (GELAP) */
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI UTILITY (CLEANING DATA)
# ==========================================
def normalize_kabkot_sekolah(series):
    """Normalisasi nama kabupaten/kota dari dataset sekolah"""
    s = (series.astype(str)
         .str.replace('\xa0', ' ', regex=False)
         .str.replace(r'\s+', ' ', regex=True)
         .str.strip().str.upper())
    
    is_kota = s.str.match(r'^KOTA\b')
    nama_inti = (s.str.replace(r'^(KOTA|KAB\.?|KABUPATEN)\b', '', regex=True)
                 .str.strip().str.title())
    
    hasil = nama_inti.where(is_kota, "Kab. " + nama_inti)
    hasil = hasil.where(~is_kota, "Kota " + nama_inti)
    return hasil

# ==========================================
# 3. SIDEBAR MENU (SIMPEL & CANTIK)
# ==========================================
with st.sidebar:
    st.title("🌿 Eco-School Analysis")
    st.markdown("Dashboard Analisis Hubungan Sekolah Adiwiyata & Kualitas Lingkungan.")
    st.divider()
    
    # Menu Navigasi Native (Aman dari Error)
    menu = st.radio(
        "Navigasi Menu:",
        ["1. Dataset Overview", "2. EDA Lengkap", "3. Modelling"],
        index=0
    )
    
    st.divider()

# ==========================================
# 4. LOGIKA KONTEN UTAMA
# ==========================================


# ==========================================
# FUNGSI NORMALISASI (PERBAIKAN DOUBLE TITIK)
# ==========================================
def normalize_kabkot_sekolah(series):
    """
    Normalisasi nama kabupaten/kota (Versi Fix Double Dot)
    """
    # 1. Bersihkan karakter aneh & spasi berlebih
    s = (
        series
        .astype(str)
        .str.replace('\xa0', ' ', regex=False)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.upper()
    )

    # 2. Cek apakah dia Kota atau Kabupaten
    is_kota = s.str.match(r'^KOTA\b')

    # 3. Ambil nama intinya saja
    # PERBAIKAN: Regex sekarang memakan spasi & titik setelah KAB/KOTA
    nama_inti = (
        s
        .str.replace(r'^(KOTA|KAB\.?|KABUPATEN)\s*\.?\s*', '', regex=True)
        .str.strip()
        .str.lstrip('.') # Hapus paksa titik di depan jika masih ada sisa
        .str.strip()
        .str.title()
    )

    # 4. Format ulang jadi "Kab. X" atau "Kota Y"
    hasil = nama_inti.where(is_kota, "Kab. " + nama_inti)
    hasil = hasil.where(~is_kota, "Kota " + nama_inti)
    
    # 5. Handle kasus khusus (misal "Kab. -")
    hasil = hasil.replace("Kab. -", "Tidak Diketahui").replace("Kota -", "Tidak Diketahui")

    return hasil

# ==========================================
# MENU 1: DATASET OVERVIEW (STRICT LOGIC)
# ==========================================
if menu == "1. Dataset Overview":
    st.title("📂 Dataset Overview & Processing")
    st.markdown("Modul ini menjalankan **Data Preparation** persis seperti spesifikasi Notebook.")

    # Konfigurasi Path File
    BASE_DIR = "Dataset_DS"
    FILES = {
        "Sekolah": "sekolah adiwiyata - sekolah adiwiyata.csv",
        "RTH": "Data_RTH.xlsx",
        "Sampah": "Data_Timbulan_Sampah.xlsx",
        "Kualitas Air": "Indeks_Kualitas_Air.csv",
        "Kualitas Udara": "indeks_kualitas_udara.csv"
    }

    # Cek Ketersediaan File
    cols = st.columns(len(FILES))
    file_paths = {}
    all_files_exist = True
    
    for i, (label, filename) in enumerate(FILES.items()):
        full_path = os.path.join(BASE_DIR, filename)
        file_paths[label] = full_path
        with cols[i]:
            if os.path.exists(full_path):
                st.success(f"✅ {label}")
            else:
                st.error(f"❌ {label}")
                all_files_exist = False

    st.divider()

    if all_files_exist:
        if st.button("🚀 Jalankan Data Preparation (Sesuai Notebook)", type="primary", use_container_width=True):
            with st.spinner("Sedang memproses data..."):
                try:
                    # 1. LOAD DATA
                    df_sekolah = pd.read_csv(file_paths["Sekolah"])
                    df_rth = pd.read_excel(file_paths["RTH"])
                    df_sampah = pd.read_excel(file_paths["Sampah"])

                    # 2. PROSES DATA SEKOLAH
                    # Pastikan nama kolom 'Kabupaten/Kota' ada (Mapping dari file asli)
                    col_kab = [c for c in df_sekolah.columns if 'Kabupaten' in c]
                    if col_kab: df_sekolah.rename(columns={col_kab[0]: 'Kabupaten/Kota'}, inplace=True)

                    # Terapkan fungsi normalisasi
                    df_sekolah["KABKOT_STD"] = normalize_kabkot_sekolah(df_sekolah["Kabupaten/Kota"])

                    # Agregasi
                    df_sekolah_wilayah = (
                        df_sekolah
                        .groupby("KABKOT_STD", as_index=False)
                        .agg(
                            JUMLAH_SEKOLAH_ADIWIYATA=("Nama Sekolah", "count")
                        )
                    )

                    # 3. PROSES DATA RTH
                    df_rth_clean = df_rth.copy()
                    
                    # Rename (Sesuai Notebook)
                    # Catatan: Kita pakai strip() dulu jaga-jaga ada spasi di header excel asli
                    df_rth_clean.columns = df_rth_clean.columns.str.strip()
                    df_rth_clean = df_rth_clean.rename(columns={
                        "Kabupaten/Kota": "KABKOT_STD",
                        "Luas Wilayah (km2)(A)": "LUAS_WILAYAH",
                        "% RTH(B/A)": "PERSEN_RTH"
                    })

                    # Cleaning Numerik (Sesuai Notebook)
                    for col in ["LUAS_WILAYAH", "PERSEN_RTH"]:
                        if col in df_rth_clean.columns:
                            df_rth_clean[col] = (
                                df_rth_clean[col]
                                .astype(str)
                                .str.replace(",", ".", regex=False)
                                .replace("-", np.nan)
                            )
                            df_rth_clean[col] = pd.to_numeric(df_rth_clean[col], errors="coerce")

                    # Drop NA pada Luas Wilayah (Sesuai Notebook)
                    df_rth_clean = df_rth_clean.dropna(subset=["LUAS_WILAYAH"])

                    # Sorting & Drop Duplicates (Ambil data terbaru)
                    if "Tahun" in df_rth_clean.columns:
                        df_rth_wilayah = (
                            df_rth_clean
                            .sort_values(["KABKOT_STD", "Tahun"], ascending=[True, False])
                            .drop_duplicates("KABKOT_STD")
                            [["KABKOT_STD", "PERSEN_RTH", "LUAS_WILAYAH"]]
                        )
                    else:
                        # Fallback jika tidak ada kolom tahun (ambil unique pertama)
                        df_rth_wilayah = df_rth_clean.drop_duplicates("KABKOT_STD")[["KABKOT_STD", "PERSEN_RTH", "LUAS_WILAYAH"]]

                    # 4. PROSES DATA SAMPAH
                    df_sampah_clean = df_sampah.copy()
                    df_sampah_clean.columns = df_sampah_clean.columns.str.strip()
                    
                    df_sampah_clean = df_sampah_clean.rename(columns={
                        "Kabupaten/Kota": "KABKOT_STD",
                        "Timbulan Sampah Harian(ton)": "SAMPAH_HARIAN_TON",
                        "Timbulan Sampah Tahunan(ton)": "SAMPAH_TAHUNAN_TON"
                    })

                    # Sorting & Drop Duplicates (Ambil data terbaru)
                    if "Tahun" in df_sampah_clean.columns:
                        df_sampah_wilayah = (
                            df_sampah_clean
                            .sort_values(["KABKOT_STD", "Tahun"], ascending=[True, False])
                            .drop_duplicates("KABKOT_STD")
                            [["KABKOT_STD", "SAMPAH_HARIAN_TON", "SAMPAH_TAHUNAN_TON"]]
                        )
                    else:
                        df_sampah_wilayah = df_sampah_clean.drop_duplicates("KABKOT_STD")[["KABKOT_STD", "SAMPAH_HARIAN_TON", "SAMPAH_TAHUNAN_TON"]]

                    # 5. PENGGABUNGAN (MERGE)
                    df_final = (
                        df_sekolah_wilayah
                        .merge(df_rth_wilayah, on="KABKOT_STD", how="left")
                        .merge(df_sampah_wilayah, on="KABKOT_STD", how="left")
                    )

                    # Simpan ke Session
                    st.session_state["df_final"] = df_final
                    st.toast("Data berhasil diproses sesuai Notebook!", icon="✅")
                
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
                    st.stop()

        # Tampilkan Hasil jika data sudah ada
        if "df_final" in st.session_state:
            df_final = st.session_state["df_final"]
            
            st.subheader("📊 Hasil Akhir Data Preparation")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Wilayah", f"{df_final['KABKOT_STD'].nunique()}")
            m2.metric("Total Sekolah", f"{df_final['JUMLAH_SEKOLAH_ADIWIYATA'].sum():,.0f}")
            m3.metric("Rata-rata RTH", f"{df_final['PERSEN_RTH'].mean():.2f}%")
            m4.metric("Total Sampah Harian", f"{df_final['SAMPAH_HARIAN_TON'].sum():,.0f} Ton")

            with st.expander("🔍 Preview Tabel (5 Baris Teratas)", expanded=True):
                st.dataframe(df_final.head(), use_container_width=True)

    else:
        st.warning("⚠️ File dataset tidak lengkap di folder `Dataset_DS`.")

# ==========================================
# MENU 2: EDA LENGKAP (FULL REPLIKASI NOTEBOOK)
# ==========================================
elif menu == "2. EDA Lengkap":
    # 1. Cek Data di Session
    if "df_final" not in st.session_state:
        st.warning("⚠️ Data belum tersedia. Silakan kembali ke menu **'1. Dataset Overview'** dan klik tombol **'🚀 Jalankan Data Preparation'**.")
        st.stop()
    
    # 2. Ambil Data
    df_final = st.session_state["df_final"].copy()
    
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Analisis karakteristik data, transformasi, dan hubungan antar variabel.")

    # --- PREPROCESSING KOLOM (Agar sesuai codingan Notebook) ---
    # 1. Upper case & Strip
    df_final.columns = df_final.columns.str.upper().str.strip()

    # 2. Mapping Nama Kolom (Agar cocok dengan variabel di plotting code)
    # Ini penting agar tidak error KeyError
    rename_map = {
        "RTH_PERSEN": "PERSEN_RTH",
        "% RTH": "PERSEN_RTH",
        "LUAS_WILAYAH_KM2": "LUAS_WILAYAH",
        "TIMBULAN_SAMPAH_HARIAN": "SAMPAH_HARIAN_TON",
        "TIMBULAN_SAMPAH_TAHUNAN": "SAMPAH_TAHUNAN_TON"
    }
    df_final.rename(columns=rename_map, inplace=True)

    # 3. Definisi Kolom Numerik Utama
    numerical_cols = [
        "JUMLAH_SEKOLAH_ADIWIYATA",
        "PERSEN_RTH",
        "LUAS_WILAYAH",
        "SAMPAH_HARIAN_TON",
        "SAMPAH_TAHUNAN_TON"
    ]

    # Validasi Kolom
    missing = [c for c in numerical_cols if c not in df_final.columns]
    if missing:
        st.error(f"❌ Kolom berikut hilang dari dataset: {missing}")
        st.stop()

    # --- TABS VISUALISASI ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Statistik & Distribusi",
        "2. Log Transform",
        "3. Korelasi",
        "4. Normalisasi Wilayah",
        "5. Visualisasi Lanjutan"
    ])

    # ========================================================
    # TAB 1: STATISTIK & DISTRIBUSI
    # ========================================================
    with tab1:
        st.subheader("Statistik Deskriptif")
        st.dataframe(
            df_final[numerical_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]),
            use_container_width=True
        )

        st.subheader("Distribusi Awal")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                sns.histplot(df_final[col], kde=True, ax=axes[i], bins=30)
                axes[i].set_title(f"Distribusi {col}")
                axes[i].set_xlabel("")
                axes[i].set_ylabel("")

        for j in range(len(numerical_cols), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Boxplot Data Asli")
        fig_box = plt.figure(figsize=(14, 6))
        sns.boxplot(data=df_final[numerical_cols], orient="h")
        plt.title("Boxplot Seluruh Variabel Numerik")
        st.pyplot(fig_box)

    # ========================================================
    # TAB 2: LOG TRANSFORM
    # ========================================================
    with tab2:
        st.subheader("Log Transform")
        st.markdown("Mengurangi skewness menggunakan `np.log1p`.")

        # Proses Transformasi
        df_stat = df_final.copy()
        log_transform_cols = numerical_cols
        
        for col in log_transform_cols:
            df_stat[f"LOG_{col}"] = np.log1p(df_stat[col])

        # Plot Perbandingan (Before vs After)
        st.write("Perbandingan Distribusi: Original vs Log")
        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 20))

        for i, col in enumerate(log_transform_cols):
            # BEFORE
            sns.histplot(df_stat[col], bins=30, kde=True, ax=axes[i, 0], color='skyblue')
            axes[i, 0].set_title(f"Sebelum: {col}")
            axes[i, 0].set_xlabel("")
            
            # AFTER
            sns.histplot(df_stat[f"LOG_{col}"], bins=30, kde=True, ax=axes[i, 1], color='orange')
            axes[i, 1].set_title(f"Sesudah: LOG_{col}")
            axes[i, 1].set_xlabel("")

        plt.tight_layout()
        st.pyplot(fig)

        # Boxplot Log Only
        st.subheader("Boxplot Data Log")
        log_cols_only = [f"LOG_{c}" for c in log_transform_cols]
        fig_box_log = plt.figure(figsize=(14, 6))
        sns.boxplot(data=df_stat[log_cols_only], orient="h")
        plt.title("Boxplot Variabel Setelah Log Transform")
        st.pyplot(fig_box_log)

    # ========================================================
    # TAB 3: KORELASI
    # ========================================================
    with tab3:
        st.subheader("Matriks Korelasi (Data Log)")
        
        corr_cols = [
            "LOG_JUMLAH_SEKOLAH_ADIWIYATA",
            "LOG_PERSEN_RTH",
            "LOG_SAMPAH_HARIAN_TON",
            "LOG_SAMPAH_TAHUNAN_TON"
        ]

        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**1. Spearman Correlation**")
            corr_spearman = df_stat[corr_cols].corr(method="spearman")
            fig_s = plt.figure(figsize=(8, 6))
            sns.heatmap(corr_spearman, annot=True, cmap="coolwarm", center=0, fmt=".2f")
            plt.title("Matriks Korelasi Spearman")
            st.pyplot(fig_s)

        with c2:
            st.markdown("**2. Pearson Correlation**")
            corr_pearson = df_stat[corr_cols].corr(method="pearson")
            fig_p = plt.figure(figsize=(8, 6))
            sns.heatmap(corr_pearson, annot=True, cmap="coolwarm", center=0, fmt=".2f")
            plt.title("Matriks Korelasi Pearson")
            st.pyplot(fig_p)

    # ========================================================
    # TAB 4: NORMALISASI WILAYAH
    # ========================================================
    with tab4:
        st.subheader("Normalisasi Berbasis Luas Wilayah")
        st.markdown("Menghitung densitas per km².")
        
        # Proses Normalisasi
        df_norm = df_stat.copy()
        df_norm = df_norm[df_norm["LUAS_WILAYAH"] > 0].copy() # Filter luas 0

        # Rasio
        df_norm["ADIWIYATA_PER_KM2"] = df_norm["JUMLAH_SEKOLAH_ADIWIYATA"] / df_norm["LUAS_WILAYAH"]
        df_norm["SAMPAH_HARIAN_PER_KM2"] = df_norm["SAMPAH_HARIAN_TON"] / df_norm["LUAS_WILAYAH"]
        df_norm["SAMPAH_TAHUNAN_PER_KM2"] = df_norm["SAMPAH_TAHUNAN_TON"] / df_norm["LUAS_WILAYAH"]

        # Log Transform Hasil Rasio
        df_norm["LOG_ADIWIYATA_PER_KM2"] = np.log1p(df_norm["ADIWIYATA_PER_KM2"])
        df_norm["LOG_SAMPAH_HARIAN_PER_KM2"] = np.log1p(df_norm["SAMPAH_HARIAN_PER_KM2"])
        df_norm["LOG_SAMPAH_TAHUNAN_PER_KM2"] = np.log1p(df_norm["SAMPAH_TAHUNAN_PER_KM2"])

        # Plot Distribusi Normalisasi
        norm_cols = ["LOG_ADIWIYATA_PER_KM2", "LOG_SAMPAH_HARIAN_PER_KM2", "LOG_SAMPAH_TAHUNAN_PER_KM2"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, col in enumerate(norm_cols):
            sns.histplot(df_norm[col], bins=30, kde=True, ax=axes[i], color='green')
            axes[i].set_title(f"Distribusi {col}")
            axes[i].set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig)

        # Boxplot Normalisasi
        fig_bn = plt.figure(figsize=(10, 4))
        sns.boxplot(data=df_norm[norm_cols], orient="h")
        plt.title("Boxplot Variabel Normalisasi (Log)")
        st.pyplot(fig_bn)

        # Korelasi Normalisasi
        st.markdown("#### Korelasi Data Ternormalisasi")
        c3, c4 = st.columns(2)
        with c3:
            st.write("Spearman")
            corr_ns = df_norm[norm_cols].corr(method="spearman")
            fig_ns = plt.figure(figsize=(6, 5))
            sns.heatmap(corr_ns, annot=True, cmap="coolwarm", center=0, fmt=".2f")
            st.pyplot(fig_ns)
        with c4:
            st.write("Pearson")
            corr_np = df_norm[norm_cols].corr(method="pearson")
            fig_np = plt.figure(figsize=(6, 5))
            sns.heatmap(corr_np, annot=True, cmap="coolwarm", center=0, fmt=".2f")
            st.pyplot(fig_np)

    # ========================================================
    # TAB 5: VISUALISASI LANJUTAN
    # ========================================================
    with tab5:
        st.subheader("Visualisasi Hubungan Lanjutan")

        # Scatter Plots
        c_sc1, c_sc2 = st.columns(2)
        with c_sc1:
            st.caption("Adiwiyata vs Sampah Harian")
            fig_sc1 = plt.figure(figsize=(7, 5))
            sns.scatterplot(data=df_norm, x="LOG_ADIWIYATA_PER_KM2", y="LOG_SAMPAH_HARIAN_PER_KM2", alpha=0.6)
            sns.regplot(data=df_norm, x="LOG_ADIWIYATA_PER_KM2", y="LOG_SAMPAH_HARIAN_PER_KM2", scatter=False, color="red")
            plt.title("Adiwiyata vs Sampah Harian per km² (Log)")
            st.pyplot(fig_sc1)
        
        with c_sc2:
            st.caption("Adiwiyata vs Sampah Tahunan")
            fig_sc2 = plt.figure(figsize=(7, 5))
            sns.scatterplot(data=df_norm, x="LOG_ADIWIYATA_PER_KM2", y="LOG_SAMPAH_TAHUNAN_PER_KM2", alpha=0.6)
            sns.regplot(data=df_norm, x="LOG_ADIWIYATA_PER_KM2", y="LOG_SAMPAH_TAHUNAN_PER_KM2", scatter=False, color="red")
            plt.title("Adiwiyata vs Sampah Tahunan per km² (Log)")
            st.pyplot(fig_sc2)

        st.divider()

        # DOSE RESPONSE ANALYSIS
        st.subheader("Analisis Dose-Response")
        
        # Siapkan data Dose Response
        df_dr = df_norm.copy()
        
        try:
            # 1. Binning 5 Level
            df_dr["BIN_ADIWIYATA"] = pd.qcut(
                df_dr["LOG_ADIWIYATA_PER_KM2"], q=5,
                labels=["Sangat Rendah", "Rendah", "Menengah", "Tinggi", "Sangat Tinggi"],
                duplicates='drop'
            )
            
            # 2. Aggregasi
            dose_plot = (
                df_dr.groupby("BIN_ADIWIYATA", observed=False)
                .agg(
                    median=("LOG_SAMPAH_HARIAN_PER_KM2", "median"),
                    q25=("LOG_SAMPAH_HARIAN_PER_KM2", lambda x: x.quantile(0.25)),
                    q75=("LOG_SAMPAH_HARIAN_PER_KM2", lambda x: x.quantile(0.75))
                )
                .reset_index()
            )

            # 3. Plot Line
            st.write("**Grafik Dose-Response**")
            fig_dr = plt.figure(figsize=(8, 5))
            plt.plot(dose_plot["BIN_ADIWIYATA"], dose_plot["median"], marker="o", label="Median")
            plt.fill_between(
                dose_plot["BIN_ADIWIYATA"], dose_plot["q25"], dose_plot["q75"],
                alpha=0.3, label="IQR (25-75%)"
            )
            plt.title("Dose-Response: Intensitas Sekolah vs Sampah")
            plt.xlabel("Tingkat Intensitas Sekolah Adiwiyata")
            plt.ylabel("Log Sampah Harian per km²")
            plt.legend()
            plt.grid(alpha=0.3)
            st.pyplot(fig_dr)

            # 4. Boxplot per Bin (Binning 4 Level)
            st.write("**Distribusi per Kategori Intensitas**")
            df_norm["BIN_ADIWIYATA_4"] = pd.qcut(
                df_norm["LOG_ADIWIYATA_PER_KM2"], q=4,
                labels=["Rendah", "Menengah", "Tinggi", "Sangat Tinggi"],
                duplicates='drop'
            )
            fig_box_dr = plt.figure(figsize=(8, 5))
            sns.boxplot(data=df_norm, x="BIN_ADIWIYATA_4", y="LOG_SAMPAH_HARIAN_PER_KM2")
            plt.title("Distribusi Sampah berdasarkan Intensitas Sekolah")
            st.pyplot(fig_box_dr)

        except Exception as e:
            st.warning(f"Tidak dapat membuat Dose-Response (Data terlalu sedikit/seragam): {e}")

        st.divider()

        # LM PLOT (SEGMENTASI)
        st.subheader("Segmentasi Pola (LM Plot)")
        st.write("Perbedaan pola pada wilayah dengan sampah Rendah vs Tinggi.")
        
        try:
            df_ctx = df_norm.copy()
            df_ctx["KELOMPOK_SAMPAH"] = pd.qcut(
                df_ctx["LOG_SAMPAH_HARIAN_PER_KM2"], q=2,
                labels=["Sampah Relatif Rendah", "Sampah Relatif Tinggi"]
            )
            
            g = sns.lmplot(
                data=df_ctx,
                x="LOG_ADIWIYATA_PER_KM2",
                y="LOG_SAMPAH_HARIAN_PER_KM2",
                hue="KELOMPOK_SAMPAH",
                scatter_kws={"alpha": 0.5},
                height=5,
                aspect=1.2
            )
            plt.title("Perbedaan Pola Hubungan")
            st.pyplot(g.fig)
        
        except Exception as e:
            st.warning(f"Gagal membuat LM Plot: {e}")

# ==========================================
# MENU 3: MODELLING (LOAD PKL VERSION)
# ==========================================
elif menu == "3. Modelling":
    st.title("🤖 Modelling: Klasifikasi Efektivitas")
    st.markdown("Menggunakan model **LightGBM Pre-trained** (`.pkl`) untuk prediksi instan tanpa training ulang.")

    # 1. Cek Data Utama
    if "df_final" not in st.session_state:
        st.warning("⚠️ Data belum tersedia. Silakan proses data di Menu 1 dulu.")
        st.stop()

    # 2. Definisi Path File
    BASE_DIR = "Dataset_DS" # Atau sesuaikan jika file pkl ada di luar folder ini
    PATH_MODEL = "model_lgbm_adiwiyata.pkl"  # <--- Pastikan nama file sama
    
    # Path file pendukung data cleaning
    PATH_IKA = os.path.join(BASE_DIR, "Indeks_Kualitas_Air.csv")
    PATH_IKU = os.path.join(BASE_DIR, "indeks_kualitas_udara.csv")
    PATH_RTH = os.path.join(BASE_DIR, "Data_RTH.xlsx") 

    # Cek Ketersediaan Model PKL
    if not os.path.exists(PATH_MODEL):
        st.error(f"❌ File Model **{PATH_MODEL}** tidak ditemukan!")
        st.info("Silakan training dulu di Notebook, simpan dengan `joblib.dump()`, lalu taruh filenya di sini.")
        st.stop()
    else:
        st.success(f"✅ Model ditemukan: `{PATH_MODEL}`")

    # 3. Tombol Prediksi
    if st.button("🚀 Load Model & Jalankan Prediksi", type="primary"):
        with st.spinner("Memuat model dan menyiapkan data..."):
            try:
                # --- A. DATA PREPARATION (Wajib dilakukan agar fitur X sama) ---
                df_final = st.session_state["df_final"].copy()
                df_model = df_final[df_final["LUAS_WILAYAH"] > 0].copy()
                df_model.columns = df_model.columns.str.upper().str.strip()
                
                # Feature Engineering (Harus SAMA PERSIS dengan saat training)
                df_model["ADIWIYATA_PER_KM2"] = df_model["JUMLAH_SEKOLAH_ADIWIYATA"] / df_model["LUAS_WILAYAH"]
                df_model["SAMPAH_HARIAN_PER_KM2"] = df_model["SAMPAH_HARIAN_TON"] / df_model["LUAS_WILAYAH"]
                df_model["SAMPAH_TAHUNAN_PER_KM2"] = df_model["SAMPAH_TAHUNAN_TON"] / df_model["LUAS_WILAYAH"]
                
                for col in ["ADIWIYATA_PER_KM2", "SAMPAH_HARIAN_PER_KM2", "SAMPAH_TAHUNAN_PER_KM2"]:
                    df_model[f"LOG_{col}"] = np.log1p(df_model[col])

                # Mapping Provinsi (Sama seperti sebelumnya)
                df_rth_raw = pd.read_excel(PATH_RTH)
                col_kab = [c for c in df_rth_raw.columns if 'Kabupaten' in c][0]
                df_rth_raw.rename(columns={col_kab: 'Kabupaten/Kota'}, inplace=True)
                df_rth_raw["KABKOT_STD"] = normalize_kabkot_sekolah(df_rth_raw["Kabupaten/Kota"])
                
                col_prov = [c for c in df_rth_raw.columns if 'Provinsi' in c or 'PROVINSI' in c][0]
                prov_map = df_rth_raw.drop_duplicates("KABKOT_STD").set_index("KABKOT_STD")[col_prov]
                df_model["PROVINSI"] = df_model["KABKOT_STD"].map(prov_map).astype(str).str.upper().str.strip()
                
                # Merge IKA/IKU
                df_ika = pd.read_csv(PATH_IKA)
                df_iku = pd.read_csv(PATH_IKU)
                df_ika.rename(columns={"Provinsi": "PROVINSI", "Indeks Kualitas Air": "IKA"}, inplace=True)
                df_iku.rename(columns={"Provinsi": "PROVINSI", "Indeks Kualitas Udara": "IKU"}, inplace=True)
                
                for df in [df_ika, df_iku]: 
                    if "PROVINSI" in df.columns: df["PROVINSI"] = df["PROVINSI"].astype(str).str.upper().str.strip()
                
                df_model = df_model.merge(df_ika[["PROVINSI", "IKA"]], on="PROVINSI", how="left") \
                                   .merge(df_iku[["PROVINSI", "IKU"]], on="PROVINSI", how="left")
                
                df_model_clean = df_model.dropna(subset=["IKA", "IKU", "LOG_ADIWIYATA_PER_KM2"])
                
                # Buat Label (Hanya untuk evaluasi akurasi)
                median_adiwiyata = df_model_clean["LOG_ADIWIYATA_PER_KM2"].median()
                median_ika = df_model_clean["IKA"].median()
                median_iku = df_model_clean["IKU"].median()
                
                df_model_clean["ADIWIYATA_TINGGI"] = df_model_clean["LOG_ADIWIYATA_PER_KM2"] >= median_adiwiyata
                df_model_clean["LINGKUNGAN_RENDAH"] = (df_model_clean["IKA"] < median_ika) | (df_model_clean["IKU"] < median_iku)
                df_model_clean["KETIDAKSESUAIAN"] = (df_model_clean["ADIWIYATA_TINGGI"] & df_model_clean["LINGKUNGAN_RENDAH"]).astype(int)

                # --- B. LOAD MODEL & PREDICT ---
                # Siapkan Fitur X
                features = ["LOG_ADIWIYATA_PER_KM2", "LOG_SAMPAH_HARIAN_PER_KM2", "LOG_SAMPAH_TAHUNAN_PER_KM2", "PERSEN_RTH", "LUAS_WILAYAH"]
                X = df_model_clean[features]
                y_actual = df_model_clean["KETIDAKSESUAIAN"]

                # Load .pkl
                model = joblib.load(PATH_MODEL)
                
                # Lakukan Prediksi
                y_pred = model.predict(X)
                
                # Hitung Akurasi (Pada seluruh data yang ada)
                acc = accuracy_score(y_actual, y_pred)
                
                # Simpan Hasil
                st.session_state["pkl_results"] = {
                    "accuracy": acc,
                    "report": classification_report(y_actual, y_pred, target_names=["Selaras", "Tdk Selaras"], output_dict=True),
                    "feature_importances": model.feature_importances_,
                    "feature_names": features,
                    "y_actual": y_actual,
                    "y_pred": y_pred,
                    "df_model": df_model_clean
                }
                st.toast("Model berhasil dimuat & prediksi selesai!", icon="⚡")

            except Exception as e:
                st.error(f"Gagal memuat model: {e}")
                st.stop()

    # 4. Tampilkan Hasil (Sama seperti sebelumnya)
    if "pkl_results" in st.session_state:
        res = st.session_state["pkl_results"]
        
        st.divider()
        st.subheader("📊 Hasil Evaluasi Model (Pre-trained)")
        
        c1, c2 = st.columns(2)
        c1.metric("Akurasi Total", f"{res['accuracy']:.2%}", help="Akurasi pada data saat ini")
        c2.metric("Jumlah Data", f"{len(res['y_actual'])} Wilayah")

        tab_eval, tab_feat, tab_dist = st.tabs(["Confusion Matrix", "Feature Importance", "Distribusi Prediksi"])
        
        with tab_eval:
            c_cm, c_rep = st.columns([1, 2])
            with c_cm:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(res['y_actual'], res['y_pred'])
                fig_cm = plt.figure(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Selaras", "Tdk Selaras"], yticklabels=["Selaras", "Tdk Selaras"])
                st.pyplot(fig_cm)
            with c_rep:
                st.write("**Classification Report**")
                rep_df = pd.DataFrame(res['report']).transpose()
                st.dataframe(rep_df.style.format("{:.2f}"), use_container_width=True)

        with tab_feat:
            st.write("**Faktor Paling Berpengaruh**")
            imp_df = pd.DataFrame({
                "Fitur": res['feature_names'],
                "Importance": res['feature_importances']
            }).sort_values(by="Importance", ascending=False)
            
            fig_imp = plt.figure(figsize=(8, 4))
            sns.barplot(data=imp_df, x="Importance", y="Fitur", palette="viridis")
            st.pyplot(fig_imp)
            
        with tab_dist:
            st.write("**Proporsi Kelas Hasil Prediksi**")
            df_res = pd.DataFrame({"Prediksi": res['y_pred']})
            count_data = df_res["Prediksi"].value_counts().reset_index()
            count_data.columns = ["Label", "Jumlah"]
            count_data["Keterangan"] = count_data["Label"].map({0: "Relatif Selaras", 1: "Tidak Selaras"})
            fig_dist = plt.figure(figsize=(6, 4))
            sns.barplot(data=count_data, x="Keterangan", y="Jumlah", palette="pastel")
            st.pyplot(fig_dist)