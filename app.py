import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

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
    # TAB 1: STATISTIK & DISTRIBUSI (EXECUTIVE SUMMARY STYLE)
    # ========================================================
    with tab1:
        st.subheader("1. Snapshot Eksekutif")
        st.markdown("Ringkasan cepat kondisi data saat ini.")

        # --- A. KEY METRICS (Agar user langsung tahu angka penting) ---
        # Kita ambil rata-rata dan nilai maksimum untuk memberi konteks
        col_m1, col_m2, col_m3 = st.columns(3)
        
        avg_sekolah = df_final["JUMLAH_SEKOLAH_ADIWIYATA"].mean()
        max_sekolah = df_final["JUMLAH_SEKOLAH_ADIWIYATA"].max()
        
        avg_sampah = df_final["SAMPAH_HARIAN_TON"].mean()
        max_sampah = df_final["SAMPAH_HARIAN_TON"].max()

        avg_rth = df_final["PERSEN_RTH"].mean()
        
        col_m1.metric(
            label="Rata-rata Sekolah Adiwiyata",
            value=f"{avg_sekolah:.1f} Unit/Wilayah",
            help="Rata-rata jumlah sekolah per Kabupaten/Kota"
        )
        col_m2.metric(
            label="Rata-rata Beban Sampah",
            value=f"{avg_sampah:,.0f} Ton/Hari",
            help="Rata-rata timbulan sampah harian"
        )
        col_m3.metric(
            label="Rata-rata RTH",
            value=f"{avg_rth:.1f}%",
            delta=f"{30 - avg_rth:.1f}% dari Target UU (30%)",
            delta_color="inverse", # Merah kalau di bawah target
            help="Target Undang-Undang adalah 30%"
        )

        st.divider()

        # --- B. TABEL STATISTIK DETIL ---
        st.subheader("2. Detail Statistik")
        
        with st.expander("📖 Cara Membaca Tabel Ini (Bahasa Manusia)", expanded=False):
            st.info("""
            **Fokus pada 2 hal ini saja:**
            1.  **Mean (Rata-rata) vs 50% (Nilai Tengah):** * Jika **Mean jauh lebih besar** dari 50%, berarti ada segelintir "Kota Raksasa" yang angkanya ekstrem tinggi, sementara mayoritas daerah angkanya kecil. Ini tanda **ketimpangan**.
            2.  **Min vs Max:** * Menunjukkan seberapa jauh jarak antara daerah paling tertinggal dan daerah paling maju.
            """)

        # Styling tabel agar angka desimal rapi
        st.dataframe(
            df_final[numerical_cols].describe(percentiles=[0.25, 0.5, 0.75]).style.format("{:,.2f}"),
            use_container_width=True
        )

        st.divider()

        # --- C. DISTRIBUSI (UBAH JADI NARASI KETIMPANGAN) ---
        st.subheader("3. Peta Ketimpangan Wilayah")
        st.markdown("""
        Grafik ini menjawab pertanyaan: *"Apakah pembangunan lingkungan kita merata?"*
        """)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                # Warna histogram: Hijau jika berhubungan dengan lingkungan (RTH/Sekolah), 
                # Merah jika berhubungan dengan beban (Sampah/Luas)
                bar_color = "#66BB6A" if "SEKOLAH" in col or "RTH" in col else "#EF5350"
                
                sns.histplot(df_final[col], kde=True, ax=axes[i], bins=25, color=bar_color)
                
                # Judul yang lebih deskriptif
                axes[i].set_title(col.replace("_", " "), fontsize=10, fontweight='bold')
                axes[i].set_xlabel("")
                axes[i].set_ylabel("Jumlah Wilayah")

        for j in range(len(numerical_cols), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        st.pyplot(fig)

        st.info("""
        💡 **Kesimpulan Visual:**
        Hampir semua grafik (terutama Sampah & Sekolah) menumpuk di kiri. 
        Ini mengonfirmasi bahwa **mayoritas wilayah Indonesia masih berada di level dasar**, 
        hanya sedikit wilayah yang mencuat jauh ke kanan (Kota Metropolitan).
        """)

        st.divider()

        # --- D. DETEKSI OUTLIER OTOMATIS (KILLER FEATURE) ---
        st.subheader("4. Deteksi Wilayah Ekstrem (Outlier)")
        st.markdown("""
        Bagian ini secara otomatis mendeteksi siapa saja **"Top Player"** (Nilai Tertinggi) dan **"Priority Alert"** (Beban Tertinggi).
        """)

        # Pilihan Variabel agar Boxplot tidak penyet (skala beda jauh)
        pilihan_outlier = st.selectbox(
            "Pilih Indikator untuk Dianalisis:",
            numerical_cols,
            format_func=lambda x: x.replace("_", " ")
        )

        col_box, col_txt = st.columns([2, 1])

        with col_box:
            # Boxplot Interaktif Tunggal
            fig_box = plt.figure(figsize=(10, 4))
            sns.boxplot(x=df_final[pilihan_outlier], color="#FFD54F")
            plt.title(f"Sebaran {pilihan_outlier}")
            plt.xlabel("")
            st.pyplot(fig_box)

        with col_txt:
            # ALGORITMA PENCARI NAMA KOTA (OTOMATIS)
            # Hitung batas outlier (IQR)
            q1 = df_final[pilihan_outlier].quantile(0.25)
            q3 = df_final[pilihan_outlier].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + (1.5 * iqr)
            
            # Cari nama kota yang tembus batas atas
            outliers = df_final[df_final[pilihan_outlier] > upper_bound].sort_values(pilihan_outlier, ascending=False)
            
            st.markdown(f"**🔍 Deteksi Otomatis:**")
            if not outliers.empty:
                top_3 = outliers.head(5)
                st.write(f"Ditemukan **{len(outliers)} wilayah** dengan nilai ekstrem tinggi:")
                
                # Tampilkan Top 5 sebagai list
                for idx, row in top_3.iterrows():
                    val = row[pilihan_outlier]
                    # Format angka biar enak dibaca
                    val_fmt = f"{val:,.0f}" if val > 100 else f"{val:.2f}"
                    st.markdown(f"- **{row['KABKOT_STD']}**: {val_fmt}")
                
                if len(outliers) > 5:
                    st.caption(f"...dan {len(outliers)-5} wilayah lainnya.")
            else:
                st.success("Data merata. Tidak ditemukan wilayah dengan nilai ekstrem (Outlier).")

        # Insight Kontekstual Berdasarkan Pilihan
        if "SAMPAH" in pilihan_outlier:
            st.error("🚨 **Rekomendasi:** Wilayah yang terdeteksi di atas memiliki beban sampah yang tidak wajar. Wajib menjadi prioritas program manajemen limbah.")
        elif "SEKOLAH" in pilihan_outlier:
            st.success("🏆 **Rekomendasi:** Wilayah di atas adalah pusat keunggulan (Center of Excellence). Jadikan mentor untuk wilayah lain.")

# ========================================================
    # TAB 2: LOG TRANSFORM (PENYETARAAN SKALA)
    # ========================================================
    with tab2:
        st.subheader("2. Penyetaraan Skala Data (Log Transform)")
        
        # Penjelasan Konsep untuk Non-Statistisi
        st.info("""
        ℹ️ **Mengapa data perlu diubah?**
        
        Data kita memiliki masalah **"Ketimpangan Ekstrem"**. 
        Contoh: Ada kota dengan sampah **800.000 ton**, sementara rata-rata hanya **200 ton**. 
        
        Jika data ini langsung dimasukkan ke komputer (Machine Learning), komputer akan **bias** dan hanya memperhatikan angka yang besar.
        Teknik **Log Transform** (`np.log1p`) berguna untuk "memampatkan" angka-angka raksasa tersebut agar skalanya lebih adil dan mendekati distribusi normal (lonceng).
        """)

        # Proses Transformasi
        df_stat = df_final.copy()
        log_transform_cols = numerical_cols
        
        for col in log_transform_cols:
            df_stat[f"LOG_{col}"] = np.log1p(df_stat[col])

        st.divider()

        # Plot Perbandingan (Before vs After)
        st.subheader("Visualisasi Dampak Transformasi")
        st.markdown("Perhatikan bagaimana grafik di sisi **Kanan (Sesudah)** menjadi lebih landai dan terpusat di tengah dibandingkan sisi **Kiri (Sebelum)**.")

        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 20))

        for i, col in enumerate(log_transform_cols):
            # BEFORE (Data Asli)
            sns.histplot(df_stat[col], bins=30, kde=True, ax=axes[i, 0], color='#EF5350') # Merah (Timpang)
            axes[i, 0].set_title(f"SEBELUM: {col}\n(Sangat Timpang ke Kiri)", fontsize=10, color='red')
            axes[i, 0].set_xlabel("")
            axes[i, 0].set_ylabel("Frekuensi")
            
            # AFTER (Data Log)
            sns.histplot(df_stat[f"LOG_{col}"], bins=30, kde=True, ax=axes[i, 1], color='#42A5F5') # Biru (Normal)
            axes[i, 1].set_title(f"SESUDAH: LOG_{col}\n(Lebih Terdistribusi Normal)", fontsize=10, color='blue')
            axes[i, 1].set_xlabel("")
            axes[i, 1].set_ylabel("")

        plt.tight_layout()
        st.pyplot(fig)

        st.divider()

        # Boxplot Log Only
        st.subheader("Peta Data Setelah Penyetaraan")
        st.markdown("""
        Setelah disetarakan (Log), kita bisa melihat sebaran data dengan lebih jelas tanpa terganggu oleh angka-angka raksasa. 
        Grafik di bawah ini adalah **data bersih** yang akan dipelajari oleh model AI.
        """)
        
        log_cols_only = [f"LOG_{c}" for c in log_transform_cols]
        
        fig_box_log = plt.figure(figsize=(14, 6))
        # Menggunakan palette 'Set2' agar warna-warni tapi lembut
        sns.boxplot(data=df_stat[log_cols_only], orient="h", palette="Set2") 
        plt.title("Distribusi Data Siap Pakai (Log Scale)")
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        st.pyplot(fig_box_log)
        
        st.success("✅ **Status Data:** Skala data kini sudah stabil. Siap untuk tahap analisis korelasi dan pemodelan.")

    with tab3:
        st.subheader("Matriks Korelasi (Data Log)")
        st.write("Analisis ini bertujuan melihat kekuatan hubungan linear (Pearson) maupun hubungan peringkat (Spearman) antar variabel.")

        corr_cols = [
            "LOG_JUMLAH_SEKOLAH_ADIWIYATA",
            "LOG_PERSEN_RTH",
            "LOG_SAMPAH_HARIAN_TON",
            "LOG_SAMPAH_TAHUNAN_TON"
        ]

        # --- Bagian Visualisasi (Atas) ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("##### 1. Spearman Correlation")
            st.caption("Mengukur hubungan monotonik (baik untuk data berdistribusi tidak normal/ada outlier).")
            corr_spearman = df_stat[corr_cols].corr(method="spearman")
            fig_s = plt.figure(figsize=(8, 6))
            sns.heatmap(corr_spearman, annot=True, cmap="coolwarm", center=0, fmt=".2f", vmin=-1, vmax=1)
            plt.title("Matriks Korelasi Spearman")
            st.pyplot(fig_s)

        with c2:
            st.markdown("##### 2. Pearson Correlation")
            st.caption("Mengukur hubungan linear murni (asumsi data berdistribusi normal).")
            corr_pearson = df_stat[corr_cols].corr(method="pearson")
            fig_p = plt.figure(figsize=(8, 6))
            sns.heatmap(corr_pearson, annot=True, cmap="coolwarm", center=0, fmt=".2f", vmin=-1, vmax=1)
            plt.title("Matriks Korelasi Pearson")
            st.pyplot(fig_p)

        st.divider()

        # --- Bagian Interpretasi (Bawah) - INI YANG BARU ---
        st.subheader("📝 Insight & Interpretasi Data")
        
        col_insight1, col_insight2 = st.columns(2)

        with col_insight1:
            st.info("""
            **🚩 Temuan 1: Redundansi Data (Multikolinearitas)**
            
            Terlihat korelasi **sempurna (1.00)** antara:
            * `LOG_SAMPAH_HARIAN_TON` 
            * `LOG_SAMPAH_TAHUNAN_TON`
            
            **Artinya:** Kedua data ini membawa informasi yang persis sama. 
            **Tindakan:** Untuk pemodelan Clustering nanti, **WAJIB** membuang salah satu agar tidak bias. Kita akan pakai yang Harian saja.
            """)

        with col_insight2:
            st.warning("""
            **🔍 Temuan 2: Hubungan Adiwiyata & Lingkungan**
            
            * **Adiwiyata vs Sampah (0.29 - 0.30):** Korelasi positif lemah. Artinya, semakin banyak sekolah Adiwiyata di suatu wilayah, volume sampah cenderung *sedikit* lebih tinggi (mungkin karena wilayah tsb lebih padat penduduk/sekolahnya).
            * **Adiwiyata vs RTH (0.11):** Korelasi sangat lemah (mendekati 0). Artinya, banyaknya sekolah Adiwiyata di data ini **belum** berbanding lurus secara signifikan dengan luas Ruang Terbuka Hijau di wilayah tersebut.
            """)

        with st.expander("📚 Cara Membaca Nilai Korelasi (Klik untuk info)"):
            st.markdown("""
            * **Nilai +1**: Hubungan positif sempurna (Satu naik, yang lain pasti naik).
            * **Nilai 0**: Tidak ada hubungan sama sekali.
            * **Nilai -1**: Hubungan negatif sempurna (Satu naik, yang lain pasti turun).
            * **0.0 - 0.2**: Sangat Lemah
            * **0.2 - 0.4**: Lemah
            * **0.4 - 0.6**: Sedang
            * **> 0.6**: Kuat
            """)

    with tab4:
        st.subheader("Normalisasi Berbasis Luas Wilayah")
        st.markdown("""
        **Tujuan:** Menghitung densitas (kepadatan) per km² untuk membandingkan wilayah secara adil, 
        terlepas dari besar/kecilnya luas wilayah tersebut.
        """)
        
        # --- 1. PROSES HITUNG ---
        df_norm = df_stat.copy()
        df_norm = df_norm[df_norm["LUAS_WILAYAH"] > 0].copy() # Filter luas 0

        # Rasio (Densitas)
        df_norm["ADIWIYATA_PER_KM2"] = df_norm["JUMLAH_SEKOLAH_ADIWIYATA"] / df_norm["LUAS_WILAYAH"]
        df_norm["SAMPAH_HARIAN_PER_KM2"] = df_norm["SAMPAH_HARIAN_TON"] / df_norm["LUAS_WILAYAH"]
        df_norm["SAMPAH_TAHUNAN_PER_KM2"] = df_norm["SAMPAH_TAHUNAN_TON"] / df_norm["LUAS_WILAYAH"]

        # Log Transform (Untuk visualisasi yang lebih baik)
        df_norm["LOG_ADIWIYATA_PER_KM2"] = np.log1p(df_norm["ADIWIYATA_PER_KM2"])
        df_norm["LOG_SAMPAH_HARIAN_PER_KM2"] = np.log1p(df_norm["SAMPAH_HARIAN_PER_KM2"])
        df_norm["LOG_SAMPAH_TAHUNAN_PER_KM2"] = np.log1p(df_norm["SAMPAH_TAHUNAN_PER_KM2"])

        norm_cols = ["LOG_ADIWIYATA_PER_KM2", "LOG_SAMPAH_HARIAN_PER_KM2", "LOG_SAMPAH_TAHUNAN_PER_KM2"]

        # --- 2. VISUALISASI DISTRIBUSI (HISTOGRAM) ---
        st.markdown("#### 1. Distribusi Data Densitas (Log Scale)")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, col in enumerate(norm_cols):
            sns.histplot(df_norm[col], bins=30, kde=True, ax=axes[i], color='green')
            axes[i].set_title(f"Distribusi {col}")
            axes[i].set_xlabel("")
        plt.tight_layout()
        st.pyplot(fig)

        # Penjelasan Histogram
        st.info("""
        **📝 Interpretasi Distribusi:**
        Grafik di atas menunjukkan sebaran kepadatan sekolah dan sampah.
        * Jika kurva condong ke kiri (nilai rendah), artinya mayoritas wilayah di Indonesia masih memiliki densitas Sekolah Adiwiyata dan Sampah yang rendah (renggang).
        * Ekor panjang ke kanan menunjukkan adanya segelintir wilayah (kemungkinan kota besar) yang sangat padat.
        """)

        # --- 3. VISUALISASI OUTLIER (BOXPLOT) ---
        st.markdown("#### 2. Deteksi Outlier (Boxplot)")
        
        fig_bn = plt.figure(figsize=(10, 4))
        sns.boxplot(data=df_norm[norm_cols], orient="h", palette="Set2")
        plt.title("Boxplot Variabel Normalisasi (Log)")
        st.pyplot(fig_bn)

        # Penjelasan Boxplot
        st.caption("""
        **Cara Baca Boxplot:** Titik-titik di luar garis batas (whisker) adalah **Outlier**. 
        Ini mengonfirmasi bahwa ada ketimpangan ekstrem: beberapa daerah sangat padat (kota metropolitan), sementara sebagian besar daerah sangat longgar (kabupaten luas).
        """)

        st.divider()

        # --- 4. VISUALISASI HUBUNGAN (KORELASI) ---
        st.markdown("#### 3. Matriks Korelasi (Data Ternormalisasi)")
        
        c3, c4 = st.columns(2)
        with c3:
            st.write("**Spearman (Rank)**")
            corr_ns = df_norm[norm_cols].corr(method="spearman")
            fig_ns = plt.figure(figsize=(6, 5))
            sns.heatmap(corr_ns, annot=True, cmap="coolwarm", center=0, fmt=".2f")
            st.pyplot(fig_ns)
        with c4:
            st.write("**Pearson (Linear)**")
            corr_np = df_norm[norm_cols].corr(method="pearson")
            fig_np = plt.figure(figsize=(6, 5))
            sns.heatmap(corr_np, annot=True, cmap="coolwarm", center=0, fmt=".2f")
            st.pyplot(fig_np)

        # Penjelasan Korelasi (INI POIN PALING PENTING)
        st.warning("""
        **💡 TEMUAN PENTING: Lonjakan Korelasi!**
        
        Bandingkan hasil Pearson di sini dengan data mentah di Tab 3:
        1.  **Data Mentah:** Korelasi Adiwiyata vs Sampah hanya berkisar **0.30 (Lemah)**.
        2.  **Data Densitas (di sini):** Korelasi melonjak menjadi **0.65 - 0.85 (Kuat)**.
        
        **Kesimpulan:** Ketika dibagi dengan luas wilayah, hubungan menjadi sangat jelas. Wilayah yang padat sekolah Adiwiyata-nya, hampir pasti padat timbulan sampahnya. 
        Ini menunjukkan fenomena **Urbanisasi**: Sekolah Adiwiyata lebih banyak terkonsentrasi di wilayah perkotaan (sempit & padat) yang juga merupakan produsen sampah terbesar.
        """)

    with tab5:
        st.subheader("3. Segmentasi Pola (LM Plot)")
        st.write("Analisis ini membedah apakah hubungan Adiwiyata-Sampah berlaku sama di semua wilayah, atau berbeda antara wilayah sampah rendah vs tinggi.")
        
        try:
            df_ctx = df_norm.copy()
            
            # Membagi data menjadi 2 kelompok (Median Split)
            df_ctx["KELOMPOK_SAMPAH"] = pd.qcut(
                df_ctx["LOG_SAMPAH_HARIAN_PER_KM2"], q=2,
                labels=["Sampah Relatif Rendah", "Sampah Relatif Tinggi"]
            )
            
            # Plotting sesuai kodemu
            g = sns.lmplot(
                data=df_ctx,
                x="LOG_ADIWIYATA_PER_KM2",
                y="LOG_SAMPAH_HARIAN_PER_KM2",
                hue="KELOMPOK_SAMPAH",
                scatter_kws={"alpha": 0.5},
                height=5,
                aspect=1.2,
                legend=False # Matikan legend bawaan agar bisa diatur posisinya jika mau, atau biarkan default
            )
            
            plt.title("Perbedaan Pola Hubungan: Wilayah Rendah vs Tinggi")
            plt.legend(title="Kategori Wilayah", loc='upper left') # Merapikan legend
            st.pyplot(g.fig)
            
            # --- BAGIAN INI YANG MEMBUATNYA INFORMATIF ---
# --- PENJELASAN VISUAL & ANALITIS (UPDATED) ---
            st.info("""
            **📖 Cara Membaca Grafik ini:**
            
            Grafik ini terdiri dari 3 elemen penting:
            1.  **Titik-Titik (Dots):** Mewakili data asli setiap wilayah. 
                * 🔵 **Biru:** Wilayah sampah rendah.
                * 🟠 **Oranye:** Wilayah sampah tinggi.
            2.  **Garis Lurus (Regression Line):** Menunjukkan **arah tren**.
                * Jika garis mendatar, artinya **tidak ada pengaruh**.
                * Jika garis menanjak, artinya **ada pengaruh positif** (Sekolah tambah banyak, sampah tambah banyak).
            3.  **Area Arsiran (Shaded Area):** Menunjukkan **Keyakinan Data (Confidence Interval)**.
                * Semakin **lebar** arsiran, artinya datanya tidak konsisten/berantakan (kurang bisa dipercaya).
                * Semakin **sempit** arsiran, artinya datanya konsisten dan polanya kuat.
            
            **🎯 Kesimpulan:**
            Terlihat garis oranye menanjak tajam dengan arsiran yang cukup sempit, sedangkan garis biru cenderung landai. Ini membuktikan bahwa hubungan kuat antara "Banyak Sekolah" dan "Banyak Sampah" **hanya terjadi di wilayah yang memang sudah padat (Urban)**.
            """)
        
        except Exception as e:
            st.warning(f"Gagal membuat LM Plot: {e}")

# ==========================================
# MENU 3: MODELLING (ULTIMATE: PKL + HUGGING FACE UI)
# ==========================================
elif menu == "3. Modelling":
    st.title("🤖 Modelling: Klasifikasi Efektivitas")
    st.markdown("Evaluasi performa model dan simulasi prediksi interaktif (Real-time).")

    # 1. Cek Data Utama
    if "df_final" not in st.session_state:
        st.warning("⚠️ Data belum tersedia. Silakan proses data di Menu 1 dulu.")
        st.stop()

    # 2. Setup Path & Dependencies
    BASE_DIR = "Dataset_DS"
    PATH_MODEL = "model_lgbm_adiwiyata.pkl"  # Pastikan file ini ada!
    
    # Cek Keberadaan Model
    if not os.path.exists(PATH_MODEL):
        st.error(f"❌ File Model `{PATH_MODEL}` tidak ditemukan! Harap upload file .pkl ke folder project.")
        st.info("Tips: Jika file ada di dalam folder 'Dataset_DS', ubah path di kode menjadi os.path.join('Dataset_DS', 'modelname.pkl')")
        st.stop()

    # 3. Load Model (Hanya sekali load agar ringan)
    try:
        model = joblib.load(PATH_MODEL)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    # --- LAYOUT UTAMA (SPLIT KIRI & KANAN) ---
    # Kiri (2.5) : Evaluasi & Metrik
    # Kanan (1)  : Widget Simulator "Try it out"
    col_eval, col_inf = st.columns([2.5, 1], gap="medium")

    # ==========================================
    # BAGIAN KANAN: WIDGET SIMULATOR
    # ==========================================
    with col_inf:
        with st.container(border=True):
            st.subheader("⚡ Try it out")
            st.caption("Uji model dengan data inputan sendiri.")
            
            # Input User
            in_luas = st.number_input("Luas Wilayah (km²)", value=500.0, step=10.0)
            in_sekolah = st.number_input("Jml Sekolah Adiwiyata", value=10, step=1)
            in_sampah = st.number_input("Sampah Harian (Ton)", value=100.0, step=5.0)
            in_rth = st.slider("% RTH", 0, 100, 20)
            
            # Estimasi Tahunan
            in_sampah_tahunan = in_sampah * 365

            if st.button("Compute Prediction", type="primary", use_container_width=True):
                try:
                    # PREPROCESSING (Harus sama persis dengan Training)
                    adiwiyata_km2 = in_sekolah / in_luas
                    sampah_harian_km2 = in_sampah / in_luas
                    sampah_tahunan_km2 = in_sampah_tahunan / in_luas
                    
                    # Susun Dataframe (Urutan kolom PENTING)
                    input_data = pd.DataFrame([{
                        "LOG_ADIWIYATA_PER_KM2": np.log1p(adiwiyata_km2),
                        "LOG_SAMPAH_HARIAN_PER_KM2": np.log1p(sampah_harian_km2),
                        "LOG_SAMPAH_TAHUNAN_PER_KM2": np.log1p(sampah_tahunan_km2),
                        "PERSEN_RTH": in_rth,
                        "LUAS_WILAYAH": in_luas
                    }])
                    
                    # PREDIKSI
                    pred_class = model.predict(input_data)[0]
                    pred_proba = model.predict_proba(input_data)[0]
                    confidence = pred_proba[pred_class] * 100
                    
                    st.divider()
                    
                    if pred_class == 0:
                        st.success("✅ **RELATIF SELARAS**")
                        st.progress(confidence/100, text=f"Confidence: {confidence:.1f}%")
                    else:
                        st.error("⚠️ **TIDAK SELARAS**")
                        st.progress(confidence/100, text=f"Confidence: {confidence:.1f}%")
                        
                except Exception as e:
                    st.error(f"Error: {e}")

    # ==========================================
    # BAGIAN KIRI: EVALUASI MODEL (DATASET ASLI)
    # ==========================================
    with col_eval:
        st.subheader("📊 Model Performance Metrics")
        st.markdown("Evaluasi model terhadap seluruh data wilayah yang tersedia.")
        
        # Cek apakah hasil evaluasi sudah ada di session state?
        if "pkl_results" in st.session_state:
            # JIKA SUDAH ADA, LANGSUNG TAMPILKAN (Biar Cepat)
            res = st.session_state["pkl_results"]
            
            # Kartu Metrik
            m1, m2, m3 = st.columns(3)
            f1 = f1_score(res['y_actual'], res['y_pred'], average='macro')
            m1.metric("Accuracy", f"{res['accuracy']:.2%}")
            m2.metric("F1-Score (Macro)", f"{f1:.2%}")
            m3.metric("Total Data", f"{len(res['y_actual'])} Wilayah")
            
            # Tabs Visualisasi
            tab_cm, tab_feat, tab_dist = st.tabs(["Confusion Matrix", "Feature Importance", "Distribution"])
            
            with tab_cm:
                c_cm, c_rep = st.columns([1, 2])
                with c_cm:
                    cm = confusion_matrix(res['y_actual'], res['y_pred'])
                    fig_cm = plt.figure(figsize=(4, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=["Selaras", "Tdk Selaras"], 
                                yticklabels=["Selaras", "Tdk Selaras"])
                    st.pyplot(fig_cm)
                with c_rep:
                    rep_df = pd.DataFrame(res['report']).transpose()
                    st.dataframe(rep_df.style.format("{:.2f}"), use_container_width=True)

            with tab_feat:
                imp_df = pd.DataFrame({
                    "Fitur": res['feature_names'],
                    "Importance": res['feature_importances']
                }).sort_values(by="Importance", ascending=False)
                fig_imp = plt.figure(figsize=(8, 3))
                sns.barplot(data=imp_df, x="Importance", y="Fitur", palette="viridis")
                st.pyplot(fig_imp)
                
            with tab_dist:
                st.info("Grafik ini membandingkan data fakta (Asli) dengan tebakan Model (Prediksi).")
                df_act = pd.DataFrame(res['y_actual']).value_counts().reset_index()
                df_act.columns = ["Label", "Count"]; df_act["Type"] = "Data Asli"
                
                df_pre = pd.DataFrame(res['y_pred']).value_counts().reset_index()
                df_pre.columns = ["Label", "Count"]; df_pre["Type"] = "Prediksi Model"
                
                df_all = pd.concat([df_act, df_pre])
                df_all["Label"] = df_all["Label"].map({0: "Selaras", 1: "Tdk Selaras"})
                
                fig_d = plt.figure(figsize=(8, 4))
                sns.barplot(data=df_all, x="Label", y="Count", hue="Type", palette="pastel")
                st.pyplot(fig_d)
                
            # Tombol Reset (Opsional)
            if st.button("🔄 Refresh Evaluasi"):
                del st.session_state["pkl_results"]
                st.rerun()

        else:
            # JIKA BELUM ADA, TAMPILKAN TOMBOL LOAD
            st.info("Klik tombol di bawah untuk menjalankan evaluasi pada seluruh dataset.")
            
            if st.button("🚀 Load Dataset Evaluation", type="primary"):
                with st.spinner("Memproses seluruh dataset & melakukan prediksi..."):
                    try:
                        # --- DATA PREP LENGKAP (Merge IKA/IKU) ---
                        # Kita perlu menyatukan data lagi untuk mendapatkan Label Asli (y_actual)
                        df_final = st.session_state["df_final"].copy()
                        df_model = df_final[df_final["LUAS_WILAYAH"] > 0].copy()
                        df_model.columns = df_model.columns.str.upper().str.strip()
                        
                        # Feature Engineering
                        df_model["ADIWIYATA_PER_KM2"] = df_model["JUMLAH_SEKOLAH_ADIWIYATA"] / df_model["LUAS_WILAYAH"]
                        df_model["SAMPAH_HARIAN_PER_KM2"] = df_model["SAMPAH_HARIAN_TON"] / df_model["LUAS_WILAYAH"]
                        df_model["SAMPAH_TAHUNAN_PER_KM2"] = df_model["SAMPAH_TAHUNAN_TON"] / df_model["LUAS_WILAYAH"]
                        
                        for col in ["ADIWIYATA_PER_KM2", "SAMPAH_HARIAN_PER_KM2", "SAMPAH_TAHUNAN_PER_KM2"]:
                            df_model[f"LOG_{col}"] = np.log1p(df_model[col])
                            
                        # Merge Data Pendukung (IKA/IKU/Provinsi)
                        path_ika = os.path.join(BASE_DIR, "Indeks_Kualitas_Air.csv")
                        path_iku = os.path.join(BASE_DIR, "indeks_kualitas_udara.csv")
                        path_rth = os.path.join(BASE_DIR, "Data_RTH.xlsx")
                        
                        # Load Mapping Provinsi
                        df_rth_raw = pd.read_excel(path_rth)
                        col_kab = [c for c in df_rth_raw.columns if 'Kabupaten' in c][0]
                        df_rth_raw.rename(columns={col_kab: 'Kabupaten/Kota'}, inplace=True)
                        df_rth_raw["KABKOT_STD"] = normalize_kabkot_sekolah(df_rth_raw["Kabupaten/Kota"])
                        col_prov = [c for c in df_rth_raw.columns if 'Provinsi' in c or 'PROVINSI' in c][0]
                        prov_map = df_rth_raw.drop_duplicates("KABKOT_STD").set_index("KABKOT_STD")[col_prov]
                        df_model["PROVINSI"] = df_model["KABKOT_STD"].map(prov_map).astype(str).str.upper().str.strip()

                        # Load & Merge IKA/IKU
                        df_ika = pd.read_csv(path_ika)
                        df_iku = pd.read_csv(path_iku)
                        df_ika.rename(columns={"Provinsi": "PROVINSI", "Indeks Kualitas Air": "IKA"}, inplace=True)
                        df_iku.rename(columns={"Provinsi": "PROVINSI", "Indeks Kualitas Udara": "IKU"}, inplace=True)
                        for df in [df_ika, df_iku]: 
                            if "PROVINSI" in df.columns: df["PROVINSI"] = df["PROVINSI"].astype(str).str.upper().str.strip()
                        
                        df_model = df_model.merge(df_ika[["PROVINSI", "IKA"]], on="PROVINSI", how="left") \
                                           .merge(df_iku[["PROVINSI", "IKU"]], on="PROVINSI", how="left")
                        
                        df_model_clean = df_model.dropna(subset=["IKA", "IKU", "LOG_ADIWIYATA_PER_KM2"])
                        
                        # Labeling (Ground Truth)
                        median_adiwiyata = df_model_clean["LOG_ADIWIYATA_PER_KM2"].median()
                        median_ika = df_model_clean["IKA"].median()
                        median_iku = df_model_clean["IKU"].median()
                        df_model_clean["ADIWIYATA_TINGGI"] = df_model_clean["LOG_ADIWIYATA_PER_KM2"] >= median_adiwiyata
                        df_model_clean["LINGKUNGAN_RENDAH"] = (df_model_clean["IKA"] < median_ika) | (df_model_clean["IKU"] < median_iku)
                        df_model_clean["KETIDAKSESUAIAN"] = (df_model_clean["ADIWIYATA_TINGGI"] & df_model_clean["LINGKUNGAN_RENDAH"]).astype(int)

                        # --- PREDIKSI MASSIF ---
                        features = ["LOG_ADIWIYATA_PER_KM2", "LOG_SAMPAH_HARIAN_PER_KM2", "LOG_SAMPAH_TAHUNAN_PER_KM2", "PERSEN_RTH", "LUAS_WILAYAH"]
                        X = df_model_clean[features]
                        y_actual = df_model_clean["KETIDAKSESUAIAN"]
                        y_pred = model.predict(X)
                        
                        # Simpan ke Session
                        st.session_state["pkl_results"] = {
                            "accuracy": accuracy_score(y_actual, y_pred),
                            "report": classification_report(y_actual, y_pred, target_names=["Selaras", "Tdk Selaras"], output_dict=True),
                            "feature_importances": model.feature_importances_,
                            "feature_names": features,
                            "y_actual": y_actual,
                            "y_pred": y_pred
                        }
                        
                        st.rerun() # Refresh halaman agar hasil muncul

                    except Exception as e:
                        st.error(f"Gagal memproses dataset: {e}")