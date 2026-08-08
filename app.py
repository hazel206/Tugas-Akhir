import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# KONFIGURASI HALAMAN & STATE
# ==========================================

st.set_page_config(
    page_title="Sistem Rekomendasi Buku",
    page_icon="📚",
    layout="wide"
)

# Inisialisasi Session State untuk mengatur navigasi halaman
if "history" not in st.session_state:
    st.session_state.history = []
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "home" # "home", "search_result", "detail"
if "detail_idx" not in st.session_state:
    st.session_state.detail_idx = None
if "search_result_data" not in st.session_state:
    st.session_state.search_result_data = None

def set_detail_view(idx):
    st.session_state.detail_idx = idx
    st.session_state.view_mode = "detail"
    # Tambahkan ke riwayat jika belum ada / naikkan ke atas
    if idx in st.session_state.history:
        st.session_state.history.remove(idx)
    st.session_state.history.insert(0, idx)
    if len(st.session_state.history) > 6:
        st.session_state.history = st.session_state.history[:6]

def set_home_view():
    st.session_state.view_mode = "home"
    st.session_state.detail_idx = None

# ==========================================
# CSS (DIREVISI UNTUK GRID PRESISI)
# ==========================================

st.markdown("""
<style>
/* Mengunci tinggi judul maksimal 2 baris agar tombol sejajar */
.book-title {
    font-size: 14px;
    font-weight: bold;
    text-align: center;
    margin-top: 10px;
    height: 42px; 
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

/* Mengunci tinggi penulis maksimal 1 baris, teks lebih akan menjadi "..." */
.book-author {
    font-size: 12px;
    color: gray;
    text-align: center;
    margin-bottom: 5px;
    height: 18px; 
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}

/* Mengunci tinggi rating */
.book-rating {
    color: orange;
    font-weight: bold;
    text-align: center;
    margin-bottom: 10px;
    font-size: 13px;
    height: 20px;
}

img {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv("fix.csv")

    # Pastikan kolom thumbnail dan rating terbawa
    data = df[['title', 'authors', 'categories', 'description', 'thumbnail', 'average_rating', 'ratings_count']]
    
    # Konversi rating ke numerik, yang error atau kosong jadikan 0
    data['average_rating'] = pd.to_numeric(data['average_rating'], errors='coerce').fillna(0)
    data['ratings_count'] = pd.to_numeric(data['ratings_count'], errors='coerce').fillna(0)
    
    data = data.fillna('')

    data['combined'] = (
        data['title'].astype(str) + " " +
        data['authors'].astype(str) + " " +
        data['categories'].astype(str) + " " +
        data['description'].astype(str)
    )

    data['combined'] = data['combined'].str.lower()
    data['tokens'] = data['combined'].apply(lambda x: x.split())

    stopwords = ['and', 'a', 'about', 'the', 'of', 'is', 'that']
    data['filtered'] = data['tokens'].apply(lambda x: [w for w in x if w not in stopwords])

    stemmer = PorterStemmer()
    data['stemmed'] = data['filtered'].apply(lambda x: [stemmer.stem(word) for word in x])
    data['final'] = data['stemmed'].apply(lambda x: ' '.join(x))

    return data


# ==========================================
# TF-IDF DAN COSINE
# ==========================================

@st.cache_resource
def build_model(data):
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    tfidf_matrix = tfidf.fit_transform(data['final'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim


# ==========================================
# REKOMENDASI
# ==========================================

def rekomendasi(judul, data, cosine_sim, k):
    judul = str(judul).strip().lower()
    data['title_clean'] = data['title'].astype(str).str.lower()
    hasil = data[data['title_clean'].str.contains(judul, na=False)]

    if hasil.empty:
        return None

    idx = hasil.index[0]
    skor = list(enumerate(cosine_sim[idx]))
    skor = sorted(skor, key=lambda x: x[1], reverse=True)

    hasil_rekom = []
    for i, score in skor[1:k+1]:
        hasil_rekom.append({
            "Index": i,
            "Judul": data.loc[i, 'title'],
            "Penulis": data.loc[i, 'authors'],
            "Kategori": data.loc[i, 'categories'],
            "Similarity": round(score, 3),
            "Thumbnail": data.loc[i, 'thumbnail'],
            "Rating": data.loc[i, 'average_rating']
        })

    return idx, hasil_rekom


# ==========================================
# METRIK EVALUASI
# ==========================================

def precision_at_k(idx, cosine_sim, data, k=3):
    skor = list(enumerate(cosine_sim[idx]))
    skor = sorted(skor, key=lambda x: x[1], reverse=True)
    rekom = [i[0] for i in skor[1:k+1]]
    kategori_asli = set(str(data.loc[idx, 'categories']).lower().split())
    relevan = 0
    for i in rekom:
        kategori_rekom = set(str(data.loc[i, 'categories']).lower().split())
        if len(kategori_asli & kategori_rekom) > 0:
            relevan += 1
    return relevan / k

def ndcg_at_k(idx, cosine_sim, data, k=3):
    skor = list(enumerate(cosine_sim[idx]))
    skor = sorted(skor, key=lambda x: x[1], reverse=True)
    rekom = [i[0] for i in skor[1:k+1]]
    kategori_asli = set(str(data.loc[idx, 'categories']).lower().split())
    relevansi = []
    for i in rekom:
        kategori_rekom = set(str(data.loc[i, 'categories']).lower().split())
        rel = 1 if len(kategori_asli & kategori_rekom) > 0 else 0
        relevansi.append(rel)
    dcg = sum([rel / np.log2(pos + 2) for pos, rel in enumerate(relevansi)])
    ideal = sorted(relevansi, reverse=True)
    idcg = sum([rel / np.log2(pos + 2) for pos, rel in enumerate(ideal)])
    if idcg == 0:
        return 0
    return dcg / idcg

def evaluasi_sistem(data, cosine_sim, k=3):
    start = time.time()
    total_precision = 0
    total_ndcg = 0
    for idx in range(len(data)):
        total_precision += precision_at_k(idx, cosine_sim, data, k)
        total_ndcg += ndcg_at_k(idx, cosine_sim, data, k)
    avg_precision = total_precision / len(data)
    avg_ndcg = total_ndcg / len(data)
    runtime = time.time() - start
    return avg_precision, avg_ndcg, runtime


# ==========================================
# HELPER UI GAMBAR (DIREVISI UNTUK ASPEK RASIO)
# ==========================================

def show_image(url, width=120):
    # Mengunci tinggi gambar berdasarkan lebar untuk proporsi 2:3 yang seragam
    height = int(width * 1.5) 
    placeholder = "https://images.unsplash.com/photo-1543002588-bfa74002ed7e?q=80&w=200&auto=format&fit=crop"
    
    if pd.isna(url) or str(url).strip() == "":
        url = placeholder
    else:
        url = str(url).strip()
        if url.startswith("http://"):
            url = url.replace("http://", "https://", 1)
            
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin-bottom: 10px;">
            <img src="{url}" width="{width}" height="{height}" 
                 style="border-radius:8px; object-fit: cover; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" 
                 onerror="this.onerror=null; this.src='{placeholder}';">
        </div>
        """, unsafe_allow_html=True
    )


# ==========================================
# TAMPILAN APLIKASI
# ==========================================

data = load_data()
cosine_sim = build_model(data)

st.title("📚 Sistem Rekomendasi Buku")

tab_utama, tab_evaluasi = st.tabs(["🏠 Beranda & Pencarian", "📊 About & Evaluasi Sistem"])

with tab_utama:
    st.write("Cari buku favoritmu dan temukan rekomendasi termirip")

    col_input1, col_input2, col_btn = st.columns([3, 1, 1])
    with col_input1:
        judul_list = sorted(data['title'].dropna().unique().tolist())
        judul = st.selectbox("Pilih Judul Buku", [""] + judul_list, index=0)
    with col_input2:
        k = st.selectbox("Top-K", [3, 5, 10], index=0)
    with col_btn:
        st.write("") 
        st.write("")
        cari = st.button("🔍 Cari", use_container_width=True)

    if cari and judul != "":
        hasil = rekomendasi(judul, data, cosine_sim, k)
        if hasil is None:
            st.error("Judul buku tidak ditemukan")
        else:
            idx, rekom = hasil
            st.session_state.search_result_data = {"idx": idx, "rekom": rekom}
            st.session_state.view_mode = "search_result"
            if idx in st.session_state.history:
                st.session_state.history.remove(idx)
            st.session_state.history.insert(0, idx)

    st.markdown("---")

    # ------------------------------------------
    # RENDER BERDASARKAN STATE VIEW MODE
    # ------------------------------------------

    if st.session_state.view_mode == "detail":
        st.button("⬅ Kembali", on_click=set_home_view)
        idx_detail = st.session_state.detail_idx
        book = data.loc[idx_detail]
        
        st.subheader("📖 Detail Buku")
        col_img, col_desc = st.columns([1, 4])
        with col_img:
            show_image(book["thumbnail"], width=180)
        with col_desc:
            st.write(f"### {book['title']}")
            st.write(f"**⭐ Rating:** {book['average_rating']}")
            st.write("**Penulis:**", book["authors"])
            st.write("**Kategori:**", book["categories"])
            st.write("**Deskripsi:**")
            desc_text = str(book["description"])
            st.write(desc_text if desc_text else "Tidak ada deskripsi tersedia.")

    elif st.session_state.view_mode == "search_result" and st.session_state.search_result_data:
        st.button("⬅ Kembali ke Beranda", on_click=set_home_view)
        res_idx = st.session_state.search_result_data["idx"]
        rekom_list = st.session_state.search_result_data["rekom"]
        buku_dicari = data.loc[res_idx]
        
        st.subheader("📖 Buku Yang Dipilih")
        col_img, col_desc = st.columns([1, 4])
        with col_img:
            show_image(buku_dicari["thumbnail"], width=180)
        with col_desc:
            st.write(f"### {buku_dicari['title']}")
            st.write(f"**⭐ Rating:** {buku_dicari['average_rating']}")
            st.write("**Penulis:**", buku_dicari["authors"])
            st.write("**Kategori:**", buku_dicari["categories"])
            st.write("**Deskripsi:**")
            st.write(str(buku_dicari["description"])[:700] + "..." if str(buku_dicari["description"]) else "Tidak ada.")

        st.divider()

        st.subheader(f"📚 Top {len(rekom_list)} Rekomendasi Buku Mirip")
        num_cols = min(len(rekom_list), 5)
        if num_cols == 0: num_cols = 1
        cols = st.columns(num_cols)
        
        for i, buku in enumerate(rekom_list):
            with cols[i % num_cols]:
                show_image(buku['Thumbnail'], width=120)
                st.markdown(f"<div class='book-title'>{buku['Judul']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='book-author'>{buku['Penulis']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='book-rating'>⭐ {buku['Rating']}</div>", unsafe_allow_html=True)
                st.button("Lihat Detail", key=f"rekom_{buku['Index']}", 
                          on_click=set_detail_view, args=(buku['Index'],), use_container_width=True)

    else:
        # ==================================
        # HALAMAN BERANDA (HOME VIEW)
        # ==================================
        
        st.subheader("🌟 Top Populer")
        top_populer = data.sort_values(by='ratings_count', ascending=False).head(6)
        cols_pop = st.columns(min(len(top_populer), 6))
        for i, (idx_pop, row_pop) in enumerate(top_populer.iterrows()):
            with cols_pop[i]:
                show_image(row_pop["thumbnail"], width=120)
                st.markdown(f"<div class='book-title'>{row_pop['title']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='book-author'>{row_pop['authors']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='book-rating'>⭐ {row_pop['average_rating']}</div>", unsafe_allow_html=True)
                st.button("Lihat Detail", key=f"pop_{idx_pop}_{i}", 
                          on_click=set_detail_view, args=(idx_pop,), use_container_width=True)

        st.divider()

        st.subheader("🕒 Riwayat Terakhir")
        if not st.session_state.history:
            st.info("Belum ada riwayat. Silakan cari dan pilih judul buku di atas.")
        else:
            cols_hist = st.columns(min(len(st.session_state.history), 6))
            for i, hist_idx in enumerate(st.session_state.history):
                row_hist = data.loc[hist_idx]
                with cols_hist[i]:
                    show_image(row_hist["thumbnail"], width=120)
                    st.markdown(f"<div class='book-title'>{row_hist['title']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='book-author'>{row_hist['authors']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='book-rating'>⭐ {row_hist['average_rating']}</div>", unsafe_allow_html=True)
                    st.button("Lihat Detail", key=f"hist_{hist_idx}_{i}", 
                              on_click=set_detail_view, args=(hist_idx,), use_container_width=True)


# ==========================================
# TAB ABOUT & EVALUASI
# ==========================================

with tab_evaluasi:
    st.header("📌 About")
    st.write("Sistem rekomendasi buku ini menggunakan metode **Content Based Filtering** (membandingkan kesamaan konten) dengan algoritma **TF-IDF Vectorizer** dan mengukur kemiripannya menggunakan **Cosine Similarity**.")
    st.markdown("---")
    st.header("📊 Evaluasi Sistem")
    st.write("Evaluasi dihitung menggunakan metrik Precision@K dan NDCG@K untuk melihat seberapa relevan kategori buku yang direkomendasikan dengan kategori buku asal.")

    if st.button("Jalankan Evaluasi Sistem"):
        with st.spinner("Sedang memproses evaluasi (mungkin memakan waktu)..."):
            p3, n3, t3 = evaluasi_sistem(data, cosine_sim, 3)
            p5, n5, t5 = evaluasi_sistem(data, cosine_sim, 5)
            p10, n10, t10 = evaluasi_sistem(data, cosine_sim, 10)

        col1, col2, col3 = st.columns(3)
        col1.metric("Precision@3", round(p3, 3))
        col2.metric("NDCG@3", round(n3, 3))
        col3.metric("Runtime", f"{round(t3, 3)} detik")

        hasil_eval = pd.DataFrame({
            "K": [3, 5, 10],
            "Precision": [p3, p5, p10],
            "NDCG": [n3, n5, n10]
        })

        st.subheader("Tabel Evaluasi")
        st.dataframe(hasil_eval, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(hasil_eval["K"], hasil_eval["Precision"], marker="o", label="Precision")
        ax.plot(hasil_eval["K"], hasil_eval["NDCG"], marker="o", label="NDCG")
        ax.set_xlabel("Nilai K")
        ax.set_ylabel("Skor")
        ax.set_title("Perbandingan Precision dan NDCG")
        ax.legend()
        st.pyplot(fig)
