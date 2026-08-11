import streamlit as st
import pandas as pd
import numpy as np

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

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
if "detail_item" not in st.session_state:
    st.session_state.detail_item = None # format: {"source": "train"/"test", "idx": idx}
if "search_result_data" not in st.session_state:
    st.session_state.search_result_data = None

def set_detail_view(source, idx):
    item = {"source": source, "idx": idx}
    st.session_state.detail_item = item
    st.session_state.view_mode = "detail"
    
    # Tambahkan ke riwayat jika belum ada / naikkan ke atas
    if item in st.session_state.history:
        st.session_state.history.remove(item)
    st.session_state.history.insert(0, item)
    if len(st.session_state.history) > 6:
        st.session_state.history = st.session_state.history[:6]

def set_home_view():
    st.session_state.view_mode = "home"
    st.session_state.detail_item = None

# ==========================================
# CSS 
# ==========================================

st.markdown("""
<style>
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
# LOAD DATA & PREPROCESSING (TRAIN-TEST SPLIT)
# ==========================================

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("fix.csv")

    # Ambil kolom yang dibutuhkan (termasuk visual/UI)
    data = df[['title', 'authors', 'categories', 'description', 'thumbnail', 'average_rating', 'ratings_count']]
    
    # Konversi rating ke numerik
    data['average_rating'] = pd.to_numeric(data['average_rating'], errors='coerce').fillna(0)
    data['ratings_count'] = pd.to_numeric(data['ratings_count'], errors='coerce').fillna(0)
    
    data = data.fillna('')

    # Penggabungan fitur teks
    data['combined'] = (
        data['title'].astype(str) + " " +
        data['authors'].astype(str) + " " +
        data['categories'].astype(str) + " " +
        data['description'].astype(str)
    )

    data['combined'] = data['combined'].str.lower()
    data['tokens'] = data['combined'].apply(lambda x: x.split())

    # Stopwords
    stopwords = ['and', 'a', 'about', 'the', 'of', 'is', 'that']
    data['filtered'] = data['tokens'].apply(lambda x: [w for w in x if w not in stopwords])

    # Stemming
    stemmer = PorterStemmer()
    data['stemmed'] = data['filtered'].apply(lambda x: [stemmer.stem(word) for word in x])
    data['final'] = data['stemmed'].apply(lambda x: ' '.join(x))

    # Split Data (80% Train, 20% Test)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    
    return train_data, test_data


# ==========================================
# TF-IDF DAN COSINE SIMILARITY
# ==========================================

@st.cache_resource
def build_model(train_df, test_df):
    # Menggunakan hyperparameter
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    # Fit pada Train, Transform pada Test
    tfidf_train = tfidf.fit_transform(train_df['final'])
    tfidf_test = tfidf.transform(test_df['final'])
    
    # Hitung Cosine Similarity antara Test dan Train
    cosine_sim = cosine_similarity(tfidf_test, tfidf_train)
    
    return cosine_sim


# ==========================================
# REKOMENDASI (DIPERBAIKI UNTUK TANDA TANYA)
# ==========================================

def rekomendasi(judul, train_df, test_df, cosine_sim, k):
    # Membersihkan input judul
    judul = str(judul).strip().lower()
    test_df['title_clean'] = test_df['title'].astype(str).str.lower()
    
    # Gunakan exact match (==) agar karakter Regex seperti "?" tidak menyebabkan error
    hasil = test_df[test_df['title_clean'] == judul]

    if hasil.empty:
        return None

    # Ambil index buku di test_data
    idx = hasil.index[0]
    
    # Ambil skor similarity (array 1D sebesar jumlah train_data)
    skor = list(enumerate(cosine_sim[idx]))
    skor = sorted(skor, key=lambda x: x[1], reverse=True)

    hasil_rekom = []
    # Ambil top K rekomendasi dari train_data
    for i, score in skor[:k]:
        hasil_rekom.append({
            "Index": i, # Index di train_data
            "Judul": train_df.loc[i, 'title'],
            "Penulis": train_df.loc[i, 'authors'],
            "Kategori": train_df.loc[i, 'categories'],
            "Similarity": round(score, 3),
            "Thumbnail": train_df.loc[i, 'thumbnail'],
            "Rating": train_df.loc[i, 'average_rating']
        })

    return idx, hasil_rekom


# ==========================================
# HELPER UI GAMBAR
# ==========================================

def show_image(url, width=120):
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
# START APLIKASI
# ==========================================

train_data, test_data = load_and_preprocess_data()
cosine_sim = build_model(train_data, test_data)

st.title("📚 Sistem Rekomendasi Buku")

tab_utama, tab_about = st.tabs(["🏠 Beranda & Pencarian", "📌 About"])

with tab_utama:
    st.write("Cari buku favoritmu untuk mendapatkan rekomendasi buku yang serupa.")

    col_input1, col_input2, col_btn = st.columns([3, 1, 1])
    with col_input1:
        # Menampilkan judul hanya dari Data Uji (Test Data)
        judul_list = sorted(test_data['title'].dropna().unique().tolist())
        judul = st.selectbox("Pilih Judul Buku", [""] + judul_list, index=0)
    with col_input2:
        k = st.selectbox("Top-K", [3, 5, 10], index=0)
    with col_btn:
        st.write("") 
        st.write("")
        cari = st.button("🔍 Cari", use_container_width=True)

    if cari:
        # Tambahan agar ada peringatan jika judul kosong (belum dipilih)
        if judul == "":
            st.warning("⚠️ Silakan pilih judul buku terlebih dahulu!")
        else:
            hasil = rekomendasi(judul, train_data, test_data, cosine_sim, k)
            if hasil is None:
                st.error("Judul buku tidak ditemukan.")
            else:
                idx, rekom = hasil
                st.session_state.search_result_data = {"idx": idx, "rekom": rekom}
                st.session_state.view_mode = "search_result"
                
                # Record history
                hist_item = {"source": "test", "idx": idx}
                if hist_item in st.session_state.history:
                    st.session_state.history.remove(hist_item)
                st.session_state.history.insert(0, hist_item)

    st.markdown("---")

    # ------------------------------------------
    # RENDER BERDASARKAN STATE VIEW MODE
    # ------------------------------------------

    if st.session_state.view_mode == "detail":
        st.button("⬅ Kembali", on_click=set_home_view)
        
        detail_info = st.session_state.detail_item
        if detail_info["source"] == "train":
            book = train_data.loc[detail_info["idx"]]
        else:
            book = test_data.loc[detail_info["idx"]]
        
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
        
        buku_dicari = test_data.loc[res_idx] # Buku dari test_data
        
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
                # Tombol mereferensikan indeks dari train_data
                st.button("Lihat Detail", key=f"rekom_{buku['Index']}", 
                          on_click=set_detail_view, args=("train", buku['Index']), use_container_width=True)

    else:
        # ==================================
        # HALAMAN BERANDA (HOME VIEW)
        # ==================================
        
        st.subheader("🌟 Top Populer")
        top_populer = train_data.sort_values(by='ratings_count', ascending=False).head(6)
        cols_pop = st.columns(min(len(top_populer), 6))
        for i, (idx_pop, row_pop) in enumerate(top_populer.iterrows()):
            with cols_pop[i]:
                show_image(row_pop["thumbnail"], width=120)
                st.markdown(f"<div class='book-title'>{row_pop['title']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='book-author'>{row_pop['authors']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='book-rating'>⭐ {row_pop['average_rating']}</div>", unsafe_allow_html=True)
                st.button("Lihat Detail", key=f"pop_{idx_pop}_{i}", 
                          on_click=set_detail_view, args=("train", idx_pop), use_container_width=True)

        st.divider()

        st.subheader("🕒 Riwayat Terakhir")
        if not st.session_state.history:
            st.info("Belum ada riwayat. Silakan cari dan pilih judul buku di atas.")
        else:
            cols_hist = st.columns(min(len(st.session_state.history), 6))
            for i, hist_item in enumerate(st.session_state.history):
                source = hist_item["source"]
                idx_hist = hist_item["idx"]
                row_hist = train_data.loc[idx_hist] if source == "train" else test_data.loc[idx_hist]
                
                with cols_hist[i]:
                    show_image(row_hist["thumbnail"], width=120)
                    st.markdown(f"<div class='book-title'>{row_hist['title']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='book-author'>{row_hist['authors']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='book-rating'>⭐ {row_hist['average_rating']}</div>", unsafe_allow_html=True)
                    st.button("Lihat Detail", key=f"hist_{source}_{idx_hist}_{i}", 
                              on_click=set_detail_view, args=(source, idx_hist), use_container_width=True)


# ==========================================
# TAB ABOUT 
# ==========================================

with tab_about:
    st.header("📌 About")
    st.write("Sistem rekomendasi buku ini menggunakan metode Content Based Filtering dengan logik pembagian data latih (80%) dan uji (20%).")
    st.write("Algoritma yang digunakan adalah TF-IDF Vectorizer dipadukan dengan Cosine Similarity.")
    st.write(f"Jumlah Data Latih: {len(train_data)} | Jumlah Data Uji: {len(test_data)}")
