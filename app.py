import streamlit as st
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==========================================
# 1. PERSIAPAN MODEL DAN PREPROCESSING
# ==========================================
# Menginisialisasi Sastrawi (Stopword & Stemmer)
@st.cache_resource # Cache agar tidak diload berulang kali saat tombol ditekan
def load_sastrawi():
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    return stopword_remover, stemmer

stopword_remover, stemmer = load_sastrawi()

# Load Model dan Vectorizer yang sudah disimpan
try:
    model_nb = joblib.load('model_naive_bayes.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("File model atau vectorizer tidak ditemukan. Pastikan file .pkl sudah ada di direktori.")

# Fungsi Preprocessing (Harus sama persis dengan saat pelatihan)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Cleaning
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Case Folding
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Stopword Removal
    text = stopword_remover.remove(text)
    # Stemming
    text = stemmer.stem(text)
    return text

# ==========================================
# 2. ANTARMUKA STREAMLIT
# ==========================================
st.set_page_config(page_title="Analisis Sentimen", page_icon="📊")

st.title("📊 Sistem Klasifikasi Sentimen")
st.write("Implementasi Model Naïve Bayes untuk menganalisis sentimen ulasan.")
st.markdown("---")

# Input pengguna
user_input = st.text_area("✍️ Masukkan teks ulasan di sini:", height=150, placeholder="Contoh: Harga bahan pokok sekarang sangat mahal dan memberatkan rakyat kecil...")

if st.button("Analisis Sentimen 🚀"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Sedang memproses dan menganalisis teks..."):
            # 1. Preprocessing input baru
            clean_text = preprocess_text(user_input)
            
            # 2. Transformasi TF-IDF
            # Perhatikan: gunakan transform(), BUKAN fit_transform()
            vektor_teks = tfidf_vectorizer.transform([clean_text])
            
            # 3. Prediksi menggunakan Naïve Bayes
            prediksi = model_nb.predict(vektor_teks)[0]
            
            # Menampilkan Hasil
            st.success("Analisis Selesai!")
            st.subheader("Hasil Klasifikasi:")
            
            if prediksi == 'Positif':
                st.info(f"Sentimen: **{prediksi}** 😃")
            elif prediksi == 'Negatif':
                st.error(f"Sentimen: **{prediksi}** 😡")
            else:
                st.warning(f"Sentimen: **{prediksi}** 😐")
                
            st.write("**Teks setelah preprocessing:**")
            st.code(clean_text)
