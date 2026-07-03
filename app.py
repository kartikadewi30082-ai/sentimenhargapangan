import streamlit as st
import joblib
import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ==========================================
# 1. PERSIAPAN MODEL DAN PREPROCESSING
# ==========================================
@st.cache_resource
def load_sastrawi():
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    return stopword_remover, stemmer

stopword_remover, stemmer = load_sastrawi()

# Load Model dan Vectorizer
try:
    model_nb = joblib.load('model_naive_bayes.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("File model atau vectorizer tidak ditemukan. Pastikan file .pkl sudah ada di direktori.")
    st.stop() # Menghentikan aplikasi jika file tidak ada

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

# ==========================================
# 2. ANTARMUKA STREAMLIT
# ==========================================
st.set_page_config(page_title="Analisis Sentimen", page_icon="📊")

st.title("📊 Sistem Klasifikasi Sentimen")
st.write("Implementasi Model Naïve Bayes untuk menganalisis sentimen ulasan.")
st.markdown("---")

user_input = st.text_area("✍️ Masukkan teks ulasan di sini:", height=150, placeholder="Contoh: Harga bahan pokok sekarang sangat mahal dan memberatkan...")

if st.button("Analisis Sentimen 🚀"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Sedang memproses dan menganalisis teks..."):
            # 1. Preprocessing input baru
            clean_text = preprocess_text(user_input)
            
            # 2. Transformasi TF-IDF
            vektor_teks = tfidf_vectorizer.transform([clean_text])
            
            # 3. Prediksi dan Probabilitas
            prediksi = model_nb.predict(vektor_teks)[0]
            probabilitas = model_nb.predict_proba(vektor_teks)[0]
            kelas_label = model_nb.classes_
            
            # Menampilkan Hasil Klasifikasi
            st.success("Analisis Selesai!")
            st.subheader("Hasil Klasifikasi:")
            
            if prediksi == 'Positif':
                st.info(f"Sentimen: **{prediksi}** 😃")
            elif prediksi == 'Negatif':
                st.error(f"Sentimen: **{prediksi}** 😡")
            else:
                st.warning(f"Sentimen: **{prediksi}** 😐")
            
            # Menampilkan Probabilitas Keyakinan Model
            st.markdown("### Tingkat Keyakinan Model:")
            # Membuat tabel dataframe untuk probabilitas agar rapi
            df_prob = pd.DataFrame([probabilitas], columns=kelas_label)
            # Format menjadi persentase
            df_prob = df_prob.applymap(lambda x: f"{x*100:.2f}%")
            st.table(df_prob)
            
            st.write("**Teks setelah preprocessing (yang dibaca oleh mesin):**")
            st.code(clean_text)