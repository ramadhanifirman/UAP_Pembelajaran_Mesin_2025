import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import (
    BertTokenizer, TFBertForSequenceClassification,
    DistilBertTokenizerFast, TFDistilBertForSequenceClassification
)

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Gojek Sentiment Dashboard", page_icon="🛵", layout="wide")

# --- Load Model Functions ---

@st.cache_resource
def load_lstm():
    # Load model .h5 dan tokenizer JSON untuk LSTM
    model = tf.keras.models.load_model('./DATA/LSTM/model_lstm_gojek.h5')
    with open('./DATA/LSTM/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

@st.cache_resource
def load_indobert():
    # Cukup arahkan ke folder yang berisi file-file tersebut
    path = './DATA/indobert' 
    tokenizer = BertTokenizer.from_pretrained(path)
    # Model secara otomatis mencari config.json dan tf_model.h5 di folder tersebut
    model = TFBertForSequenceClassification.from_pretrained(path)
    return model, tokenizer

@st.cache_resource
def load_distilbert():
    # Cukup arahkan ke folder yang berisi file-file tersebut
    path = './DATA/distilbert'
    tokenizer = DistilBertTokenizerFast.from_pretrained(path)
    # Model secara otomatis mencari config.json dan tf_model.h5 di folder tersebut
    model = TFDistilBertForSequenceClassification.from_pretrained(path)
    return model, tokenizer

# --- Navigasi Tab ---
tab1, tab2 = st.tabs(["🔍 Prediksi Sentimen", "📊 Visualisasi Dataset"])

# ==========================================
# TAB 1: PREDIKSI SENTIMEN
# ==========================================
with tab1:
    st.title("Pilih Model & Analisis Review")
    
    col_input, col_res = st.columns([1, 1])
    
    with col_input:
        pilihan_model = st.selectbox(
            "Pilih Arsitektur Model:",
            ("LSTM", "IndoBERT", "DistilBERT")
        )
        user_review = st.text_area("Masukkan ulasan:", placeholder="Tulis ulasan di sini...", height=150)
        btn_proses = st.button("Jalankan Analisis")

    if btn_proses and user_review:
        with st.spinner('Memproses...'):
            # Logika prediksi sesuai pilihan (sama dengan kode sebelumnya)
            if pilihan_model == "LSTM":
                model, tokenizer = load_lstm()
                seq = tokenizer.texts_to_sequences([user_review])
                padded = pad_sequences(seq, maxlen=100)
                prob = model.predict(padded)[0]
            elif pilihan_model == "IndoBERT":
                model, tokenizer = load_indobert()
                inputs = tokenizer([user_review], padding=True, truncation=True, max_length=64, return_tensors='tf')
                prob = tf.nn.softmax(model(inputs).logits, axis=-1).numpy()[0]
            else:
                model, tokenizer = load_distilbert()
                inputs = tokenizer([user_review], padding=True, truncation=True, max_length=64, return_tensors='tf')
                prob = tf.nn.softmax(model(inputs).logits, axis=-1).numpy()[0]

            result_idx = np.argmax(prob)
            labels = {0: "Negatif 😡", 1: "Netral 😐", 2: "Positif 🤩"}
            
            with col_res:
                st.subheader("Hasil Analisis")
                st.metric("Label Dominan", labels[result_idx])
                st.write(f"Confidence: {np.max(prob)*100:.2f}%")
                
                # Grafik Probabilitas
                fig_prob, ax_prob = plt.subplots(figsize=(5, 3))
                sns.barplot(x=['Negatif', 'Netral', 'Positif'], y=prob, palette=['red', 'gray', 'green'], ax=ax_prob)
                st.pyplot(fig_prob)

# ==========================================
# TAB 2: VISUALISASI DATASET
# ==========================================
with tab2:
    st.title("Dashboard Statistik Data Review")
    
    try:
        # Load data untuk visualisasi
        df = pd.read_csv('Gojek.csv')
        
        # Preprocessing label untuk chart
        def labeling(score):
            if score <= 2: return 'Negatif'
            elif score == 3: return 'Netral'
            else: return 'Positif'
        
        if 'label' not in df.columns:
            df['label'] = df['score'].apply(labeling)

        # Row 1: Statistik Dasar
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Review", len(df))
        c2.metric("Rating Rata-rata", round(df['score'].mean(), 2))
        c3.metric("Sentimen Terbanyak", df['label'].mode()[0])

        st.divider()

        # Row 2: Distribusi Sentimen & WordCloud
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Distribusi Sentimen")
            fig_dist, ax_dist = plt.subplots()
            sns.countplot(data=df, x='label', palette='viridis', ax=ax_dist)
            st.pyplot(fig_dist)

        with col_right:
            st.subheader("Kata Sering Muncul (WordCloud)")
            all_text = " ".join(df['content'].astype(str))
            wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)

    except FileNotFoundError:
        st.error("File 'Gojekk.csv' tidak ditemukan. Pastikan file dataset ada di folder yang sama.")