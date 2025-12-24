# 🛵 Gojek Sentiment Analysis Dashboard

Aplikasi Web berbasis Streamlit untuk mengklasifikasi sentimen ulasan pengguna aplikasi Gojek menggunakan tiga arsitektur Deep Learning yang berbeda: **LSTM**, **IndoBERT**, dan **DistilBERT**.

## 📊 Fitur Utama
- **Multi-Model Prediction**: Bandingkan hasil klasifikasi dari 3 model (LSTM, IndoBERT, dan DistilBERT).
- **Interactive Dashboard**: Visualisasi distribusi sentimen dataset asli.
- **WordCloud**: Melihat kata-kata yang paling sering muncul dalam ulasan pengguna.
- **Confidence Score**: Menampilkan tingkat kepercayaan model untuk setiap prediksi.

## 📁 Struktur Folder
```text
.
├── DATA/
│   ├── LSTM/
│   │   ├── model_lstm_gojek.h5
│   │   └── tokenizer.pickle
│   ├── indobert/
│   │   ├── tf_model.h5
│   │   ├── config.json
│   │   └── vocab.txt
│   └── distilbert/
│       ├── tf_model.h5
│       ├── config.json
│       └── vocab.txt
├── app.py                 # File utama Streamlit
├── Gojek.csv              # Dataset ulasan
├── requirements.txt       # Daftar library python
└── README.md
