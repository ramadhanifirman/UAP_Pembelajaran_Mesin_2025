# 🛵 Analisis Sentimen Ulasan Aplikasi Gojek

Proyek ini bertujuan untuk mengklasifikasikan sentimen ulasan pengguna aplikasi Gojek di Google Play Store ke dalam tiga kategori: **Negatif**, **Netral**, dan **Positif**. Proyek ini membandingkan performa arsitektur Deep Learning klasik (LSTM) dengan model berbasis Transformer (IndoBERT & DistilBERT).

## 📄 Deskripsi Proyek
Aplikasi ini menyediakan antarmuka interaktif bagi pengguna untuk menguji model secara langsung dan melihat statistik ulasan dari dataset. Fokus utama proyek adalah melihat bagaimana model pemrosesan bahasa alami (NLP) menangani bahasa informal (slang) dan konteks ulasan pengguna di Indonesia.

## 🧪 Dataset dan Preprocessing

### 1. Dataset
- **Sumber Data**: Scraping ulasan aplikasi Gojek di Google Play Store.
- **Jumlah Kelas**: 3 (Negatif, Netral, Positif).
- **Fitur Utama**: `content` (teks ulasan) dan `score` (rating 1-5).

### 2. Preprocessing
Tahapan pengolahan data yang dilakukan sebelum masuk ke model:
- **Cleaning**: Menghapus karakter khusus, angka, emoji, dan tanda baca.
- **Case Folding**: Mengubah semua teks menjadi huruf kecil (lowercase).
- **Formalization**: Mengonversi kata gaul/singkatan menjadi kata baku (misal: "yg" -> "yang").
- **Stopword Removal**: Menghapus kata-kata umum yang tidak memiliki makna signifikan.
- **Tokenization**: Memecah kalimat menjadi token/kata.
- **Padding**: Menyamakan panjang urutan kata (Sequence) agar sesuai dengan input model.

## 🧠 Penjelasan Model

Proyek ini mengimplementasikan dan membandingkan tiga arsitektur:

1.  **LSTM (Long Short-Term Memory)**:
    - Jenis Recurrent Neural Network (RNN) yang mampu menangani dependensi jangka panjang.
    - Menggunakan layer `Embedding` yang dipelajari dari awal selama training.
    - Cocok untuk memahami urutan kata dalam kalimat ulasan yang pendek.

2.  **IndoBERT**:
    - Model berbasis BERT (Bidirectional Encoder Representations from Transformers) yang telah dilatih secara khusus pada dataset besar bahasa Indonesia.
    - Mampu memahami konteks kata secara dua arah (bidirectional).
    - Memiliki pemahaman bahasa Indonesia yang sangat kuat termasuk konteks formal maupun semi-formal.

3.  **DistilBERT (Multilingual)**:
    - Versi "ringan" (distilled) dari model BERT.
    - Mempertahankan sekitar 97% kinerja BERT tetapi 40% lebih kecil dan 60% lebih cepat.
    - Digunakan sebagai alternatif model yang lebih efisien secara komputasi namun tetap akurat.

## 📊 Hasil Evaluasi & Analisis Perbandingan

| Model      | Accuracy | F1-Score | Keterangan |
|------------|----------|----------|------------|
| **LSTM** | ~XX%     | ~XX%     | Stabil, namun terkadang sulit menangani konteks kalimat yang sangat kompleks. |
| **IndoBERT**| **~XX%** | **~XX%** | **Performa terbaik** dalam memahami konteks ulasan bahasa Indonesia. |
| **DistilBERT**| ~XX%     | ~XX%     | Kecepatan inferensi paling cepat dengan akurasi yang kompetitif. |

**Analisis Singkat**: Model berbasis Transformer (khususnya IndoBERT) cenderung lebih unggul dalam mengenali sentimen karena fitur *Self-Attention* yang memungkinkan model fokus pada kata-kata kunci dalam ulasan yang panjang atau sarkastik.

## 💻 Panduan Menjalankan Secara Lokal

### Prasyarat
- Python 3.9 - 3.12
- File model (`.h5` dan folder BERT) sudah ada di folder `DATA/`

### Langkah-langkah
1. **Clone Repositori**:
   ```bash
   git clone [https://github.com/ramadhanifirman/UAP_Pembelajaran_Mesin_2025.git](https://github.com/ramadhanifirman/UAP_Pembelajaran_Mesin_2025.git)
   cd UAP_Pembelajaran_Mesin_2025
