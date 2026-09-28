# AI Model Experiment & Evaluation: Sentiment Analysis

## 1. Problem Statement

### Objective
Tim produk e-commerce ingin membuat fitur otomatis untuk mengklasifikasikan sentimen ulasan pelanggan di halaman produk. Eksperimen ini membandingkan dua pendekatan, yaitu model klasik Scikit-learn dan LLM via Gemini API, untuk menentukan pendekatan yang paling layak dikembangkan ke produksi.

### Target / Label
- Input: teks ulasan pelanggan (`review_text`)
- Output: sentimen `positif` atau `negatif` (`sentiment`)

### Batasan & Asumsi
- Klasifikasi biner (positif/negatif), tanpa kelas netral
- Kedua pendekatan dievaluasi pada test set yang sama
- Hanya memakai dataset yang disediakan
- Tidak melatih model deep learning dari nol
- Dataset asli 200 baris, tetapi hanya 40 ulasan unik (160 duplikat). Duplikat dihapus di memori sebelum split, file CSV asli tidak diubah

## 2. Dataset
- File: `data/customer_reviews_sentiment.csv`
- Kolom: `review_id`, `product_name`, `review_text`, `sentiment`
- Setelah deduplikasi: 40 ulasan (20 positif, 20 negatif)
- Split: 28 data latih dan 12 data uji (6 positif, 6 negatif), `test_size=0.3`, `random_state=42`, `stratify`
- Temuan: sebelum deduplikasi, model klasik mendapat akurasi 1.00 karena data banyak yang sama di train dan test

## 3. Ringkasan Eksperimen
**Pendekatan 1: Model klasik**
- TF-IDF untuk mengubah teks menjadi fitur, lalu Logistic Regression
- Dilatih pada 28 ulasan, diuji pada 12 ulasan

**Pendekatan 2: Gemini API**
- Model: `gemini-3.5-flash-lite` biaya murah dan hemat token
- Teknik: zero-shot (tanpa contoh di prompt)
- Parameter: `temperature=0.0` agar jawaban konsisten
- Pengaturan thinking tidak dipakai karena request dengan parameter itu sempat ditolak error 400
- Hasil prediksi disimpan di `documentation/gemini_predictions.csv`

## 4. Hasil Evaluasi
| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Waktu per ulasan |
|---|---|---|---|---|---|
| Model Klasik (TF-IDF + LogReg) | 1.0 | 1.0 | 1.0 | 1.0 | (0.24) ms |
| Gemini | 1.0 | 1.0 | 1.0 | 1.0 | 15,07 detik |

Confusion matrix kedua model sama: `[[6, 0], [0, 6]]`.

Interpretasi:
- Kedua pendekatan seri di semua metrik, sehingga skor tidak bisa menentukan pemenang.
- Test set hanya 12 ulasan, jadi skor sempurna belum cukup untuk menyimpulkan performa umum.
- Cross-validation 5-fold model klasik menghasilkan F1 macro rata-rata 0,68 (rentang 0,37 sampai 0,87), sehingga skor 1.00 di test set sebagian besar kebetulan.

## 5. Analisis Trade-off dan Limitation
| Aspek | Model Klasik | Gemini API |
|---|---|---|
| Implementasi | Perlu training | Tanpa training, cukup prompt |
| Kecepatan | (isi) ms per ulasan | 15,07 detik per ulasan |
| Biaya | Gratis | Berbayar per panggilan |
| Koneksi | Lokal, tanpa internet | Perlu internet dan API key |
| Stabilitas | Konsisten | Bisa error atau model dihentikan |

Limitation:
- Hanya 40 ulasan unik dan test set 12 ulasan, satu kesalahan setara turun sekitar 8%
- Ulasan tergolong mudah, tidak menguji hal kompleks, campuran bahasa, atau ulasan netral, sehingga belum ada contoh salah prediksi untuk dianalisis
- TF-IDF tidak menangkap negasi dan konteks (contoh. "tidak mengecewakan")
- Gemini: free tier dibatasi limitasi token, koneksi putus, dan beberapa model lama sudah tidak tersedia (404)

## 6. Rekomendasi Technical Approach
Rekomendasi: gunakan Gemini API untuk tahap awal, dan siapkan
transisi ke model klasik setelah data berlabel bertambah.

Alasan:
1. Skor kedua pendekatan seri, sehingga keputusan bertumpu pada trade-off.
2. Data berlabel yang tersedia sangat kecil yaitu 40 ulasan unik. Pada kondisi
   ini model klasik tidak stabil (CV F1 0.68), sedangkan Gemini tidak butuh
   data latih dan langsung berfungsi.
3. Untuk volume kecil sampai menengah, waktu 15 detik per ulasan masih bisa
   ditangani lewat pemrosesan di belakang layar (batch atau antrean) dan
   hasilnya disimpan.

Syarat dan risikonya:
- Untuk trafik besar atau kebutuhan real-time, Gemini terlalu lambat dan
  biayanya naik seiring jumlah ulasan.
- Perlu paket berbayar, penanganan error (retry), dan pemantauan model
  yang dihentikan.

Langkah lanjutan: kumpulkan data berlabel lebih banyak (memakai hasil
Gemini yang dikoreksi manusia), lalu latih ulang model klasik dan
evaluasi ulang dengan test set yang lebih besar. Jika performanya setara,
pindah ke model klasik karena jauh lebih cepat dan murah.

## 7. Cara Menjalankan
1. Install dependensi:

```
   pip install -r requirements.txt
```
2. Buat file `.env` di folder utama:

```
   GEMINI_API_KEY=isi_api_key_kamu
```
3. Buka `notebook/experiment_notebook.ipynb` lalu jalankan semua sel.
4. Hasil Gemini dimuat dari `documentation/gemini_predictions.csv` sehingga API tidak dipanggil ulang. Untuk memanggil API dari awal, hapus file itu dan pastikan API key sudah terisi.

## Struktur Folder
```
model-experiment-assignment/
├── data/customer_reviews_sentiment.csv
├── notebook/experiment_notebook.ipynb
├── documentation/gemini_predictions.csv
├── README.md
└── requirements.txt
```