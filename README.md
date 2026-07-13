# Sistem Klasifikasi Jenis Hewan Berbasis Web

Proyek UAS — Sistem klasifikasi otomatis 10 jenis hewan dari gambar, menggunakan transfer learning DenseNet121, diakses melalui web app berbasis Flask.

---

## Fungsi dan Fitur Proyek

Aplikasi web ini memungkinkan pengguna untuk:
- **Mengunggah foto hewan** (drag & drop atau klik pilih file) melalui antarmuka web
- **Mendapatkan hasil klasifikasi otomatis** — sistem akan memprediksi jenis hewan pada gambar dari 10 kategori yang tersedia
- **Melihat tingkat keyakinan (confidence score)** — ditampilkan dalam persentase untuk 3 prediksi teratas (top-3), bukan hanya satu jawaban tunggal
- **Melihat nama hewan dalam Bahasa Indonesia** — hasil prediksi otomatis diterjemahkan dari label asli dataset (Bahasa Italia) ke Bahasa Indonesia

**10 Kelas Hewan yang Dapat Diklasifikasikan:**

| No | Kelas (Italia) | Nama (Indonesia) |
|---|---|---|
| 1 | cane | Anjing |
| 2 | cavallo | Kuda |
| 3 | elefante | Gajah |
| 4 | farfalla | Kupu-kupu |
| 5 | gallina | Ayam |
| 6 | gatto | Kucing |
| 7 | mucca | Sapi |
| 8 | pecora | Domba |
| 9 | ragno | Laba-laba |
| 10 | scoiattolo | Tupai |

---

## Bahasa Pemrograman yang Digunakan

- **Python** — untuk training model machine learning dan backend web (Flask)
- **HTML, CSS, JavaScript** — untuk frontend/antarmuka web (tanpa framework tambahan, vanilla JS)

---

## Framework, Library, dan Tools yang Digunakan

### Training Model (Google Colab)
| Library | Kegunaan |
|---|---|
| TensorFlow / Keras | Membangun & melatih model deep learning (transfer learning DenseNet121) |
| NumPy | Operasi array & komputasi numerik |
| Matplotlib & Seaborn | Visualisasi data (distribusi kelas, learning curve, confusion matrix) |
| Pillow (PIL) | Membuka & memproses gambar |
| scikit-learn | Menghitung `class_weight`, `classification_report`, `confusion_matrix` |
| split-folders | Membagi dataset menjadi folder train/validation/test |

### Backend (Web Server)
| Library | Kegunaan |
|---|---|
| Flask | Framework web Python untuk membangun REST API |
| Flask-CORS | Mengizinkan komunikasi antara frontend dan backend berbeda origin |
| TensorFlow/Keras | Memuat model hasil training dan menjalankan prediksi (`model.predict()`) |
| Pillow (PIL) | Memproses gambar yang diunggah pengguna sebelum diprediksi |
| NumPy | Mengubah gambar menjadi array numerik untuk input model |

### Frontend (Antarmuka Web)
- **HTML5** — struktur halaman
- **CSS3** — styling tampilan (tema "field guide")
- **JavaScript (Fetch API)** — mengirim gambar ke backend secara asynchronous dan menampilkan hasil tanpa reload halaman

### Model Machine Learning
- **Arsitektur**: DenseNet121 (transfer learning, pretrained ImageNet)
- **Metode**: Fine-tuning 2 tahap — feature extraction (freeze base model) dilanjutkan fine-tuning (unfreeze 30 layer terakhir)

---

## Dataset

**Nama Dataset**: Animal Image Dataset (Animals-10)
**Sumber/Link**: https://www.kaggle.com/datasets/alessiocorrado99/animals10

### Penjelasan Dataset
- Dataset berisi **~26.000 gambar** hewan yang terbagi dalam **10 kelas** berbeda: anjing, kucing, kuda, kupu-kupu, ayam, sapi, domba, gajah, tupai, dan laba-laba
- Gambar dikumpulkan dari Google Images, dengan variasi ukuran, sudut pandang, dan latar belakang yang beragam
- Nama folder/label pada dataset menggunakan **Bahasa Italia** (cane, cavallo, elefante, dst.), sehingga pada proyek ini ditambahkan mapping ke Bahasa Indonesia agar lebih mudah dipahami pengguna
- Dataset **tidak seimbang sempurna** antar kelas — beberapa kelas (misalnya anjing dan laba-laba) memiliki jumlah gambar lebih banyak dibanding kelas lain (misalnya gajah dan sapi). Hal ini ditangani menggunakan teknik `class_weight` saat training agar model tidak bias terhadap kelas mayoritas

**Pembagian data yang digunakan:**
- 80% Training
- 10% Validation
- 10% Testing

---

## Kelebihan Proyek

1. **Akurasi tinggi** — model mencapai **96.65% test accuracy**, dengan precision dan recall rata-rata di atas 0.96 untuk hampir semua kelas
2. **Transfer learning yang efisien** — menggunakan DenseNet121 pretrained ImageNet sehingga tidak perlu training dari nol, mempercepat proses dan hasil lebih akurat meski dataset per kelas terbatas
3. **Penanganan ketidakseimbangan kelas** — menggunakan `class_weight` yang lebih hemat memori dibanding oversampling manual, sehingga tetap bisa dijalankan pada dataset besar tanpa membebani RAM
4. **Training 2 tahap (freeze → fine-tuning)** — mengikuti praktik terbaik transfer learning untuk mencegah rusaknya bobot pretrained (catastrophic forgetting)
5. **Antarmuka web yang responsif dan mudah digunakan** — mendukung drag & drop gambar, menampilkan hasil prediksi beserta confidence score secara visual (progress bar)
6. **Arsitektur backend-frontend terpisah** — memudahkan pengembangan lebih lanjut (misalnya frontend bisa diganti framework lain tanpa mengubah backend)
7. **Dokumentasi lengkap** — proses training didokumentasikan dalam notebook Jupyter yang mencakup eksplorasi data (EDA), augmentasi, training, hingga evaluasi

---

## Kekurangan Proyek (Bug/Warning)

1. **Server Flask masih dalam mode development** — saat dijalankan muncul peringatan:
   ```
   WARNING: This is a development server. Do not use it in a production deployment.
   ```
   Untuk penggunaan produksi/publik, sebaiknya menggunakan WSGI server seperti Gunicorn atau Waitress, bukan `app.run()` bawaan Flask.

2. **Belum di-deploy secara online** — aplikasi saat ini hanya berjalan di `localhost` (komputer lokal), belum bisa diakses melalui internet oleh pengguna lain.

3. **Terbatas pada 10 kelas hewan** — model hanya bisa mengenali 10 jenis hewan sesuai dataset Animals-10. Jika pengguna mengunggah gambar hewan di luar 10 kelas tersebut, model tetap akan memaksa memberi salah satu dari 10 label yang ada (bukan "tidak dikenali").

4. **Sensitif terhadap kualitas gambar input** — akurasi prediksi bisa menurun jika gambar terlalu gelap, buram, memiliki banyak objek lain, atau sudut pengambilan gambar tidak umum.

5. **Peringatan (warning) non-kritis dari TensorFlow saat backend dijalankan**, seperti:
   ```
   oneDNN custom operations are on...
   This TensorFlow binary is optimized to use available CPU instructions...
   ```
   Peringatan ini bersifat informatif terkait optimisasi komputasi CPU dan tidak memengaruhi jalannya aplikasi maupun hasil prediksi.

6. **Belum ada validasi ketat terhadap file yang diunggah** — misalnya belum ada pembatasan ukuran maksimal file gambar yang bisa diunggah pengguna.

---

## Struktur Proyek

```
webapp/
├── app.py                                    # Backend Flask (API prediksi)
├── index.html                                # Frontend (antarmuka upload gambar)
├── requirements.txt                          # Daftar dependency Python
├── README.md                                 # Dokumentasi proyek
├── Animals10_DenseNet121_Training.ipynb      # Notebook Jupyter (proses training model)
└── model/
    ├── animal_classifier_densenet121.keras   # Model hasil training
    └── class_names_id.json                   # Mapping label kelas (Italia → Indonesia)
```

---

## Cara Menjalankan Proyek

### 1. Install dependency
```bash
pip install -r requirements.txt
```

### 2. Jalankan backend
```bash
python app.py
```
Backend akan berjalan di `http://localhost:5000`

### 3. Buka frontend
Buka file `index.html` di browser, atau gunakan extension **Live Server** di VS Code.

### 4. Gunakan aplikasi
Unggah foto hewan (anjing, kucing, kuda, gajah, kupu-kupu, ayam, sapi, domba, laba-laba, atau tupai), lalu lihat hasil prediksinya.

---

## Hasil Evaluasi Model

| Metrik | Nilai |
|---|---|
| Test Accuracy | **96.65%** |
| Test Loss | 0.1530 |
| Precision (rata-rata) | ~0.96–0.97 |
| Recall (rata-rata) | ~0.96–0.97 |

Detail lengkap proses training, learning curve, classification report, dan confusion matrix dapat dilihat pada notebook `Animals10_DenseNet121_Training.ipynb`.

---

## Author

Dibuat untuk memenuhi tugas UAS — Klasifikasi Gambar menggunakan Transfer Learning.
