# Sistem Pengenalan Wajah dan Deteksi Suku

Aplikasi ini merupakan sistem komprehensif untuk pengenalan wajah dan deteksi suku/etnis menggunakan Computer Vision dan Deep Learning. Sistem ini mengimplementasikan algoritma canggih seperti MTCNN untuk deteksi wajah, FaceNet untuk perbandingan wajah, dan ResNet18 untuk klasifikasi suku/etnis.

![Sistem Pengenalan Wajah dan Deteksi Suku](https://miro.medium.com/max/1400/1*uOQ4SUcanFVxsD9u4YKWpw.png)

## Fitur Utama

1. **Preprocessing Dataset Otomatis**
   - Deteksi dan crop wajah dari semua gambar di folder `data/raw`
   - Pemisahan dataset menjadi Train, Val, Test berdasarkan etnis
   - Pengelompokan data sesuai etnis (Jawa, Batak, Sunda)

2. **Deteksi Wajah (MTCNN)**
   - Mendeteksi wajah dalam gambar dengan berbagai pose dan kondisi
   - Visualisasi bounding box dan facial landmarks
   - Ekstraksi wajah untuk analisis lebih lanjut

3. **Perbandingan Wajah (FaceNet)**
   - Mengukur kemiripan antara dua wajah dengan skor similarity
   - Threshold yang dapat disesuaikan untuk verifikasi identitas
   - Visualisasi hasil perbandingan

4. **Deteksi Suku/Etnis (ResNet18)**
   - Klasifikasi wajah ke dalam 3 kategori suku (Jawa, Batak, Sunda)
   - Training model dengan transfer learning
   - Visualisasi hasil prediksi dengan confidence scores

## Algoritma yang Digunakan

### 1. MTCNN (Multi-task Cascaded Convolutional Networks)
- Deep learning framework dengan tiga tahap deteksi (P-Net, R-Net, O-Net)
- Akurasi tinggi dalam mendeteksi wajah dan landmark
- Robust terhadap variasi pose, pencahayaan, dan oklusi

### 2. FaceNet
- Model yang menghasilkan embedding wajah 128-dimensional
- Menggunakan triplet loss untuk mengoptimalkan jarak antar wajah
- Akurasi tinggi dalam perbandingan wajah (face verification)

### 3. ResNet18 dengan Transfer Learning
- Arsitektur CNN pre-trained pada ImageNet
- Fine-tuning untuk klasifikasi suku/etnis
- Performa tinggi dengan dataset terbatas

## Persyaratan Sistem

- Python 3.8 atau lebih baru
- CUDA-compatible GPU (opsional, untuk training lebih cepat)
- RAM minimal 4GB (8GB direkomendasikan)
- Storage minimal 2GB untuk dataset dan model

## Struktur Proyek

```
sistem_deteksi_wajah/
│
├── app.py                    # Aplikasi Streamlit utama
├── preprocess_dataset.py     # Script untuk preprocessing dataset
├── train_ethnic_model.py     # Script untuk training model etnis
│
├── data/                     # Folder untuk dataset
│   ├── raw/                  # Gambar asli sebelum preprocessing
│   ├── cropped_mtcnn/        # Hasil crop wajah dengan MTCNN
│   ├── Train/                # Data training (terorganisir berdasarkan etnis)
│   ├── Val/                  # Data validasi
│   └── Test/                 # Data testing
│
├── models/                   # Folder untuk menyimpan model terlatih
│   └── ethnic_detector.pth   # Model deteksi suku/etnis
│
└── README.md                 # Dokumentasi proyek
```

## Instalasi dan Setup

### 1. Clone Repository atau Siapkan Folder Proyek

```bash
# Buat folder proyek
mkdir sistem_deteksi_wajah
cd sistem_deteksi_wajah
```

### 2. Buat Virtual Environment

```bash
# Buat dan aktifkan virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS
```

### 3. Instal Dependency

```bash
# Instal semua dependency yang diperlukan
pip install -r requirements.txt
```

Atau instal secara manual:

```bash
# Core packages
pip install numpy==1.23.5 pandas==1.5.3 streamlit==1.28.1

# Image processing and vision
pip install opencv-python==4.8.1.78 Pillow==10.0.1

# Data science and visualization
pip install matplotlib==3.7.2 seaborn==0.12.2 scikit-learn==1.2.2 tqdm==4.66.1

# Deep learning
pip install tensorflow==2.12.0 torch==2.0.1 torchvision==0.15.2

# Face detection and recognition
pip install mtcnn==0.1.1 facenet-pytorch==2.5.3

# Image augmentation
pip install albumentations==1.3.0
```

### 4. Siapkan Struktur Folder

```bash
# Buat struktur folder yang diperlukan
mkdir -p data/raw data/cropped_mtcnn models
mkdir -p data/Train/Jawa data/Train/Batak data/Train/Sunda
mkdir -p data/Val/Jawa data/Val/Batak data/Val/Sunda
mkdir -p data/Test/Jawa data/Test/Batak data/Test/Sunda
```

### 5. Letakkan Gambar Dataset di Folder Raw

Letakkan semua gambar dataset di folder `data/raw`. Format penamaan yang direkomendasikan:
- Format: `[Nama]_[Kondisi].jpg`
- Contoh: `Abay_Dekat.jpg`, `Ahmad_Indoor.jpg`, `Akbar_Jauh.jpg`

## Cara Menjalankan Aplikasi

### Opsi 1: Langsung Menggunakan Streamlit (All-in-One)

```bash
# Pastikan virtual environment aktif
venv\Scripts\activate

# Jalankan aplikasi Streamlit
streamlit run app.py
```

Streamlit akan membuka browser secara otomatis dengan aplikasi berjalan di http://localhost:8501

### Opsi 2: Menjalankan Preprocessing dan Training Terpisah

```bash
# Preprocessing dataset (mendeteksi wajah dan split data)
python preprocess_dataset.py

# Training model deteksi etnis
python train_ethnic_model.py

# Jalankan aplikasi Streamlit
streamlit run app.py
```

## Workflow Penggunaan Aplikasi

### 1. Preprocessing Dataset

- Buka halaman "Preprocessing Dataset" dari sidebar
- Klik tombol "Mulai Preprocessing Dataset"
- Tunggu hingga proses selesai (deteksi wajah, cropping, dan split dataset)
- Lihat hasil preprocessing yang akan menampilkan contoh wajah yang telah di-crop

### 2. Deteksi Wajah

- Buka halaman "Deteksi Wajah" dari sidebar
- Upload gambar yang berisi wajah
- Sistem akan menampilkan hasil deteksi beserta bounding box dan landmarks
- Lihat wajah yang telah di-crop dengan confidence score

### 3. Perbandingan Wajah

- Buka halaman "Perbandingan Wajah" dari sidebar
- Upload dua gambar wajah yang ingin dibandingkan
- Sesuaikan threshold kemiripan menggunakan slider
- Sistem akan menampilkan skor kemiripan dan hasil keputusan (sama/berbeda)

### 4. Deteksi Suku/Etnis

- Buka halaman "Deteksi Suku/Etnis" dari sidebar
- Pilih tab "Training Model" jika ingin melatih model (jika belum dilatih)
- Atur parameter training (epochs, batch size) dan klik "Latih Model"
- Tunggu hingga proses training selesai dan lihat hasil evaluasi
- Pilih tab "Prediksi Suku/Etnis" untuk memprediksi etnis dari gambar wajah
- Upload gambar wajah dan lihat hasil prediksi beserta confidence scores

## Troubleshooting

### Masalah Umum dan Solusinya

#### 1. Error "ModuleNotFoundError: No module named 'tensorflow'"

```bash
pip install tensorflow==2.12.0
```

#### 2. Error "numpy.dtype size changed, may indicate binary incompatibility"

```bash
pip uninstall -y numpy pandas
pip install numpy==1.23.5 pandas==1.5.3
```

#### 3. Error "No face detected"

- Pastikan gambar memiliki wajah yang jelas dan tidak tertutup
- Coba gunakan gambar dengan pencahayaan yang lebih baik
- Pastikan resolusi gambar cukup tinggi

#### 4. Aplikasi Berjalan Lambat

- Tutup aplikasi lain yang berjalan di latar belakang
- Pastikan komputer Anda memenuhi persyaratan sistem minimal
- Jika memungkinkan, gunakan GPU untuk pemrosesan yang lebih cepat

#### 5. Error Memori

Jika mengalami error kehabisan memori:

```bash
# Restart aplikasi dengan parameter memori yang lebih kecil
streamlit run app.py --server.maxUploadSize=50
```

## Performance dan Metrik

### MTCNN (Deteksi Wajah)
- Akurasi deteksi: >95% pada dataset benchmark
- Kecepatan: ~0.2-0.5 detik per gambar (tergantung resolusi)

### FaceNet (Perbandingan Wajah)
- AUC ROC: 0.97
- EER (Equal Error Rate): ~0.08
- Threshold optimal: 0.84

### ResNet18 (Deteksi Suku/Etnis)
- Akurasi: ~85-90% pada dataset test
- F1-Score rata-rata: ~0.87

## Pengembangan Lebih Lanjut

Beberapa ide untuk pengembangan sistem lebih lanjut:

1. **Penambahan Suku/Etnis**: Menambahkan lebih banyak kategori suku/etnis
2. **Deteksi Atribut Tambahan**: Usia, gender, emosi, dll
3. **Real-Time Processing**: Integrasi dengan webcam untuk analisis real-time
4. **Aplikasi Mobile**: Mengembangkan versi aplikasi untuk perangkat mobile
5. **Cloud Deployment**: Menyebarkan aplikasi ke platform cloud untuk akses publik

## Tim Pengembang

Sistem ini dikembangkan oleh kelompok 2A_018_025_026 untuk memenuhi tugas Praktikum Pengolahan Citra Digital.

## Lisensi

Proyek ini dikembangkan untuk tujuan pendidikan dan penelitian.

---

*Untuk pertanyaan atau masalah teknis, silakan buat issue baru atau hubungi tim pengembang.*