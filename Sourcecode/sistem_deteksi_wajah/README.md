# Sistem Pengenalan Wajah dan Deteksi Suku

Aplikasi ini merupakan implementasi dari tugas praktikum pengembangan sistem pengenalan wajah dan deteksi suku menggunakan Computer Vision dan Deep Learning. Dengan antarmuka Streamlit yang user-friendly, sistem ini dapat melakukan deteksi wajah, perbandingan kemiripan wajah, dan klasifikasi suku/etnis.

![Screenshot Aplikasi](https://miro.medium.com/max/1400/1*uOQ4SUcanFVxsD9u4YKWpw.png)

## Fitur Utama

1. **Deteksi Wajah** - Mendeteksi wajah dalam gambar menggunakan algoritma MTCNN
2. **Perbandingan Wajah** - Membandingkan dua wajah untuk menentukan kemiripan menggunakan FaceNet
3. **Deteksi Suku/Etnis** - Mengklasifikasikan wajah ke dalam kategori suku/etnis (Jawa, Batak, Sunda) menggunakan CNN dengan Transfer Learning

## Teknologi yang Digunakan

- **MTCNN (Multi-task Cascaded Convolutional Networks)** - Untuk deteksi wajah
- **FaceNet** - Untuk perhitungan kemiripan wajah
- **ResNet18 dengan Transfer Learning** - Untuk klasifikasi suku/etnis
- **Streamlit** - Untuk antarmuka web interaktif
- **TensorFlow** - Sebagai backend untuk MTCNN
- **PyTorch** - Untuk model FaceNet dan ResNet18
- **OpenCV** - Untuk pemrosesan gambar

## Persyaratan Sistem

- Python 3.8 atau lebih baru
- Minimal 4GB RAM (8GB direkomendasikan)
- Kapasitas disk minimal 2GB untuk instalasi dependency
- Koneksi internet untuk mengunduh model pre-trained
- Webcam (opsional) untuk penggunaan dengan gambar real-time

## Struktur Proyek

```
sistem_deteksi_wajah/
│
├── app.py                   # File utama aplikasi Streamlit
├── modules/                 # Modul-modul untuk fitur berbeda
│   ├── __init__.py
│   ├── face_detection.py    # Modul deteksi wajah dengan MTCNN
│   ├── face_similarity.py   # Modul perbandingan wajah dengan FaceNet
│   └── ethnic_detection.py  # Modul deteksi etnis dengan CNN
│
├── utils/                   # Utilitas dan helper functions
│   └── __init__.py
│
├── models/                  # Folder untuk menyimpan model terlatih
│
├── data/                    # Folder untuk dataset
│   ├── raw/                 # Gambar asli
│   ├── cropped/             # Wajah yang di-crop
│   ├── Train/               # Data training
│   ├── Val/                 # Data validasi
│   └── Test/                # Data testing
│
├── requirements.txt         # File requirements untuk dependency
└── README.md                # File dokumentasi ini
```

## Instalasi dan Setup

### 1. Clone Repository atau Siapkan Folder Proyek

```bash
# Jika menggunakan git
git clone https://github.com/yourusername/sistem_deteksi_wajah.git
cd sistem_deteksi_wajah

# Atau buat folder baru
mkdir sistem_deteksi_wajah
cd sistem_deteksi_wajah
```

### 2. Buat Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Siapkan Struktur Folder

```bash
mkdir -p modules utils models data/raw data/cropped data/Train data/Val data/Test
```

### 4. Buat File Requirements.txt

Buat file `requirements.txt` dengan konten berikut:

```
# Core packages
numpy==1.23.5
pandas==1.5.3
streamlit==1.28.1

# Image processing and vision
opencv-python==4.8.1.78
Pillow==10.0.1

# Data science and visualization
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.2.2
tqdm==4.66.1

# Deep learning
tensorflow==2.12.0
torch==2.0.1
torchvision==0.15.2

# Face detection and recognition
mtcnn==0.1.1
facenet-pytorch==2.5.3

# Image augmentation
albumentations==1.3.0
```

### 5. Instal Dependency

```bash
pip install -r requirements.txt
```

### 6. Siapkan File-file Kode

Letakkan semua file kode yang sudah disediakan ke dalam struktur folder yang telah dibuat:

1. `app.py` di direktori root
2. File-file modul di folder `modules/`
3. Utilitas di folder `utils/`

## Cara Menjalankan Aplikasi

### 1. Pastikan Virtual Environment Aktif

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

### 3. Akses Aplikasi

Setelah menjalankan perintah di atas, browser web akan terbuka secara otomatis menampilkan aplikasi. Jika tidak, buka browser dan akses URL yang ditampilkan di terminal (biasanya http://localhost:8501).

## Penggunaan Fitur

### Deteksi Wajah

1. Pilih mode "Deteksi Wajah" dari sidebar
2. Upload gambar yang berisi wajah
3. Aplikasi akan mendeteksi wajah menggunakan MTCNN dan menampilkan hasilnya
4. Anda dapat menyimpan dan mengunduh wajah terdeteksi

### Perbandingan Wajah

1. Pilih mode "Perbandingan Wajah" dari sidebar
2. Upload dua gambar wajah yang ingin dibandingkan
3. Atur threshold kemiripan sesuai kebutuhan
4. Aplikasi akan menghitung dan menampilkan skor kemiripan
5. Hasil akan menunjukkan apakah kedua wajah berasal dari orang yang sama atau berbeda

### Deteksi Suku/Etnis

1. Pilih mode "Deteksi Suku/Etnis" dari sidebar
2. Upload gambar wajah
3. Aplikasi akan mengklasifikasikan wajah ke dalam suku/etnis Jawa, Batak, atau Sunda
4. Hasil akan menampilkan prediksi dan confidence scores untuk setiap kategori

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

## Pelatihan Model Deteksi Suku/Etnis (Opsional)

Jika ingin melatih ulang model deteksi suku/etnis:

1. Siapkan dataset dengan struktur yang sesuai di folder `data/Train`, `data/Val`, dan `data/Test`
2. Buat script Python untuk melatih model (contoh telah disediakan di panduan)
3. Jalankan script training
4. Model terlatih akan disimpan di folder "models" dan akan digunakan secara otomatis oleh aplikasi

## Kontribusi dan Pengembangan Lebih Lanjut

Beberapa ide untuk pengembangan lebih lanjut:
- Menambahkan lebih banyak kategori suku/etnis
- Implementasi deteksi emosi
- Integrasi dengan kamera real-time
- Pengembangan fitur pengenalan wajah untuk keamanan
- Optimasi untuk perangkat mobile

## Tim Pengembang

Aplikasi ini dikembangkan oleh kelompok 2A_018_025_026 untuk tugas Pengolahan Citra Digital.

## Lisensi

Proyek ini bersifat akademis dan untuk keperluan pembelajaran.

---

*Terima kasih telah menggunakan Sistem Pengenalan Wajah dan Deteksi Suku. Jika ada pertanyaan atau masalah, silakan buat issue baru di repositori ini.*
