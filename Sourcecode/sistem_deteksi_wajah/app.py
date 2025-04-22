"""
Sistem Pengenalan Wajah dan Deteksi Suku Menggunakan Computer Vision
Aplikasi Streamlit dengan fitur preprocessing otomatis
"""
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim
import time
from pathlib import Path
import tempfile
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
import itertools
from modules.shape_analysis import ShapeAnalyzer

# Mengatur tampilan halaman Streamlit
st.set_page_config(
    page_title="Sistem Pengenalan Wajah dan Deteksi Suku",
    page_icon="🧑‍🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Direktori utama
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CROPPED_DIR = DATA_DIR / "cropped_mtcnn"
MODELS_DIR = BASE_DIR / "models"

# Membuat direktori jika belum ada
for directory in [DATA_DIR, RAW_DIR, CROPPED_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Dictionary mapping nama orang ke etnis (hardcoded dari dataset)
ETHNICITY_MAPPING = {
    'Abay': 'batak',
    'Ahmad': 'jawa',
    'Akbar': 'jawa',
    'Ambon': 'jawa',
    'Azwa': 'batak',
    'Faris': 'jawa',
    'Humsans': 'sunda',
    'Ibrahim': 'sunda',
    'Mutiah': 'sunda',
    'Rafka': 'sunda',
    'Rara': 'batak',
    'Rizal': 'jawa',
    'Rizky': 'sunda',
    'Saskia': 'batak',
    'Tian': 'batak',
}

# Fungsi untuk load modul (lazy loading)
@st.cache_resource
def load_face_detector():
    from mtcnn import MTCNN
    return MTCNN()

@st.cache_resource
def load_face_similarity_model():
    from facenet_pytorch import InceptionResnetV1
    return InceptionResnetV1(pretrained='vggface2').eval()

# Fungsi preprocessing gambar
def detect_and_crop_faces(image_path, detector, output_folder):
    """Deteksi dan crop wajah dari gambar"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    
    if not faces:
        return None
    
    # Crop wajah pertama yang terdeteksi
    face = faces[0]
    x, y, w, h = face['box']
    face_crop = img[y:y+h, x:x+w]
    
    # Buat nama file output
    filename = os.path.basename(image_path)
    name_part = os.path.splitext(filename)[0]
    save_path = os.path.join(output_folder, f"{name_part}_face1.jpg")
    
    # Simpan gambar cropped
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(save_path, face_crop)
    
    return save_path

def preprocess_face(face_img, target_size=(160, 160)):
    """Preprocess wajah untuk model FaceNet"""
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(face_rgb, target_size)
    tensor = torch.tensor(resized).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)

def calculate_similarity(image1, image2, model):
    """Hitung kemiripan antara dua wajah"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    preprocessed1 = preprocess_face(image1)
    preprocessed2 = preprocess_face(image2)
    
    with torch.no_grad():
        embedding1 = model(preprocessed1)
        embedding2 = model(preprocessed2)
        
    sim = cosine_similarity(embedding1.numpy(), embedding2.numpy())[0][0]
    
    # Normalisasi ke rentang 0-1
    normalized_sim = (sim + 1) / 2
    
    return normalized_sim

def process_dataset_and_split(cropped_dir, ethnicity_mapping):
    """Proses dataset dan bagi menjadi Train/Val/Test"""
    names = list(ethnicity_mapping.keys())
    labels = list(ethnicity_mapping.values())
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        names, labels, test_size=0.4, stratify=labels, random_state=1
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=35
    )
    
    train_set = {name: ethnicity_mapping[name] for name in X_train}
    val_set = {name: ethnicity_mapping[name] for name in X_val}
    test_set = {name: ethnicity_mapping[name] for name in X_test}
    
    # Buat folders Train/Val/Test
    splits = ['Train', 'Val', 'Test']
    labels = ['jawa', 'batak', 'sunda']
    
    for split in splits:
        for label in labels:
            os.makedirs(os.path.join(DATA_DIR, split, label.capitalize()), exist_ok=True)
    
    # Salin gambar ke folder yang sesuai
    all_sets = {
        'Train': train_set,
        'Val': val_set,
        'Test': test_set
    }
    
    for split, name_label_dict in all_sets.items():
        for name, label in name_label_dict.items():
            for file in os.listdir(cropped_dir):
                if file.startswith(name.capitalize()):
                    src_path = os.path.join(cropped_dir, file)
                    dst_path = os.path.join(DATA_DIR, split, label.capitalize(), file)
                    shutil.copy(src_path, dst_path)
    
    return train_set, val_set, test_set

def generate_image_pairs(image_folder):
    """Buat semua pasangan gambar untuk evaluasi kemiripan"""
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    pairs = list(itertools.combinations(image_files, 2))
    
    data = []
    for img1, img2 in pairs:
        name1 = img1.split('_')[0].lower()
        name2 = img2.split('_')[0].lower()
        actual = "Yes" if name1 == name2 else "No"
        data.append({'img1': img1, 'img2': img2, 'actual': actual})
    
    return pd.DataFrame(data)

def plot_roc_curve(y_true, y_scores):
    """Plot ROC Curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold using F1 score
    from sklearn.metrics import f1_score
    import numpy as np
    
    f1_scores = [f1_score(y_true, y_scores >= t) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', 
             label=f'Optimal Threshold = {optimal_threshold:.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve + Optimal Threshold')
    ax.legend()
    ax.grid(True)
    
    return fig, optimal_threshold

# Komponen UI utama
def main():
    st.title("Sistem Pengenalan Wajah dan Deteksi Suku")
    st.markdown("""
    Sistem berbasis Computer Vision dan Deep Learning untuk:
    1. **Deteksi Wajah** - Menggunakan MTCNN untuk mendeteksi wajah dalam gambar
    2. **Perbandingan Wajah** - Menggunakan FaceNet untuk membandingkan kemiripan dua wajah
    3. **Deteksi Suku/Etnis** - Mengklasifikasikan wajah ke dalam kategori suku/etnis (Jawa, Batak, Sunda)
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigasi")
    app_mode = st.sidebar.selectbox(
    "Pilih Mode Aplikasi",
    ["Beranda", "Preprocessing Dataset", "Deteksi Wajah", "Perbandingan Wajah", 
     "Deteksi Suku/Etnis", "Analisis Bentuk Wajah", "Tentang"]
    )
    
    # ========== HOME PAGE ==========
    if app_mode == "Beranda":
        st.header("Selamat datang di Sistem Pengenalan Wajah dan Deteksi Suku")
        
        # Gambar ilustrasi
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image("https://miro.medium.com/max/1400/1*uOQ4SUcanFVxsD9u4YKWpw.png", 
                    caption="Ilustrasi Face Recognition", use_column_width=True)
        
        st.subheader("Panduan Penggunaan")
        st.markdown("""
        ### Langkah 1: Preprocessing Dataset
        - Pada halaman "Preprocessing Dataset", anda dapat mengolah seluruh gambar di folder `data/raw`
        - Sistem akan mendeteksi dan crop wajah, lalu membaginya ke folder Train/Val/Test
        
        ### Langkah 2: Deteksi Wajah
        - Upload gambar untuk mendeteksi wajah menggunakan MTCNN
        
        ### Langkah 3: Perbandingan Wajah
        - Upload dua gambar wajah untuk membandingkan kemiripan
        - Sesuaikan threshold kemiripan untuk menentukan keputusan
        
        ### Langkah 4: Deteksi Suku/Etnis
        - Upload gambar wajah untuk memprediksi suku/etnis
        - Model akan menklasifikasikan ke kategori Jawa, Batak, atau Sunda
        """)
    
    # ========== PREPROCESSING PAGE ==========
    elif app_mode == "Preprocessing Dataset":
        st.header("Preprocessing Dataset")
        
        st.write("Halaman ini memungkinkan anda melakukan preprocessing pada seluruh dataset.")
        st.write("Dataset akan diproses melalui tahapan berikut:")
        st.markdown("""
        1. **Deteksi & Cropping Wajah** - Menggunakan MTCNN pada seluruh gambar di folder `data/raw`
        2. **Pemisahan Dataset** - Membagi data menjadi Train/Val/Test berdasarkan etnis
        3. **Augmentasi Data** - Melakukan augmentasi data (untuk training)
        """)
        
        # Tampilkan contoh gambar dari folder raw
        if os.path.exists(RAW_DIR) and any(os.scandir(RAW_DIR)):
            st.subheader("Contoh Gambar di Folder Raw")
            
            sample_images = []
            for root, dirs, files in os.walk(RAW_DIR):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        sample_images.append(os.path.join(root, file))
                    if len(sample_images) >= 5:
                        break
                if len(sample_images) >= 5:
                    break
            
            cols = st.columns(len(sample_images))
            for i, (img_path, col) in enumerate(zip(sample_images, cols)):
                img = Image.open(img_path)
                col.image(img, caption=os.path.basename(img_path), use_column_width=True)
        
        # Button untuk memulai preprocessing
        if st.button("Mulai Preprocessing Dataset"):
            with st.spinner('Mendeteksi dan Crop Wajah...'):
                # Deteksi dan crop wajah dengan MTCNN
                detector = load_face_detector()
                
                # Cari semua gambar di folder raw
                raw_images = []
                for root, dirs, files in os.walk(RAW_DIR):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            raw_images.append(os.path.join(root, file))
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Crop wajah
                cropped_faces = []
                for i, img_path in enumerate(raw_images):
                    status_text.text(f"Processing {os.path.basename(img_path)}...")
                    face_path = detect_and_crop_faces(img_path, detector, str(CROPPED_DIR))
                    if face_path:
                        cropped_faces.append(face_path)
                    progress_bar.progress((i + 1) / len(raw_images))
                
                status_text.text(f"Selesai cropping {len(cropped_faces)} wajah dari {len(raw_images)} gambar")
            
            with st.spinner('Memisahkan Dataset...'):
                # Split dataset ke Train/Val/Test
                train_set, val_set, test_set = process_dataset_and_split(CROPPED_DIR, ETHNICITY_MAPPING)
                
                # Tampilkan hasil
                st.success("Preprocessing selesai!")
                st.write("Dataset telah dipisahkan menjadi:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Training Set:**")
                    for name, etnis in train_set.items():
                        st.write(f"- {name}: {etnis}")
                
                with col2:
                    st.write("**Validation Set:**")
                    for name, etnis in val_set.items():
                        st.write(f"- {name}: {etnis}")
                
                with col3:
                    st.write("**Test Set:**")
                    for name, etnis in test_set.items():
                        st.write(f"- {name}: {etnis}")
            
            # Opsi untuk melihat hasil cropping
            if st.checkbox("Lihat Hasil Cropping"):
                st.subheader("Hasil Cropping Wajah")
                
                # Ambil beberapa contoh hasil crop
                sample_cropped = []
                for file in os.listdir(CROPPED_DIR):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        sample_cropped.append(os.path.join(CROPPED_DIR, file))
                    if len(sample_cropped) >= 5:
                        break
                
                cols = st.columns(len(sample_cropped))
                for i, (img_path, col) in enumerate(zip(sample_cropped, cols)):
                    img = Image.open(img_path)
                    col.image(img, caption=os.path.basename(img_path), use_column_width=True)
    
    # ========== FACE DETECTION PAGE ==========
    elif app_mode == "Deteksi Wajah":
        st.header("Deteksi Wajah dengan MTCNN")
        
        st.write("""
        Upload gambar untuk mendeteksi wajah menggunakan algoritma MTCNN.
        Algoritma ini dapat mendeteksi wajah dengan berbagai pose, ekspresi, dan kondisi pencahayaan.
        """)
        
        # Initialize face detector
        detector = load_face_detector()
        
        # Upload image
        uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Show original image
            st.subheader("Gambar Asli")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Detect faces
            with st.spinner("Mendeteksi wajah..."):
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(img_rgb)
                
                # Draw bounding boxes
                result_img = image.copy()
                for face in faces:
                    x, y, w, h = face['box']
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw key points
                    keypoints = face['keypoints']
                    for point in keypoints.values():
                        cv2.circle(result_img, point, 2, (0, 0, 255), 2)
            
            # Show processed image
            st.subheader("Hasil Deteksi Wajah")
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            
            # Show detection results
            st.subheader(f"Terdeteksi {len(faces)} wajah")
            
            if len(faces) > 0:
                # Display each detected face
                cols = st.columns(min(len(faces), 4))
                
                for i, (face, col) in enumerate(zip(faces, cols)):
                    x, y, w, h = face['box']
                    face_img = image[y:y+h, x:x+w]
                    col.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), caption=f"Wajah {i+1}")
                    col.metric("Confidence", f"{face['confidence']:.2f}")
    
    # ========== FACE SIMILARITY PAGE ==========
    elif app_mode == "Perbandingan Wajah":
        st.header("Perbandingan Kemiripan Wajah dengan FaceNet")
        
        st.write("""
        Upload dua gambar wajah untuk membandingkan kemiripan mereka.
        Sistem menggunakan model FaceNet untuk mengekstrak fitur wajah dan menghitung kemiripan.
        """)
        
        # Initialize models
        detector = load_face_detector()
        face_model = load_face_similarity_model()
        
        # Adjust similarity threshold
        threshold = st.slider(
            "Threshold Kemiripan", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.8,
            help="Nilai threshold untuk menentukan apakah dua wajah adalah orang yang sama."
        )
        
        # Upload two images
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gambar Wajah 1")
            uploaded_file1 = st.file_uploader("Upload gambar wajah 1", type=["jpg", "jpeg", "png"], key="face1")
            
        with col2:
            st.subheader("Gambar Wajah 2")
            uploaded_file2 = st.file_uploader("Upload gambar wajah 2", type=["jpg", "jpeg", "png"], key="face2")
        
        if uploaded_file1 is not None and uploaded_file2 is not None:
            # Read images
            file_bytes1 = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
            file_bytes2 = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)
            image1 = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)
            image2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
            
            # Show original images
            col1.image(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            col2.image(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
            
            # Detect faces
            with st.spinner("Mendeteksi wajah..."):
                img_rgb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                img_rgb2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                faces1 = detector.detect_faces(img_rgb1)
                faces2 = detector.detect_faces(img_rgb2)
            
            if not faces1 or not faces2:
                st.error("Tidak dapat mendeteksi wajah pada salah satu atau kedua gambar. Silakan coba gambar lain.")
            else:
                # Get first face from each image
                face1_box = faces1[0]['box']
                face2_box = faces2[0]['box']
                
                x1, y1, w1, h1 = face1_box
                x2, y2, w2, h2 = face2_box
                
                face1_img = image1[y1:y1+h1, x1:x1+w1]
                face2_img = image2[y2:y2+h2, x2:x2+w2]
                
                # Show cropped faces
                st.subheader("Wajah Terdeteksi")
                display_col1, display_col2 = st.columns(2)
                display_col1.image(cv2.cvtColor(face1_img, cv2.COLOR_BGR2RGB), caption="Wajah 1")
                display_col2.image(cv2.cvtColor(face2_img, cv2.COLOR_BGR2RGB), caption="Wajah 2")
                
                # Calculate similarity
                with st.spinner("Menghitung kemiripan..."):
                    similarity = calculate_similarity(face1_img, face2_img, face_model)
                
                # Show similarity result
                st.subheader("Hasil Perbandingan")
                
                # Create metrics
                col1, col2 = st.columns(2)
                col1.metric("Skor Kemiripan", f"{similarity:.4f}")
                col2.metric("Hasil", "Sama" if similarity >= threshold else "Berbeda")
                
                # Create progress bar for similarity
                st.progress(similarity)
                
                # Show interpretation
                if similarity >= threshold:
                    st.success(f"Kedua wajah kemungkinan berasal dari orang yang sama (skor kemiripan: {similarity:.4f} ≥ {threshold:.2f}).")
                else:
                    st.warning(f"Kedua wajah kemungkinan berasal dari orang yang berbeda (skor kemiripan: {similarity:.4f} < {threshold:.2f}).")
                    # Tambahkan checkbox untuk ROC curve analysis
                if st.checkbox("Tampilkan Analisis ROC Curve"):
                    st.subheader("Analisis ROC Curve")
                    
                    # Generate ROC curve dari image pairs
                    with st.spinner("Menganalisis dataset untuk ROC curve..."):
                        # Jika sudah ada data cropped_mtcnn dan ingin menggunakan data real
                        if os.path.exists(CROPPED_DIR) and len(os.listdir(CROPPED_DIR)) > 0:
                            # Buat pasangan gambar
                            df_pairs = generate_image_pairs(str(CROPPED_DIR))
                            
                            # Hitung similarity untuk setiap pasangan
                            similarities = []
                            progress_bar = st.progress(0)
                            
                            for i, (_, row) in enumerate(df_pairs.iterrows()):
                                img1_path = os.path.join(CROPPED_DIR, row['img1'])
                                img2_path = os.path.join(CROPPED_DIR, row['img2'])
                                
                                if os.path.exists(img1_path) and os.path.exists(img2_path):
                                    img1 = cv2.imread(str(img1_path))
                                    img2 = cv2.imread(str(img2_path))
                                    
                                    sim = calculate_similarity(img1, img2, face_model)
                                    similarities.append(sim)
                                else:
                                    similarities.append(None)
                                
                                progress_bar.progress((i + 1) / len(df_pairs))
                            
                            # Update dataframe
                            df_pairs['proba'] = similarities
                            
                            # Prepare data for ROC curve
                            y_true = df_pairs['actual'].map({'Yes': 1, 'No': 0})
                            y_scores = df_pairs['proba'].dropna()
                            
                            # Plot ROC curve
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            # Calculate ROC curve
                            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                            roc_auc = auc(fpr, tpr)
                            
                            # Plot ROC curve
                            ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                            
                            # Find optimal threshold
                            from sklearn.metrics import f1_score
                            f1_scores = [f1_score(y_true, y_scores >= t) for t in thresholds]
                            optimal_idx = np.argmax(f1_scores)
                            optimal_threshold = thresholds[optimal_idx]
                            
                            # Highlight optimal threshold
                            ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', 
                                      label=f'Optimal Threshold = {optimal_threshold:.2f}')
                            
                            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title('ROC Curve + Optimal Threshold')
                            ax.legend()
                            ax.grid(True)
                            
                            st.pyplot(fig)
                            
                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            col1.metric("AUC", f"{roc_auc:.4f}")
                            col2.metric("Optimal Threshold", f"{optimal_threshold:.4f}")
                            
                            # Calculate metrics at optimal threshold
                            y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
                            from sklearn.metrics import accuracy_score, precision_score, recall_score
                            accuracy = accuracy_score(y_true, y_pred_optimal)
                            precision = precision_score(y_true, y_pred_optimal)
                            recall = recall_score(y_true, y_pred_optimal)
                            
                            col3.metric("Accuracy", f"{accuracy:.4f}")
                            
                            # More metrics
                            st.markdown("##### Detailed Metrics at Optimal Threshold")
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            metric_col1.metric("Precision", f"{precision:.4f}")
                            metric_col2.metric("Recall", f"{recall:.4f}")
                            metric_col3.metric("F1 Score", f"{f1_scores[optimal_idx]:.4f}")
                            
                            # Interpretasi
                            st.info(f"""
                            **Hasil Analisis ROC Curve:**
                            - **AUC (Area Under Curve)**: {roc_auc:.4f} - Semakin tinggi nilainya (mendekati 1), semakin baik model membedakan wajah yang sama dan berbeda
                            - **Optimal Threshold**: {optimal_threshold:.4f} - Threshold ini memberikan keseimbangan terbaik antara True Positive Rate dan False Positive Rate
                            
                            Anda dapat menyesuaikan threshold perbandingan wajah ke nilai optimal ({optimal_threshold:.4f}) untuk hasil terbaik.
                            """)
                        else:
                            # Jika tidak ada data, gunakan data dummy
                            st.warning("Data wajah belum diproses. Gunakan fitur 'Preprocessing Dataset' terlebih dahulu.")
                            
                            # Tampilkan contoh ROC curve dengan data dummy
                            y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
                            y_scores = np.array([0.1, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
                            
                            # Plot ROC curve
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            # Calculate ROC curve
                            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                            roc_auc = auc(fpr, tpr)
                            
                            # Plot the curve
                            ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                            ax.set_xlabel('False Positive Rate')
                            ax.set_ylabel('True Positive Rate')
                            ax.set_title('ROC Curve (Example)')
                            ax.legend()
                            ax.grid(True)
                            
                            st.pyplot(fig)
                            
                            st.info("Ini adalah contoh ROC curve. Untuk mendapatkan ROC curve dari dataset Anda, gunakan fitur 'Preprocessing Dataset' terlebih dahulu.")

                
    
    # ========== ETHNIC DETECTION PAGE ==========
    elif app_mode == "Deteksi Suku/Etnis":
        st.header("Deteksi Suku/Etnis dengan CNN")
        
        st.write("""
        Upload gambar wajah untuk memprediksi suku/etnis.
        Sistem menggunakan CNN dengan transfer learning untuk mengklasifikasikan wajah ke dalam 3 kategori suku/etnis (Jawa, Batak, Sunda).
        """)
        
        # Cek apakah model sudah terlatih
        model_path = MODELS_DIR / "ethnic_detector.pth"
        model_trained = model_path.exists()
        
        # Tab untuk Training dan Prediksi
        tab1, tab2 = st.tabs(["Training Model", "Prediksi Suku/Etnis"])
        
        with tab1:
            st.subheader("Training Model Deteksi Suku/Etnis")
            
            if not model_trained:
                st.warning("Model belum dilatih. Silakan latih model terlebih dahulu.")
            else:
                st.success("Model sudah dilatih dan siap digunakan.")
            
            # Opsi training
            epochs = st.slider("Jumlah Epochs", min_value=5, max_value=50, value=20)
            batch_size = st.slider("Batch Size", min_value=8, max_value=64, value=32, step=8)
            
            if st.button("Latih Model"):
                import torch
                from torch import nn
                from torchvision import models
                import torchvision
                import albumentations as A
                from albumentations.pytorch import ToTensorV2
                
                # Cek ketersediaan data training
                train_dir = DATA_DIR / "Train"
                val_dir = DATA_DIR / "Val"
                test_dir = DATA_DIR / "Test"
                
                if not (train_dir.exists() and val_dir.exists() and test_dir.exists()):
                    st.error("Data training/validasi/test belum disiapkan. Silakan lakukan preprocessing terlebih dahulu.")
                else:
                    # Persiapkan transformasi gambar
                    class AlbumentationsTransform:
                        def __init__(self, transform):
                            self.transform = transform
                        
                        def __call__(self, img):
                            # Convert PIL image to NumPy array
                            image_np = np.array(img)
                            
                            # Apply Albumentations transform
                            augmented = self.transform(image=image_np)
                            image = augmented['image']
                            
                            return image
                    
                    train_transform = AlbumentationsTransform(
                        A.Compose([
                            A.Resize(224, 224),
                            A.Rotate(limit=15, p=0.8),
                            A.HorizontalFlip(p=0.5),
                            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                            A.MotionBlur(blur_limit=3, p=0.2),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2()
                        ])
                    )
                    
                    val_test_transform = AlbumentationsTransform(
                        A.Compose([
                            A.Resize(224, 224),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2()
                        ])
                    )
                    
                    # Buat datasets
                    with st.spinner("Memuat dataset..."):
                        train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
                        val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_test_transform)
                        test_dataset = datasets.ImageFolder(root=str(test_dir), transform=val_test_transform)
                        
                        # Buat data loaders
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                        
                        st.info(f"Dataset dimuat: {len(train_dataset)} training, {len(val_dataset)} validasi, {len(test_dataset)} test")
                    
                    # Inisialisasi model
                    model = models.resnet18(pretrained=True)
                    model.fc = nn.Linear(model.fc.in_features, 3)  # 3 kelas: Jawa, Batak, Sunda
                    
                    # Loss function dan optimizer
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    
                    # Training loop
                    train_losses = []
                    val_losses = []
                    train_accs = []
                    val_accs = []
                    
                    # Progress bar dan status
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    metrics_text = st.empty()
                    loss_chart = st.empty()
                    
                    # Function to update loss chart
                    def update_loss_chart(train_losses, val_losses, train_accs, val_accs):
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                        
                        ax1.plot(train_losses, label='Train Loss')
                        ax1.plot(val_losses, label='Validation Loss')
                        ax1.set_xlabel('Epochs')
                        ax1.set_ylabel('Loss')
                        ax1.set_title('Training and Validation Loss')
                        ax1.legend()
                        ax1.grid(True)
                        
                        ax2.plot(train_accs, label='Train Accuracy')
                        ax2.plot(val_accs, label='Validation Accuracy')
                        ax2.set_xlabel('Epochs')
                        ax2.set_ylabel('Accuracy')
                        ax2.set_title('Training and Validation Accuracy')
                        ax2.legend()
                        ax2.grid(True)
                        
                        return fig
                    
                    # Training loop
                    for epoch in range(epochs):
                        status_text.text(f"Epoch {epoch+1}/{epochs}")
                        
                        # Training phase
                        model.train()
                        running_train_loss = 0.0
                        running_train_corrects = 0
                        train_samples = 0
                        
                        for inputs, labels in train_loader:
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            
                            running_train_loss += loss.item() * inputs.size(0)
                            _, preds = torch.max(outputs, 1)
                            running_train_corrects += torch.sum(preds == labels.data)
                            train_samples += labels.size(0)
                        
                        epoch_train_loss = running_train_loss / train_samples
                        epoch_train_acc = running_train_corrects.double() / train_samples
                        
                        # Validation phase
                        model.eval()
                        running_val_loss = 0.0
                        running_val_corrects = 0
                        val_samples = 0
                        
                        with torch.no_grad():
                            for inputs, labels in val_loader:
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)
                                
                                running_val_loss += loss.item() * inputs.size(0)
                                _, preds = torch.max(outputs, 1)
                                running_val_corrects += torch.sum(preds == labels.data)
                                val_samples += labels.size(0)
                        
                        epoch_val_loss = running_val_loss / val_samples
                        epoch_val_acc = running_val_corrects.double() / val_samples
                        
                        # Update metrics
                        train_losses.append(epoch_train_loss)
                        val_losses.append(epoch_val_loss)
                        train_accs.append(epoch_train_acc.item())
                        val_accs.append(epoch_val_acc.item())
                        
                        # Update UI
                        progress_bar.progress((epoch + 1) / epochs)
                        metrics_text.text(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                                          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
                        
                        # Update loss chart every few epochs
                        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == epochs - 1:
                            loss_chart.pyplot(update_loss_chart(train_losses, val_losses, train_accs, val_accs))
                    
                    # Selesai training
                    status_text.text("Training selesai!")
                    
                    # Simpan model
                    torch.save(model.state_dict(), str(model_path))
                    st.success(f"Model berhasil disimpan ke {model_path}")
                    
                    # Evaluasi model pada test set
                    st.subheader("Evaluasi Model pada Test Set")
                    
                    model.eval()
                    all_preds = []
                    all_labels = []
                    
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            all_preds.extend(preds.numpy())
                            all_labels.extend(labels.numpy())
                    
                    # Confusion matrix
                    cm = confusion_matrix(all_labels, all_preds)
                    class_names = ['Jawa', 'Batak', 'Sunda']
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
                    plt.xlabel("Predicted")
                    plt.ylabel("True")
                    plt.title("Confusion Matrix")
                    st.pyplot(fig)
                    
                    # Classification report
                    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.write("Classification Report:")
                    st.dataframe(report_df)
        
        with tab2:
            st.subheader("Prediksi Suku/Etnis")
            
            # Initialize face detector
            detector = load_face_detector()
            
            # Check if model exists
            if not model_path.exists():
                st.warning("Model belum dilatih. Silakan latih model terlebih dahulu pada tab Training.")
            else:
                import torch 
                from torch import nn
                from torchvision import models, transforms
    
                # Load model
                model = models.resnet18(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, 3)  # 3 kelas: Jawa, Batak, Sunda
                model.load_state_dict(torch.load(str(model_path)))
                model.eval()
                
                # Define preprocessing for prediction
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                # Upload image
                uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png"])
                
                if uploaded_file is not None:
                    # Read image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Gambar Asli", width=300)
                    
                    # Convert to OpenCV format for face detection
                    img_array = np.array(image)
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Detect face
                    with st.spinner("Mendeteksi wajah..."):
                        faces = detector.detect_faces(img_array)
                    
                    if not faces:
                        st.error("Tidak dapat mendeteksi wajah pada gambar. Silakan coba gambar lain.")
                    else:
                        # Get first face
                        face = faces[0]
                        x, y, w, h = face['box']
                        face_img_cv = img_cv[y:y+h, x:x+w]
                        face_img = Image.fromarray(cv2.cvtColor(face_img_cv, cv2.COLOR_BGR2RGB))
                        
                        # Show cropped face
                        st.image(face_img, caption="Wajah Terdeteksi", width=200)
                        
                        # Preprocess for model
                        input_tensor = preprocess(face_img).unsqueeze(0)
                        
                        # Predict
                        with st.spinner("Memprediksi suku/etnis..."):
                            with torch.no_grad():
                                output = model(input_tensor)
                                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                                predicted_idx = torch.argmax(probabilities).item()
                                
                                class_names = ['Jawa', 'Batak', 'Sunda']
                                predicted_class = class_names[predicted_idx]
                                confidence = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
                        
                        # Show prediction
                        st.subheader(f"Prediksi: {predicted_class}")
                        
                        # Show confidence scores
                        st.subheader("Confidence Scores")
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 5))
                        bars = ax.bar(list(confidence.keys()), list(confidence.values()), color=['#FF9999', '#66B2FF', '#99FF99'])
                        ax.set_ylim(0, 1.0)
                        ax.set_ylabel('Confidence Score')
                        ax.set_title('Confidence Scores per Suku/Etnis')
                        
                        # Add values on top of bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                        
                        # Interpretation
                        max_confidence = max(confidence.values())
                        if max_confidence > 0.7:
                            st.success(f"Wajah ini diprediksi sebagai suku {predicted_class} dengan confidence yang tinggi ({max_confidence:.2f}).")
                        elif max_confidence > 0.4:
                            st.info(f"Wajah ini diprediksi sebagai suku {predicted_class}, tetapi confidence cukup rendah ({max_confidence:.2f}).")
                        else:
                            st.warning(f"Sistem tidak dapat memprediksi suku dengan confidence yang tinggi. Confidence tertinggi adalah {predicted_class} ({max_confidence:.2f}).")
    
    elif app_mode == "Analisis Bentuk Wajah":
        st.header("Analisis Bentuk Wajah Menggunakan Metode Komputer Vision")
        
        st.write("""
        Fitur ini memungkinkan analisis lebih mendalam terhadap bentuk wajah menggunakan metode:
        1. **Kode Rantai (Chain Code)** - Merepresentasikan kontur wajah sebagai rangkaian arah
        2. **Deteksi Tepi (Edge Detection)** - Mengekstraksi tepi-tepi signifikan pada wajah
        3. **Proyeksi Integral** - Menganalisis distribusi piksel untuk karakterisasi bentuk
        """)
        
        # Inisialisasi analyzer
        shape_analyzer = ShapeAnalyzer()
        
        # Upload image
        uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["Analisis Kontur", "Deteksi Tepi", "Proyeksi Integral"])
            
            with tab1:
                st.subheader("Analisis Kontur Wajah dengan Kode Rantai")
                
                # Detect face first using existing detector
                detector = load_face_detector()
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(img_rgb)
                
                if faces:
                    face = faces[0]
                    x, y, w, h = face['box']
                    face_img = image[y:y+h, x:x+w]
                    
                    # Show original face
                    st.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), caption="Wajah Terdeteksi")
                    
                    # Generate contour visualization
                    with st.spinner("Menganalisis kontur wajah..."):
                        fig = shape_analyzer.visualize_contour_analysis(face_img)
                        st.pyplot(fig)
                        
                    # Show shape metrics
                    metrics = shape_analyzer.analyze_facial_shape(face_img)
                    st.write("### Metrik Bentuk Wajah")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Aspect Ratio", f"{metrics['aspect_ratio']:.3f}")
                        st.metric("Circularity", f"{metrics['circularity']:.3f}")
                    
                    with col2:
                        st.metric("Convexity", f"{metrics['convexity']:.3f}")
                        st.metric("Area", f"{metrics['area']:.0f} piksel²")
                else:
                    st.error("Tidak dapat mendeteksi wajah pada gambar. Silakan coba gambar lain.")
            
            with tab2:
                st.subheader("Deteksi Tepi Wajah dengan Canny Edge Detection")
                
                # Parameters for Canny
                low_threshold = st.slider("Threshold Bawah", 0, 255, 50)
                high_threshold = st.slider("Threshold Atas", 0, 255, 150)
                
                # Detect face
                detector = load_face_detector()
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(img_rgb)
                
                if faces:
                    face = faces[0]
                    x, y, w, h = face['box']
                    face_img = image[y:y+h, x:x+w]
                    
                    # Generate edge visualization
                    with st.spinner("Mendeteksi tepi wajah..."):
                        fig = shape_analyzer.visualize_edges(face_img, low_threshold, high_threshold)
                        st.pyplot(fig)
                    
                    # Explanation
                    st.info("""
                    **Tentang Deteksi Tepi Canny:**
                    - **Threshold Bawah**: Pixel dengan gradient di bawah nilai ini tidak dianggap sebagai tepi
                    - **Threshold Atas**: Pixel dengan gradient di atas nilai ini selalu dianggap sebagai tepi
                    - Pixel antara kedua threshold ini dianggap tepi jika terhubung dengan tepi yang pasti
                    """)
                else:
                    st.error("Tidak dapat mendeteksi wajah pada gambar. Silakan coba gambar lain.")
            
            with tab3:
                st.subheader("Analisis Proyeksi Integral Wajah")
                
                # Detect face
                detector = load_face_detector()
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(img_rgb)
                
                if faces:
                    face = faces[0]
                    x, y, w, h = face['box']
                    face_img = image[y:y+h, x:x+w]
                    
                    # Generate projections visualization
                    with st.spinner("Menghitung proyeksi integral..."):
                        fig = shape_analyzer.visualize_projections(face_img)
                        st.pyplot(fig)
                    
                    # Explanation
                    st.info("""
                    **Tentang Proyeksi Integral:**
                    - **Proyeksi Horizontal**: Menunjukkan distribusi piksel sepanjang sumbu X (fitur vertikal)
                    - **Proyeksi Vertikal**: Menunjukkan distribusi piksel sepanjang sumbu Y (fitur horizontal)
                    - Berguna untuk mengidentifikasi garis-garis fitur utama pada wajah
                    """)
                else:
                    st.error("Tidak dapat mendeteksi wajah pada gambar. Silakan coba gambar lain.")

    # ========== ABOUT PAGE ==========
    elif app_mode == "Tentang":
        st.header("Tentang Sistem")
        
        st.markdown("""
        ### Sistem Pengenalan Wajah dan Deteksi Suku
        
        Sistem ini dikembangkan sebagai implementasi tugas praktikum dalam mata kuliah Pengolahan Citra Digital. 
        Sistem menggunakan teknik Computer Vision dan Deep Learning untuk mendeteksi wajah, 
        membandingkan kemiripan wajah, dan mengklasifikasikan suku/etnis.
        
        ### Algoritma yang Digunakan
        
        1. **Deteksi Wajah**: MTCNN (Multi-task Cascaded Convolutional Networks)
           - Deep learning framework dengan tiga tahap deteksi (P-Net, R-Net, O-Net)
           - Akurasi tinggi, mendeteksi landmark wajah, robust terhadap pose
        
        2. **Perbandingan Wajah**: FaceNet
           - Model yang menghasilkan embedding wajah 128-dimensional
           - Embeddings yang kompak dan diskriminatif, akurasi tinggi
        
        3. **Deteksi Suku/Etnis**: CNN dengan Transfer Learning
           - Memanfaatkan model pre-trained (ResNet18) dan fine-tuning untuk klasifikasi etnis
           - Performa tinggi, tidak perlu training dari awal, cocok untuk dataset terbatas
        
        ### Dataset
        
        Dataset yang digunakan terdiri dari:
        - Jumlah subjek: 15 orang berbeda
        - Kondisi gambar: Dekat, Indoor, Jauh, Outdoor, Senyum, Serius
        - Variasi: 6 gambar per orang dengan kondisi berbeda
        - Keragaman etnis: 3 suku/etnis (Jawa, Batak, Sunda)
        
        ### Tim Pengembang
        
        Aplikasi ini dikembangkan oleh kelompok 2A_018_025_026 untuk memenuhi tugas praktikum mata kuliah Pengolahan Citra Digital.
        """)
        
        # Display system architecture
        st.subheader("Arsitektur Sistem")
        
        # Create system architecture diagram
        system_architecture = """
        graph TD
            A[Input Gambar] --> B[MTCNN Face Detection]
            B --> C{Fitur yang Digunakan}
            C --> D[Face Similarity - FaceNet]
            C --> E[Ethnic Detection - ResNet18]
            D --> F[Hasil Perbandingan Wajah]
            E --> G[Hasil Prediksi Suku/Etnis]
        """
        
        st.graphviz_chart(system_architecture)

# Entry point
if __name__ == "__main__":
    main()