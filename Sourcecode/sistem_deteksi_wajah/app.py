"""
Sistem Pengenalan Wajah dan Deteksi Suku Menggunakan Computer Vision
Aplikasi Streamlit
"""
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from pathlib import Path
import tempfile

# Import modules
from modules.face_detection import FaceDetector
from modules.face_similarity import FaceSimilarity
from modules.ethnic_detection import EthnicDetector
import utils

# Set page configuration
st.set_page_config(
    page_title="Sistem Pengenalan Wajah dan Deteksi Suku",
    page_icon="🧑‍🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create base directories if they don't exist
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "temp"

for directory in [DATA_DIR, MODELS_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# Initialize modules
@st.cache_resource
def load_face_detector():
    return FaceDetector()

@st.cache_resource
def load_face_similarity_model():
    return FaceSimilarity(model_name='vggface2')

@st.cache_resource
def load_ethnic_detector():
    model_path = MODELS_DIR / "ethnic_detector.pth"
    if model_path.exists():
        return EthnicDetector(model_path=str(model_path))
    return EthnicDetector()

# App title and description
st.title("Sistem Pengenalan Wajah dan Deteksi Suku")
st.markdown("""
Aplikasi ini menggunakan Computer Vision dan Deep Learning untuk:
1. **Deteksi Wajah** - Menggunakan MTCNN untuk mendeteksi wajah dalam gambar
2. **Perbandingan Wajah** - Menggunakan FaceNet untuk membandingkan kemiripan dua wajah
3. **Deteksi Suku/Etnis** - Mengklasifikasikan wajah ke dalam kategori suku/etnis (Jawa, Batak, Sunda)
""")

# Sidebar navigation
st.sidebar.title("Navigasi")
app_mode = st.sidebar.selectbox(
    "Pilih Mode Aplikasi",
    ["Beranda", "Deteksi Wajah", "Perbandingan Wajah", "Deteksi Suku/Etnis", "Tentang"]
)

# Function to process uploaded image
def process_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    return None

# ========== HOME PAGE ==========
if app_mode == "Beranda":
    st.header("Selamat datang di Sistem Pengenalan Wajah dan Deteksi Suku")
    
    st.subheader("Tentang Sistem")
    st.write("""
    Sistem ini merupakan implementasi dari tugas praktikum pengembangan sistem pengenalan wajah dan 
    deteksi suku menggunakan teknik Computer Vision. Sistem ini dapat melakukan tiga fungsi utama:
    
    1. **Deteksi Wajah**: Mendeteksi wajah dalam gambar menggunakan algoritma MTCNN
    2. **Perbandingan Wajah**: Membandingkan dua wajah untuk menentukan kemiripan menggunakan FaceNet
    3. **Deteksi Suku/Etnis**: Mengklasifikasikan wajah ke dalam kategori suku/etnis menggunakan CNN
    """)
    
    st.subheader("Cara Penggunaan")
    st.write("""
    - Pilih mode aplikasi di sidebar sebelah kiri
    - Upload gambar yang ingin dianalisis
    - Ikuti petunjuk yang diberikan di setiap mode
    """)
    
    st.image("https://miro.medium.com/max/1400/1*uOQ4SUcanFVxsD9u4YKWpw.png", 
             caption="Ilustrasi Face Recognition System", use_column_width=True)

# ========== FACE DETECTION PAGE ==========
elif app_mode == "Deteksi Wajah":
    st.header("Deteksi Wajah dengan MTCNN")
    
    # Initialize face detector
    face_detector = load_face_detector()
    
    st.write("""
    Pada mode ini, sistem akan mendeteksi wajah dalam gambar menggunakan algoritma MTCNN 
    (Multi-task Cascaded Convolutional Networks). Algoritma ini dapat mendeteksi wajah dengan 
    berbagai pose, ekspresi, dan kondisi pencahayaan.
    """)
    
    # Upload image
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Process uploaded image
        image = process_uploaded_image(uploaded_file)
        
        if image is not None:
            # Show original image
            st.subheader("Gambar Asli")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Process image with face detection
            with st.spinner("Mendeteksi wajah..."):
                processed_image, faces = face_detector.process_image_for_display(image)
            
            # Show processed image
            st.subheader("Hasil Deteksi Wajah")
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Show face detection results
            st.subheader(f"Terdeteksi {len(faces)} wajah")
            
            if len(faces) > 0:
                # Display each detected face
                cols = st.columns(min(len(faces), 4))
                for i, (face, col) in enumerate(zip(faces, cols)):
                    x, y, w, h = face['box']
                    face_img = image[y:y+h, x:x+w]
                    col.image(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB), caption=f"Wajah {i+1}")
                    col.write(f"Confidence: {face['confidence']:.2f}")
                
                # Option to save cropped faces
                if st.button("Simpan Wajah Terdeteksi"):
                    temp_dir = tempfile.mkdtemp()
                    saved_paths = []
                    
                    for i, face in enumerate(faces):
                        x, y, w, h = face['box']
                        face_img = image[y:y+h, x:x+w]
                        
                        # Save face
                        save_path = os.path.join(temp_dir, f"face_{i+1}.jpg")
                        cv2.imwrite(save_path, face_img)
                        saved_paths.append(save_path)
                    
                    # Create zip file
                    import zipfile
                    zip_path = os.path.join(TEMP_DIR, "cropped_faces.zip")
                    
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for file in saved_paths:
                            zipf.write(file, os.path.basename(file))
                    
                    # Provide download link
                    with open(zip_path, "rb") as fp:
                        st.download_button(
                            label="Download Wajah Terdeteksi (ZIP)",
                            data=fp,
                            file_name="cropped_faces.zip",
                            mime="application/zip"
                        )

# ========== FACE SIMILARITY PAGE ==========
elif app_mode == "Perbandingan Wajah":
    st.header("Perbandingan Kemiripan Wajah dengan FaceNet")
    
    # Initialize face detector and similarity model
    face_detector = load_face_detector()
    face_similarity = load_face_similarity_model()
    
    st.write("""
    Pada mode ini, sistem akan membandingkan dua wajah untuk menentukan apakah keduanya berasal dari 
    orang yang sama. Sistem menggunakan algoritma FaceNet untuk mengekstrak fitur wajah dan 
    menghitung kemiripan antara dua wajah.
    """)
    
    # Adjust threshold
    threshold = st.slider(
        "Threshold Kemiripan", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5,
        help="Nilai threshold untuk menentukan apakah dua wajah adalah orang yang sama. Nilai yang lebih tinggi berarti dua wajah harus sangat mirip untuk dianggap sama."
    )
    face_similarity.set_threshold(threshold)
    
    # Upload two images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Wajah 1")
        uploaded_file1 = st.file_uploader("Upload gambar wajah 1", type=["jpg", "jpeg", "png"], key="face1")
        
    with col2:
        st.subheader("Gambar Wajah 2")
        uploaded_file2 = st.file_uploader("Upload gambar wajah 2", type=["jpg", "jpeg", "png"], key="face2")
    
    if uploaded_file1 is not None and uploaded_file2 is not None:
        # Process uploaded images
        image1 = process_uploaded_image(uploaded_file1)
        image2 = process_uploaded_image(uploaded_file2)
        
        if image1 is not None and image2 is not None:
            # Show original images
            col1.image(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB), use_column_width=True)
            col2.image(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Detect faces
            with st.spinner("Mendeteksi wajah..."):
                faces1 = face_detector.detect_faces(image1)
                faces2 = face_detector.detect_faces(image2)
            
            if not faces1 or not faces2:
                st.error("Tidak dapat mendeteksi wajah pada salah satu atau kedua gambar. Silakan coba gambar lain.")
            else:
                # Get first face from each image
                face1 = face_detector.crop_face(image1, faces1[0])
                face2 = face_detector.crop_face(image2, faces2[0])
                
                # Show cropped faces
                st.subheader("Wajah Terdeteksi")
                display_col1, display_col2 = st.columns(2)
                display_col1.image(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB), caption="Wajah 1", use_column_width=True)
                display_col2.image(cv2.cvtColor(face2, cv2.COLOR_BGR2RGB), caption="Wajah 2", use_column_width=True)
                
                # Calculate similarity
                with st.spinner("Menghitung kemiripan..."):
                    similarity, _, _ = face_similarity.calculate_similarity(face1, face2)
                
                # Show similarity result
                st.subheader("Hasil Perbandingan")
                
                # Create metrics
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("Skor Kemiripan", f"{similarity:.2f}")
                metric_col2.metric("Hasil", "Sama" if similarity >= threshold else "Berbeda")
                
                # Create progress bar for similarity
                st.progress(similarity)
                
                # Show interpretation
                if similarity >= threshold:
                    st.success(f"Kedua wajah kemungkinan berasal dari orang yang sama (skor kemiripan: {similarity:.2f} ≥ {threshold:.2f}).")
                else:
                    st.warning(f"Kedua wajah kemungkinan berasal dari orang yang berbeda (skor kemiripan: {similarity:.2f} < {threshold:.2f}).")

# ========== ETHNIC DETECTION PAGE ==========
elif app_mode == "Deteksi Suku/Etnis":
    st.header("Deteksi Suku/Etnis dengan CNN")
    
    # Initialize face detector and ethnic detector
    face_detector = load_face_detector()
    ethnic_detector = load_ethnic_detector()
    
    st.write("""
    Pada mode ini, sistem akan mengklasifikasikan wajah ke dalam kategori suku/etnis 
    (Jawa, Batak, Sunda) menggunakan Convolutional Neural Network (CNN) dengan transfer learning.
    """)
    
    # Upload image
    uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Process uploaded image
        image = process_uploaded_image(uploaded_file)
        
        if image is not None:
            # Show original image
            st.subheader("Gambar Asli")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Detect face
            with st.spinner("Mendeteksi wajah..."):
                faces = face_detector.detect_faces(image)
            
            if not faces:
                st.error("Tidak dapat mendeteksi wajah pada gambar. Silakan coba gambar lain.")
            else:
                # Get first face
                face = face_detector.crop_face(image, faces[0])
                
                # Show cropped face
                st.subheader("Wajah Terdeteksi")
                st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), width=300)
                
                # Predict ethnicity
                with st.spinner("Memprediksi suku/etnis..."):
                    predicted_class, confidence = ethnic_detector.predict(face)
                
                # Show prediction result
                st.subheader("Hasil Prediksi Suku/Etnis")
                
                # Display prediction
                st.markdown(f"### Prediksi: **{predicted_class}**")
                
                # Display confidence scores
                st.subheader("Confidence Scores")
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                classes = list(confidence.keys())
                scores = list(confidence.values())
                
                bars = ax.bar(classes, scores, color=['#FF9999', '#66B2FF', '#99FF99'])
                ax.set_ylim(0, 1.0)
                ax.set_ylabel('Confidence Score')
                ax.set_title('Confidence Scores per Suku/Etnis')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Interpretation
                max_confidence = max(confidence.values())
                if max_confidence > 0.7:
                    st.success(f"Wajah ini diprediksi sebagai suku {predicted_class} dengan confidence yang tinggi ({max_confidence:.2f}).")
                elif max_confidence > 0.4:
                    st.info(f"Wajah ini diprediksi sebagai suku {predicted_class}, tetapi confidence cukup rendah ({max_confidence:.2f}).")
                else:
                    st.warning(f"Sistem tidak dapat memprediksi suku dengan confidence yang tinggi. Confidence tertinggi adalah {predicted_class} ({max_confidence:.2f}).")

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
    - Jumlah gambar per subjek: Minimal 4 citra wajah per orang
    - Variasi: Ekspresi wajah, sudut pengambilan, pencahayaan, dan jarak pengambilan
    - Keragaman etnis: 3 suku/etnis (Jawa, Batak, Sunda)
    
    ### Tim Pengembang
    
    Aplikasi ini dikembangkan oleh kelompok 2A_018_025_026 untuk memenuhi tugas ETS mata kuliah Pengolahan Citra Digital.
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

# Add footer
st.markdown("---")
st.markdown("© 2023 Sistem Pengenalan Wajah dan Deteksi Suku | Kelompok 2A_018_025_026")