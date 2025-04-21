"""
Script untuk preprocessing dataset wajah dan persiapan dataset.
Script ini melakukan:
1. Deteksi dan crop wajah dari gambar di folder raw
2. Pembagian data menjadi train, validation, dan test
3. Penyusunan struktur folder berdasarkan etnis
"""
import os
import cv2
import numpy as np
from mtcnn import MTCNN
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Dictionary pemetaan nama orang ke etnisnya
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

# Path untuk dataset
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CROPPED_DIR = DATA_DIR / "cropped_mtcnn"
MODELS_DIR = BASE_DIR / "models"

# Membuat direktori jika belum ada
for directory in [DATA_DIR, RAW_DIR, CROPPED_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

def detect_and_crop_faces(image_path, detector, output_folder):
    """Deteksi dan crop wajah dari gambar"""
    print(f"Processing {image_path}")
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    
    if not faces:
        print(f"No face detected in {image_path}")
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
    
    print("Train set:", train_set)
    print("\nValidation set:", val_set)
    print("\nTest set:", test_set)
    
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
                    print(f"Copying {src_path} to {dst_path}")
                    shutil.copy(src_path, dst_path)
    
    return train_set, val_set, test_set

def main():
    """Fungsi utama untuk preprocessing dataset"""
    print("Starting dataset preprocessing...")
    
    # Inisialisasi MTCNN detector
    print("Initializing MTCNN detector...")
    detector = MTCNN()
    
    # Deteksi dan crop wajah
    print("\nDetecting and cropping faces...")
    raw_images = []
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                raw_images.append(os.path.join(root, file))
    
    print(f"Found {len(raw_images)} images in raw directory")
    
    # Crop wajah dengan progress bar
    cropped_faces = []
    for img_path in tqdm(raw_images, desc="Processing Images"):
        face_path = detect_and_crop_faces(img_path, detector, str(CROPPED_DIR))
        if face_path:
            cropped_faces.append(face_path)
    
    print(f"\nSuccessfully cropped {len(cropped_faces)} faces from {len(raw_images)} images")
    
    # Hapus file face yang bukan face1 (sesuai dengan kode asli)
    for filename in os.listdir(CROPPED_DIR):
        if not filename.endswith('_face1.jpg'):
            file_path = os.path.join(CROPPED_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Removed non-primary face: {filename}")
    
    # Split dataset dan sesuaikan struktur folder
    print("\nSplitting dataset and organizing folders...")
    train_set, val_set, test_set = process_dataset_and_split(CROPPED_DIR, ETHNICITY_MAPPING)
    
    print("\nDataset preprocessing completed!")
    print(f"- Cropped faces: {len(cropped_faces)}")
    print(f"- Training samples: {sum(len(os.listdir(os.path.join(DATA_DIR, 'Train', label.capitalize()))) for label in ['jawa', 'batak', 'sunda'])}")
    print(f"- Validation samples: {sum(len(os.listdir(os.path.join(DATA_DIR, 'Val', label.capitalize()))) for label in ['jawa', 'batak', 'sunda'])}")
    print(f"- Test samples: {sum(len(os.listdir(os.path.join(DATA_DIR, 'Test', label.capitalize()))) for label in ['jawa', 'batak', 'sunda'])}")

if __name__ == "__main__":
    main()