"""
Script untuk training model deteksi suku/etnis
Menggunakan ResNet18 dengan transfer learning untuk klasifikasi 3 etnis (Jawa, Batak, Sunda)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# Path untuk dataset
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Parameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 3  # Jawa, Batak, Sunda
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AlbumentationsTransform:
    """Class untuk mengaplikasikan transformasi Albumentations pada dataset PyTorch"""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, img):
        # Convert PIL image to NumPy array
        image_np = np.array(img)
        
        # Apply Albumentations transform
        augmented = self.transform(image=image_np)
        image = augmented['image']
        
        return image

def create_transformations():
    """Buat transformasi untuk dataset training dan validasi/testing"""
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
    
    return train_transform, val_test_transform

def load_datasets_and_create_loaders(train_transform, val_test_transform):
    """Load dataset dan buat data loaders"""
    train_dir = DATA_DIR / "Train"
    val_dir = DATA_DIR / "Val"
    test_dir = DATA_DIR / "Test"
    
    if not (train_dir.exists() and val_dir.exists() and test_dir.exists()):
        raise ValueError("Dataset directories not found. Please run preprocessing script first.")
    
    # Create datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_names = train_dataset.classes
    
    return train_loader, val_loader, test_loader, class_names

def initialize_model():
        # Saat inisialisasi model dalam tab Training
    model = models.resnet18(pretrained=True)

    # Unfreeze beberapa layer sebelum layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Tambahkan dropout untuk regularisasi
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Dropout untuk regularisasi
        nn.Linear(model.fc.in_features, 3)  # 3 kelas: Jawa, Batak, Sunda
    )

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """Train model dan evaluasi pada validation set"""
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        running_train_corrects = 0
        train_samples = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_train_corrects += torch.sum(preds == labels.data).item()
            train_samples += labels.size(0)
        
        # Calculate epoch metrics for training
        epoch_train_loss = running_train_loss / train_samples
        epoch_train_acc = running_train_corrects / train_samples
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_val_corrects += torch.sum(preds == labels.data).item()
                val_samples += labels.size(0)
        
        # Calculate epoch metrics for validation
        epoch_val_loss = running_val_loss / val_samples
        epoch_val_acc = running_val_corrects / val_samples
        
        # Save metrics
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
    
    return model, train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, class_names):
    """Evaluasi model pada test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on test set"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return cm, all_labels, all_preds

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def main():
    """Main function to train and evaluate the model"""
    print(f"Using device: {DEVICE}")
    
    # Create transformations
    train_transform, val_test_transform = create_transformations()
    
    # Load datasets and create loaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_names = load_datasets_and_create_loaders(
        train_transform, val_test_transform
    )
    print(f"Classes: {class_names}")
    
    # Initialize model
    print("Initializing model...")
    model = initialize_model()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )
    
    # Save model
    model_path = MODELS_DIR / "ethnic_detector.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training metrics
    plot_training_metrics(train_losses, val_losses, train_accs, val_accs)
    
    # Evaluate model on test set
    print("Evaluating model on test set...")
    cm, all_labels, all_preds = evaluate_model(model, test_loader, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()