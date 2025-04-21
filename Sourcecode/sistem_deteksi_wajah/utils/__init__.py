"""
Utility functions for the face detection system
"""
import os
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc

def create_folder_structure(base_path: str):
    """
    Create folder structure for the project
    
    Args:
        base_path: Base directory path
    """
    # Main data folders
    os.makedirs(os.path.join(base_path, 'data', 'raw'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'data', 'cropped'), exist_ok=True)
    
    # Dataset split folders
    splits = ['Train', 'Val', 'Test']
    labels = ['Jawa', 'Batak', 'Sunda']
    
    for split in splits:
        for label in labels:
            os.makedirs(os.path.join(base_path, 'data', split, label), exist_ok=True)
    
    # Models folder
    os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)

def get_image_pairs(image_folder: str) -> pd.DataFrame:
    """
    Generate all possible pairs of images from a folder
    
    Args:
        image_folder: Folder containing images
        
    Returns:
        DataFrame with image pairs
    """
    # Get all image files
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Generate all possible pairs
    pairs = list(itertools.combinations(image_files, 2))
    
    # Create DataFrame
    data = []
    for img1, img2 in pairs:
        name1 = img1.split('_')[0].lower()
        name2 = img2.split('_')[0].lower()
        actual = "Yes" if name1 == name2 else "No"
        data.append({'img1': img1, 'img2': img2, 'actual': actual})
    
    return pd.DataFrame(data)

def plot_similarity_distribution(df: pd.DataFrame):
    """
    Plot similarity score distribution
    
    Args:
        df: DataFrame with similarity scores
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='proba', hue='actual', kde=True, bins=50, palette=['red', 'green'])
    plt.title('Distribusi Skor Similaritas')
    plt.xlabel('Similarity Score')
    plt.ylabel('Jumlah Pasangan')
    plt.legend(title='Actual Label', labels=['Different (No)', 'Same (Yes)'])
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_roc_curve(y_true: List[int], y_scores: List[float], threshold: Optional[float] = None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels (0/1)
        y_scores: Predicted probabilities
        threshold: Optional threshold to highlight
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    
    # Add threshold point if provided
    if threshold is not None:
        # Find closest threshold index
        threshold_idx = np.argmin(np.abs(thresholds - threshold))
        plt.scatter(fpr[threshold_idx], tpr[threshold_idx], color='red', 
                   label=f'Threshold = {threshold:.2f}')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_confusion_matrix(cm: np.ndarray, classes: List[str]):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        classes: Class names
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
               xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    return plt

def plot_training_history(history: Dict):
    """
    Plot training history
    
    Args:
        history: Dictionary with training history
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    return plt