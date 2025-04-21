"""
Module for face similarity using FaceNet
"""
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional, Union
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

class FaceSimilarity:
    """
    Face similarity comparison using FaceNet model
    """
    def __init__(self, model_name: str = 'vggface2'):
        """
        Initialize FaceNet model
        
        Args:
            model_name: Pre-trained model name ('vggface2' or 'casia-webface')
        """
        # Load model
        self.model = InceptionResnetV1(pretrained=model_name).eval()
        self.threshold = 0.5  # Default threshold
        
    def set_threshold(self, threshold: float):
        """
        Set similarity threshold
        
        Args:
            threshold: Threshold value between 0 and 1
        """
        self.threshold = threshold
        
    def preprocess_face(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for the model
        
        Args:
            face_img: Face image in BGR format
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Resize to 160x160 (FaceNet input size)
        resized = cv2.resize(face_rgb, (160, 160))
        
        # Convert to tensor and normalize
        tensor = torch.tensor(resized).permute(2, 0, 1).float() / 255.0
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Get face embedding from face image
        
        Args:
            face_img: Face image in BGR format
            
        Returns:
            Face embedding vector
        """
        # Preprocess face
        preprocessed = self.preprocess_face(face_img)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model(preprocessed)
            
        return embedding.numpy()
    
    def calculate_similarity(self, face_img1: np.ndarray, face_img2: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Calculate similarity between two face images
        
        Args:
            face_img1: First face image
            face_img2: Second face image
            
        Returns:
            Tuple of (similarity score, embedding1, embedding2)
        """
        # Get embeddings
        embedding1 = self.get_embedding(face_img1)
        embedding2 = self.get_embedding(face_img2)
        
        # Calculate cosine similarity
        sim = cosine_similarity(embedding1, embedding2)[0][0]
        
        # Normalize to 0-1 range (cosine similarity is between -1 and 1)
        normalized_sim = (sim + 1) / 2
        
        return normalized_sim, embedding1, embedding2
    
    def is_same_person(self, similarity_score: float) -> bool:
        """
        Determine if two faces are of the same person based on similarity score
        
        Args:
            similarity_score: Similarity score between 0 and 1
            
        Returns:
            True if same person, False otherwise
        """
        return similarity_score >= self.threshold
    
    def find_optimal_threshold(self, pairs_data: List[Dict]) -> float:
        """
        Find optimal threshold using ROC curve
        
        Args:
            pairs_data: List of dictionaries with 'actual' and 'proba' keys
            
        Returns:
            Optimal threshold value
        """
        from sklearn.metrics import roc_curve, f1_score
        import numpy as np
        
        # Extract actual labels and scores
        y_true = [1 if pair['actual'] == 'Yes' else 0 for pair in pairs_data]
        scores = [pair['proba'] for pair in pairs_data]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        
        # Calculate F1 score for each threshold
        f1_scores = []
        for threshold in thresholds:
            y_pred = [1 if score >= threshold else 0 for score in scores]
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)
            
        # Find threshold with highest F1 score
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold