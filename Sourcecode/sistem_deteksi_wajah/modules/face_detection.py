"""
Module for face detection using MTCNN
"""
import os
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
from typing import Tuple, List, Dict, Optional, Union

class FaceDetector:
    """
    Face detection class using MTCNN algorithm
    """
    def __init__(self):
        """Initialize MTCNN detector"""
        self.detector = MTCNN()
        
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image
        
        Args:
            image: Input image in BGR format (OpenCV format)
            
        Returns:
            List of dictionaries with face detection results
        """
        # Convert BGR to RGB (MTCNN expects RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Detect faces
        faces = self.detector.detect_faces(image_rgb)
        return faces
    
    def crop_face(self, image: np.ndarray, face: Dict) -> np.ndarray:
        """
        Crop a face from an image based on bounding box
        
        Args:
            image: Input image in BGR format
            face: Face detection result dictionary
            
        Returns:
            Cropped face image
        """
        x, y, w, h = face['box']
        face_crop = image[y:y+h, x:x+w]
        return face_crop
    
    def crop_and_save_faces(self, image_path: str, output_folder: str) -> List[str]:
        """
        Detect, crop and save faces from an image
        
        Args:
            image_path: Path to input image
            output_folder: Folder to save cropped faces
            
        Returns:
            List of paths to saved face images
        """
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        # Detect faces
        faces = self.detect_faces(image)
        
        saved_paths = []
        # Crop and save each face
        for i, face in enumerate(faces):
            face_crop = self.crop_face(image, face)
            
            # Generate output filename
            filename = os.path.basename(image_path)
            name_part = os.path.splitext(filename)[0]
            save_path = os.path.join(output_folder, f"{name_part}_face{i+1}.jpg")
            
            # Save cropped face
            cv2.imwrite(save_path, face_crop)
            saved_paths.append(save_path)
            
        return saved_paths
    
    def process_image_for_display(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process image for display with face detection results
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (image with bounding boxes, list of face dictionaries)
        """
        # Make a copy to avoid modifying original
        display_image = image.copy()
        
        # Detect faces
        faces = self.detect_faces(image)
        
        # Draw bounding boxes
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw key points
            keypoints = face['keypoints']
            for point in keypoints.values():
                cv2.circle(display_image, point, 2, (0, 0, 255), 2)
                
        return display_image, faces
    
    def get_aligned_face(self, image: np.ndarray, face: Dict, target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
        """
        Get aligned face using facial landmarks
        
        Args:
            image: Input image
            face: Face dictionary from MTCNN
            target_size: Size to resize the face to
            
        Returns:
            Aligned face image
        """
        # Extract face box and keypoints
        x, y, w, h = face['box']
        keypoints = face['keypoints']
        
        # Get coordinates for left eye, right eye
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        
        # Calculate angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate center of face
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Crop face after rotation (with some margin)
        margin_w = int(0.3 * w)
        margin_h = int(0.3 * h)
        face_crop = rotated[max(0, y-margin_h):min(rotated.shape[0], y+h+margin_h), 
                            max(0, x-margin_w):min(rotated.shape[1], x+w+margin_w)]
        
        # Resize
        if face_crop.size > 0:  # Make sure the crop is not empty
            face_resized = cv2.resize(face_crop, target_size)
            return face_resized
        else:
            # If alignment fails, just crop and resize
            face_crop = image[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, target_size)
            return face_resized