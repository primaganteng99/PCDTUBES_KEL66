"""
Module for shape analysis and feature extraction from face contours
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import os

class ShapeAnalyzer:
    """
    Class for analyzing facial shape features using contour analysis
    """
    def __init__(self):
        """Initialize the shape analyzer"""
        pass
    
    def generate_freeman_chain_code(self, contour) -> List[int]:
        """
        Generate Freeman 8-direction Chain Code from an OpenCV contour
        
        Args:
            contour: Contour from cv2.findContours with CHAIN_APPROX_NONE
            
        Returns:
            List of chain code directions (0-7)
        """
        chain_code = []
        if len(contour) < 2:
            return chain_code  # Contour must have at least 2 points
        
        # Map (dx, dy) to Freeman direction code (Y-axis positive downward)
        directions = {
            (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
            (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7
        }
        
        for i in range(len(contour)):
            p1 = contour[i][0]  # Current point (format: [[x, y]])
            # Get next point, use modulo % to return to start point
            # on the last iteration (handles closed contours)
            p2 = contour[(i + 1) % len(contour)][0]
            
            dx = p2[0] - p1[0]  # X difference
            dy = p2[1] - p1[1]  # Y difference (Remember: Y positive downward)
            
            norm_dx = np.sign(dx)
            norm_dy = np.sign(dy)
            
            code = directions.get((norm_dx, norm_dy))
            if code is not None:
                chain_code.append(code)
        
        return chain_code
    
    def extract_face_contour(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract contour from a face image
        
        Args:
            face_img: Face image in BGR format
            
        Returns:
            Largest contour found or None if no contour is detected
        """
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()
        
        # Binarize the image
        _, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Return largest contour if any
        if contours:
            return max(contours, key=cv2.contourArea)
        return None
    
    def analyze_facial_shape(self, face_img: np.ndarray) -> Dict:
        """
        Analyze facial shape features
        
        Args:
            face_img: Face image in BGR format
            
        Returns:
            Dictionary with shape analysis results
        """
        # Extract contour
        contour = self.extract_face_contour(face_img)
        if contour is None:
            return {"error": "No facial contour detected"}
        
        # Generate chain code
        chain_code = self.generate_freeman_chain_code(contour)
        
        # Calculate basic shape metrics
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate shape circularity (1.0 for perfect circle)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Calculate convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = float(area) / hull_area if hull_area > 0 else 0
        
        return {
            "chain_code_length": len(chain_code),
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "convexity": convexity
        }
    
    def visualize_contour_analysis(self, face_img: np.ndarray, 
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize contour analysis results
        
        Args:
            face_img: Face image in BGR format
            save_path: Path to save the visualization (optional)
            
        Returns:
            Matplotlib figure with visualization
        """
        # Extract contour
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()
            
        _, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Create figure for visualization
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        
        # Plot original image
        if len(face_img.shape) == 3:
            axs[0, 0].imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        else:
            axs[0, 0].imshow(face_img, cmap='gray')
        axs[0, 0].set_title('Gambar Asli')
        axs[0, 0].axis('off')
        
        # Plot binary image
        axs[0, 1].imshow(binary_img, cmap='gray')
        axs[0, 1].set_title('Citra Biner')
        axs[0, 1].axis('off')
        
        # Initialize variables for result
        chain_code_str = "Tidak ada kontur ditemukan."
        img_contour_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Process contour and generate chain code if contours exist
        if contours:
            # Select largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Draw contour on BGR image for color visualization
            img_contour_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img_contour_display, [largest_contour], -1, (0, 255, 0), 1)
            
            # Generate chain code
            chain_code = self.generate_freeman_chain_code(largest_contour)
            
            # Calculate shape metrics
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Format chain code text for display
            chain_code_str = (
                f"Jumlah Kontur: {len(contours)}\n"
                f"Chain Code Length: {len(chain_code)}\n"
                f"Area: {area:.1f} pixels²\n"
                f"Perimeter: {perimeter:.1f} pixels\n"
                f"Circularity: {circularity:.3f}"
            )
        
        # Plot contour image
        axs[1, 0].imshow(cv2.cvtColor(img_contour_display, cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title('Kontur Terdeteksi')
        axs[1, 0].axis('off')
        
        # Display chain code and metrics
        axs[1, 1].axis('off')
        axs[1, 1].text(0.05, 0.95, chain_code_str, ha='left', va='top', fontsize=9)
        axs[1, 1].set_title('Analisis Bentuk')
        
        # Layout adjustments
        plt.tight_layout(pad=1.5)
        plt.suptitle("Analisis Kontur Wajah", fontsize=16)
        plt.subplots_adjust(top=0.92)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            
        return fig

    def calculate_projections(self, face_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate horizontal and vertical projections of a face image
        
        Args:
            face_img: Face image in BGR or grayscale format
            
        Returns:
            Tuple of (horizontal_projection, vertical_projection)
        """
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()
        
        # Binarize
        _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Normalize to 0-1
        binary_norm = binary_img / 255.0
        
        # Calculate projections
        # Horizontal projection (sum per column -> Vertical profile)
        horizontal_projection = np.sum(binary_norm, axis=0)
        
        # Vertical projection (sum per row -> Horizontal profile)
        vertical_projection = np.sum(binary_norm, axis=1)
        
        return horizontal_projection, vertical_projection
    
    def visualize_projections(self, face_img: np.ndarray, 
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize integral projections of a face image
        
        Args:
            face_img: Face image in BGR or grayscale format
            save_path: Path to save the visualization (optional)
            
        Returns:
            Matplotlib figure with visualization
        """
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()
        
        # Binarize
        _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Normalize to 0-1
        binary_norm = binary_img / 255.0
        
        # Calculate projections
        horizontal_projection = np.sum(binary_norm, axis=0)
        vertical_projection = np.sum(binary_norm, axis=1)
        
        # Get dimensions
        height, width = binary_norm.shape
        
        # Create figure with GridSpec for better layout control
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                             left=0.1, right=0.9, bottom=0.1, top=0.9,
                             wspace=0.05, hspace=0.05)
        
        # Axes for binary image (bottom left)
        ax_img = fig.add_subplot(gs[1, 0])
        ax_img.imshow(binary_norm, cmap='gray')
        ax_img.set_title('Wajah Biner')
        ax_img.set_xlabel('Indeks Kolom')
        ax_img.set_ylabel('Indeks Baris')
        
        # Axes for Horizontal Projection (above binary image)
        ax_hproj = fig.add_subplot(gs[0, 0], sharex=ax_img)
        ax_hproj.plot(np.arange(width), horizontal_projection, color='blue')
        ax_hproj.set_title('Proyeksi Horizontal (Profil Vertikal)')
        ax_hproj.set_ylabel('Jumlah Piksel')
        plt.setp(ax_hproj.get_xticklabels(), visible=False)
        ax_hproj.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Axes for Vertical Projection (right of binary image)
        ax_vproj = fig.add_subplot(gs[1, 1], sharey=ax_img)
        ax_vproj.plot(vertical_projection, np.arange(height), color='red')
        ax_vproj.set_title('Proyeksi Vertikal')
        ax_vproj.set_xlabel('Jumlah Piksel')
        ax_vproj.invert_yaxis()
        plt.setp(ax_vproj.get_yticklabels(), visible=False)
        ax_vproj.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Overall title
        plt.suptitle("Analisis Proyeksi Integral Wajah", fontsize=14)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def detect_edges(self, face_img: np.ndarray, 
                    low_threshold: int = 50, 
                    high_threshold: int = 150) -> np.ndarray:
        """
        Detect edges in a face image using Canny edge detector
        
        Args:
            face_img: Face image in BGR or grayscale format
            low_threshold: Lower threshold for Canny (default: 50)
            high_threshold: Higher threshold for Canny (default: 150)
            
        Returns:
            Edge map as binary image
        """
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()
        
        # Apply Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        return edges
    
    def visualize_edges(self, face_img: np.ndarray, 
                       low_threshold: int = 50, 
                       high_threshold: int = 150,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize edge detection on a face image
        
        Args:
            face_img: Face image in BGR or grayscale format
            low_threshold: Lower threshold for Canny (default: 50)
            high_threshold: Higher threshold for Canny (default: 150)
            save_path: Path to save the visualization (optional)
            
        Returns:
            Matplotlib figure with visualization
        """
        # Ensure image is in BGR format for display
        if len(face_img.shape) == 2:
            display_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
        else:
            display_img = face_img.copy()
        
        # Convert to grayscale for processing
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        plt.title('Gambar Asli')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(blurred, cmap='gray')
        plt.title('Blur Gaussian')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(edges, cmap='gray')
        plt.title(f'Tepi Canny (Th={low_threshold},{high_threshold})')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            
        return plt.gcf()