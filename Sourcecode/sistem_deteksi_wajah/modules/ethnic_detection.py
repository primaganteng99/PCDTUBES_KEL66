"""
Module for ethnic/suku detection using CNN with Transfer Learning
"""
import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
from torch import nn
from typing import Tuple, List, Dict, Optional, Union

class EthnicDetector:
    """
    Ethnic/suku detection class using CNN with Transfer Learning
    """
    def __init__(self, model_path: Optional[str] = None, num_classes: int = 3):
        """
        Initialize ethnic detector
        
        Args:
            model_path: Path to pre-trained model
            num_classes: Number of ethnic classes
        """
        # Define class names
        self.class_names = ['Jawa', 'Batak', 'Sunda']
        
        # Initialize model (ResNet18)
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for the model
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def predict(self, image: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Predict ethnic class for a face image
        
        Args:
            image: Face image in BGR format
            
        Returns:
            Tuple of (predicted class, confidence scores dictionary)
        """
        # Preprocess image
        tensor = self.preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get predicted class and confidence
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = self.class_names[predicted_idx]
        
        # Create confidence dictionary
        confidence = {self.class_names[i]: float(probabilities[i]) for i in range(len(self.class_names))}
        
        return predicted_class, confidence
    
    def train_model(self, train_loader, val_loader, epochs=20, learning_rate=0.001, save_path=None):
        """
        Train the ethnic detection model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            save_path: Path to save trained model
        
        Returns:
            Dictionary with training history
        """
        # Set model to training mode
        self.model.train()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_train_loss = 0.0
            running_train_corrects = 0
            train_samples = 0
            
            for inputs, labels in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_train_corrects += torch.sum(preds == labels.data)
                train_samples += labels.size(0)
            
            # Calculate epoch metrics
            epoch_train_loss = running_train_loss / train_samples
            epoch_train_acc = running_train_corrects.double() / train_samples
            
            # Validation phase
            self.model.eval()
            running_val_loss = 0.0
            running_val_corrects = 0
            val_samples = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Statistics
                    running_val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    running_val_corrects += torch.sum(preds == labels.data)
                    val_samples += labels.size(0)
            
            # Calculate epoch metrics
            epoch_val_loss = running_val_loss / val_samples
            epoch_val_acc = running_val_corrects.double() / val_samples
            
            # Update history
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc.item())
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc.item())
            
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # Save model if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        
        # Set model back to evaluation mode
        self.model.eval()
        
        return history
    
    def evaluate_model(self, test_loader):
        """
        Evaluate model on test data
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Set model to evaluation mode
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        # Predict on test data
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=self.class_names, output_dict=True)
        
        # Return evaluation results
        return {
            'confusion_matrix': cm,
            'classification_report': report
        }