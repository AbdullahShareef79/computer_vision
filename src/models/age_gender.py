import cv2
import numpy as np
import logging
from typing import Tuple, Dict
from collections import deque

logger = logging.getLogger("face_analysis")

class AgeGenderPredictor:
    def __init__(self):
        """Initialize age and gender prediction with smoothing."""
        self.age_ranges = [
            (0, 2), (4, 6), (8, 12), (15, 20), (25, 32),
            (38, 43), (48, 53), (60, 100)
        ]
        self.gender_list = ['Female', 'Male']
        self.history = {}  # Store prediction history for each face
        self.history_length = 10  # Number of frames to consider for smoothing
        logger.info("Age and gender prediction initialized in demo mode")

    def _get_face_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Extract simple features from face region for consistent predictions."""
        x, y, w, h = bbox
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return 0.0, 0.0
        
        # Convert to grayscale
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate features
        avg_intensity = np.mean(gray_roi)
        texture = np.std(gray_roi)
        
        return avg_intensity, texture

    def _get_face_id(self, bbox: Tuple[int, int, int, int]) -> str:
        """Generate a simple face ID based on position."""
        x, y, w, h = bbox
        return f"{x//10}_{y//10}"  # Quantize position for stability

    def predict(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, str]:
        """Predict age and gender for a face with temporal smoothing.
        
        Args:
            frame: Input BGR image
            bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Tuple of (age_range, gender)
        """
        try:
            face_id = self._get_face_id(bbox)
            
            # Initialize history for new faces
            if face_id not in self.history:
                self.history[face_id] = {
                    'age': deque(maxlen=self.history_length),
                    'gender': deque(maxlen=self.history_length)
                }
            
            # Get face features
            intensity, texture = self._get_face_features(frame, bbox)
            
            # Map intensity to age index (higher intensity -> younger)
            age_idx = int((1.0 - intensity / 255.0) * (len(self.age_ranges) - 1))
            age_idx = max(0, min(age_idx, len(self.age_ranges) - 1))
            
            # Map texture to gender (higher texture -> male)
            gender_idx = int(texture > 50)  # Simple threshold
            
            # Add to history
            self.history[face_id]['age'].append(age_idx)
            self.history[face_id]['gender'].append(gender_idx)
            
            # Get most common predictions from history
            if len(self.history[face_id]['age']) > 0:
                age_counts = np.bincount(list(self.history[face_id]['age']))
                age_idx = np.argmax(age_counts)
                
                gender_counts = np.bincount(list(self.history[face_id]['gender']))
                gender_idx = np.argmax(gender_counts)
                
                age_range = self.age_ranges[age_idx]
                age_str = f"{age_range[0]}-{age_range[1]}"
                gender = self.gender_list[gender_idx]
                
                return age_str, gender
            
            return "Unknown", "Unknown"
            
        except Exception as e:
            logger.error(f"Error in age/gender prediction: {str(e)}")
            return "Unknown", "Unknown"

    def cleanup(self):
        """Clean up old face histories."""
        self.history.clear() 