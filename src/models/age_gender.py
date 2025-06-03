import cv2
import numpy as np
import logging
from typing import Tuple, Dict
from collections import deque

logger = logging.getLogger("face_analysis")

class AgeGenderPredictor:
    def __init__(self):
        """Initialize age and gender prediction using advanced heuristics."""
        self.history = {}
        self.history_length = 5
        logger.info("Age and gender prediction initialized")

    def _get_face_id(self, bbox: Tuple[int, int, int, int]) -> str:
        """Generate a face ID based on position."""
        x, y, w, h = bbox
        return f"{x//10}_{y//10}"

    def _extract_features(self, face_roi: np.ndarray) -> Dict:
        """Extract comprehensive facial features."""
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Basic features
        features = {
            'avg_intensity': np.mean(gray),
            'std_intensity': np.std(gray),
            'face_ratio': face_roi.shape[1] / face_roi.shape[0],  # width/height ratio
        }
        
        # Texture features using gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features['texture_energy'] = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        # Region-specific features
        h, w = gray.shape
        forehead = gray[0:h//3, w//4:3*w//4]
        cheeks = gray[h//3:2*h//3, :]
        jaw = gray[2*h//3:, :]
        
        features.update({
            'forehead_smoothness': np.std(forehead),
            'cheek_contrast': np.std(cheeks),
            'jaw_definition': np.mean(np.abs(cv2.Sobel(jaw, cv2.CV_64F, 0, 1)))
        })
        
        return features

    def _estimate_age_gender(self, face_roi: np.ndarray) -> Tuple[str, str]:
        """Estimate age and gender using advanced facial features."""
        features = self._extract_features(face_roi)
        
        # Gender estimation using multiple features
        gender_score = 0
        gender_score += 1 if features['face_ratio'] > 0.78 else -1  # Face shape
        gender_score += 1 if features['jaw_definition'] > 25 else -1  # Jaw definition
        gender_score += 1 if features['texture_energy'] > 15 else -1  # Facial texture
        
        gender = "Man" if gender_score > 0 else "Woman"
        
        # Age estimation using multiple features
        # Younger faces typically have:
        # - Smoother skin (lower texture energy)
        # - More uniform regions (lower std_intensity)
        # - Smoother forehead
        age_score = (
            features['texture_energy'] * 0.4 +
            features['std_intensity'] * 0.3 +
            features['forehead_smoothness'] * 0.3
        )
        
        # Age brackets based on combined score
        if age_score < 12:
            age = "15-25"
        elif age_score < 18:
            age = "25-35"
        elif age_score < 25:
            age = "35-45"
        elif age_score < 35:
            age = "45-55"
        else:
            age = "55+"
        
        return age, gender

    def predict(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, str]:
        """Predict age and gender with temporal smoothing."""
        try:
            face_id = self._get_face_id(bbox)
            
            # Initialize history for new faces
            if face_id not in self.history:
                self.history[face_id] = {
                    'age': deque(maxlen=self.history_length),
                    'gender': deque(maxlen=self.history_length)
                }
            
            # Extract face ROI
            x, y, w, h = bbox
            face_roi = frame[y:y+h, x:x+w]
            
            # Ensure minimum size and square-ish aspect ratio
            if (face_roi.size == 0 or w < 20 or h < 20 or 
                w/h < 0.5 or w/h > 2.0):  # Filter out poor detections
                return "Unknown", "Unknown"
            
            # Get predictions
            age, gender = self._estimate_age_gender(face_roi)
            
            # Add to history
            self.history[face_id]['age'].append(age)
            self.history[face_id]['gender'].append(gender)
            
            # Get smoothed predictions with longer history for gender
            if len(self.history[face_id]['age']) > 0:
                # Most common age prediction
                age_counts = {}
                for a in self.history[face_id]['age']:
                    age_counts[a] = age_counts.get(a, 0) + 1
                smoothed_age = max(age_counts.items(), key=lambda x: x[1])[0]
                
                # Most common gender prediction
                gender_counts = {}
                for g in self.history[face_id]['gender']:
                    gender_counts[g] = gender_counts.get(g, 0) + 1
                smoothed_gender = max(gender_counts.items(), key=lambda x: x[1])[0]
                
                return smoothed_age, smoothed_gender
            
            return age, gender
            
        except Exception as e:
            logger.error(f"Error in age/gender prediction: {str(e)}")
            return "Unknown", "Unknown"

    def cleanup(self):
        """Clean up resources."""
        self.history.clear() 