import cv2
import numpy as np
import logging
from typing import Tuple, Dict
from collections import deque

logger = logging.getLogger("face_analysis")

class EmotionDetector:
    def __init__(self):
        """Initialize emotion detection using simple heuristics."""
        self.history = {}
        self.history_length = 5
        logger.info("Emotion detection initialized")

    def _get_face_id(self, bbox: Tuple[int, int, int, int]) -> str:
        """Generate a face ID based on position."""
        x, y, w, h = bbox
        return f"{x//10}_{y//10}"

    def _estimate_emotion(self, face_roi: np.ndarray) -> str:
        """Estimate emotion using facial features."""
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate facial features
        features = {
            'avg_intensity': np.mean(gray),
            'std_intensity': np.std(gray),
            'gradient_y': np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1))),
            'gradient_x': np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0)))
        }
        
        # Analyze vertical gradient (useful for detecting smiles)
        mouth_region = gray[int(gray.shape[0]*0.6):int(gray.shape[0]*0.9), :]
        mouth_gradient = np.mean(np.abs(cv2.Sobel(mouth_region, cv2.CV_64F, 0, 1)))
        
        # Simple emotion classification based on features
        if mouth_gradient > 20:  # High vertical gradient in mouth region
            if features['std_intensity'] < 40:  # Smooth texture
                return "Happy"
            else:
                return "Surprised"
        else:  # Low vertical gradient
            if features['std_intensity'] > 45:  # High texture variation
                return "Angry"
            elif features['avg_intensity'] < 100:  # Darker regions
                return "Sad"
            else:
                return "Neutral"

    def detect(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """Detect emotion with temporal smoothing."""
        try:
            face_id = self._get_face_id(bbox)
            
            # Initialize history for new faces
            if face_id not in self.history:
                self.history[face_id] = deque(maxlen=self.history_length)
            
            # Extract face ROI
            x, y, w, h = bbox
            face_roi = frame[y:y+h, x:x+w]
            
            # Ensure minimum size
            if face_roi.size == 0 or w < 20 or h < 20:
                return "Unknown"
            
            # Get emotion prediction
            emotion = self._estimate_emotion(face_roi)
            
            # Add to history
            self.history[face_id].append(emotion)
            
            # Get most common emotion from history
            if len(self.history[face_id]) > 0:
                emotion_counts = {}
                for e in self.history[face_id]:
                    emotion_counts[e] = emotion_counts.get(e, 0) + 1
                
                smoothed_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
                return smoothed_emotion
            
            return emotion
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            return "Unknown"

    def cleanup(self):
        """Clean up resources."""
        self.history.clear() 