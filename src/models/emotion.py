import cv2
import numpy as np
import logging
from typing import Tuple, Dict
from collections import deque

logger = logging.getLogger("face_analysis")

class EmotionDetector:
    def __init__(self):
        """Initialize emotion detection with smoothing."""
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.history = {}  # Store prediction history for each face
        self.history_length = 10  # Number of frames to consider for smoothing
        logger.info("Emotion detection initialized in demo mode")

    def _get_face_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Extract simple features from face region for consistent predictions."""
        x, y, w, h = bbox
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return 0.0
        
        # Convert to grayscale and calculate average intensity
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        return np.mean(gray_roi)

    def _get_face_id(self, bbox: Tuple[int, int, int, int]) -> str:
        """Generate a simple face ID based on position."""
        x, y, w, h = bbox
        return f"{x//10}_{y//10}"  # Quantize position for stability

    def detect(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """Detect emotion in a face image with temporal smoothing.
        
        Args:
            frame: Input BGR image
            bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Predicted emotion label
        """
        try:
            face_id = self._get_face_id(bbox)
            
            # Initialize history for new faces
            if face_id not in self.history:
                self.history[face_id] = deque(maxlen=self.history_length)
            
            # Get face features
            feature = self._get_face_features(frame, bbox)
            
            # Map feature value to emotion index
            emotion_idx = int((feature / 255.0) * (len(self.emotions) - 1))
            emotion_idx = max(0, min(emotion_idx, len(self.emotions) - 1))
            
            # Add to history
            self.history[face_id].append(emotion_idx)
            
            # Get most common emotion from history
            if len(self.history[face_id]) > 0:
                counts = np.bincount(list(self.history[face_id]))
                emotion_idx = np.argmax(counts)
                
            return self.emotions[emotion_idx]
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            return "Unknown"

    def cleanup(self):
        """Clean up old face histories."""
        self.history.clear() 