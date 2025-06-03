import cv2
import numpy as np
import logging
from typing import Tuple, Dict, List
from collections import deque
from deepface import DeepFace

logger = logging.getLogger("face_analysis")

class EmotionDetector:
    def __init__(self):
        """Initialize emotion detection with DeepFace and temporal smoothing."""
        self.emotions = ['neutral', 'happy', 'sad', 'surprise', 'angry', 'fear', 'disgust']
        self.history = {}  # Store prediction history for each face
        self.history_length = 5  # Number of frames to keep in history
        logger.info("Emotion detection initialized with DeepFace")

    def _get_face_id(self, bbox: Tuple[int, int, int, int]) -> str:
        """Generate a stable face ID based on position."""
        x, y, w, h = bbox
        center_x, center_y = x + w//2, y + h//2
        return f"{center_x//20}_{center_y//20}"  # More stable quantization

    def detect(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """Detect emotion in a face image using DeepFace with temporal smoothing.
        
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
            
            # Extract face region with margin
            x, y, w, h = bbox
            margin = 20
            face_region = frame[
                max(0, y-margin):min(frame.shape[0], y+h+margin),
                max(0, x-margin):min(frame.shape[1], x+w+margin)
            ]
            
            # Convert to RGB (DeepFace expects RGB)
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Analyze emotions using DeepFace
            result = DeepFace.analyze(
                rgb_face,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Get dominant emotion
            emotion = result[0]['dominant_emotion'].lower()
            
            # Add to history
            self.history[face_id].append(emotion)
            
            # Get most common emotion from history for stability
            if len(self.history[face_id]) > 0:
                stable_emotion = max(set(self.history[face_id]), key=list(self.history[face_id]).count)
                return stable_emotion.capitalize()
            
            return emotion.capitalize()
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            return "Neutral"  # Default to neutral on error

    def cleanup(self):
        """Clean up resources."""
        self.history.clear() 