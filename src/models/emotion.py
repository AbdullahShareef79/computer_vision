import cv2
import numpy as np
import tensorflow as tf
import logging
from typing import Tuple
from pathlib import Path
import gdown

logger = logging.getLogger("face_analysis")

class EmotionDetector:
    def __init__(self):
        """Initialize emotion detection model."""
        self.model = None
        self.input_size = (48, 48)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Download and load model
        self._setup_model()
        
    def _setup_model(self) -> None:
        """Download and load the pre-trained emotion detection model."""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / "emotion_model.h5"
        
        try:
            if not model_path.exists():
                logger.info("Downloading emotion detection model...")
                gdown.download(
                    "https://drive.google.com/uc?id=1-L3LnxVXv4vByg_hqxXMZPvnKn2ZZNnk",
                    str(model_path)
                )
            
            # Load model
            self.model = tf.keras.models.load_model(str(model_path))
            logger.info("Emotion detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error setting up emotion detection model: {str(e)}")
            raise
            
    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face image for emotion detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        resized = cv2.resize(gray, self.input_size)
        
        # Normalize pixel values
        normalized = resized.astype('float32') / 255.0
        
        # Reshape for model input
        return np.expand_dims(np.expand_dims(normalized, -1), 0)
        
    def detect(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """Detect emotion in a face image.
        
        Args:
            frame: Input BGR image
            bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Predicted emotion label
        """
        try:
            x, y, w, h = bbox
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size == 0:
                return "Unknown"
                
            # Preprocess
            preprocessed = self._preprocess_face(face_img)
            
            # Predict emotion
            predictions = self.model.predict(preprocessed, verbose=0)[0]
            emotion_idx = predictions.argmax()
            
            return self.emotions[emotion_idx]
            
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            return "Unknown" 