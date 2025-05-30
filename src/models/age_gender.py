import cv2
import numpy as np
import logging
from typing import Tuple
import os
import gdown
from pathlib import Path

logger = logging.getLogger("face_analysis")

class AgeGenderPredictor:
    def __init__(self):
        """Initialize age and gender prediction models."""
        self.age_model = None
        self.gender_model = None
        self.input_size = (64, 64)
        self.age_ranges = [
            (0, 2), (4, 6), (8, 12), (15, 20), (25, 32),
            (38, 43), (48, 53), (60, 100)
        ]
        self.gender_list = ['Female', 'Male']
        
        # Download and load models
        self._setup_models()
        
    def _setup_models(self) -> None:
        """Download and load the pre-trained models."""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Age model
        age_model_path = models_dir / "age_net.caffemodel"
        age_config_path = models_dir / "age_deploy.prototxt"
        
        # Gender model
        gender_model_path = models_dir / "gender_net.caffemodel"
        gender_config_path = models_dir / "gender_deploy.prototxt"
        
        try:
            # Download models if they don't exist
            if not age_model_path.exists():
                logger.info("Downloading age model...")
                gdown.download(
                    "https://drive.google.com/uc?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW",
                    str(age_model_path)
                )
            
            if not gender_model_path.exists():
                logger.info("Downloading gender model...")
                gdown.download(
                    "https://drive.google.com/uc?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ",
                    str(gender_model_path)
                )
            
            # Load models
            self.age_model = cv2.dnn.readNet(str(age_config_path), str(age_model_path))
            self.gender_model = cv2.dnn.readNet(str(gender_config_path), str(gender_model_path))
            
            logger.info("Age and gender models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error setting up age/gender models: {str(e)}")
            raise
            
    def _preprocess_face(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face image for model input."""
        blob = cv2.dnn.blobFromImage(
            face_img,
            1.0,
            self.input_size,
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )
        return blob
        
    def predict(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, str]:
        """Predict age and gender for a face.
        
        Args:
            frame: Input BGR image
            bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Tuple of (age_range, gender)
        """
        try:
            x, y, w, h = bbox
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size == 0:
                return "Unknown", "Unknown"
                
            # Preprocess
            blob = self._preprocess_face(face_img)
            
            # Age prediction
            self.age_model.setInput(blob)
            age_preds = self.age_model.forward()
            age_idx = age_preds[0].argmax()
            age_range = f"{self.age_ranges[age_idx][0]}-{self.age_ranges[age_idx][1]}"
            
            # Gender prediction
            self.gender_model.setInput(blob)
            gender_preds = self.gender_model.forward()
            gender_idx = gender_preds[0].argmax()
            gender = self.gender_list[gender_idx]
            
            return age_range, gender
            
        except Exception as e:
            logger.error(f"Error in age/gender prediction: {str(e)}")
            return "Unknown", "Unknown" 