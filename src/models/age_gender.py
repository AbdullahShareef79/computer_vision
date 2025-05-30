import cv2
import numpy as np
import logging
from typing import Tuple
import random

logger = logging.getLogger("face_analysis")

class AgeGenderPredictor:
    def __init__(self):
        """Initialize age and gender prediction in demo mode."""
        self.age_ranges = [
            (0, 2), (4, 6), (8, 12), (15, 20), (25, 32),
            (38, 43), (48, 53), (60, 100)
        ]
        self.gender_list = ['Female', 'Male']
        logger.info("Age and gender prediction initialized in demo mode")

    def predict(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, str]:
        """Predict age and gender for a face (demo mode).
        
        Args:
            frame: Input BGR image
            bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Tuple of (age_range, gender)
        """
        try:
            # In demo mode, return random predictions
            age_range = random.choice(self.age_ranges)
            age_str = f"{age_range[0]}-{age_range[1]}"
            gender = random.choice(self.gender_list)
            
            return age_str, gender
            
        except Exception as e:
            logger.error(f"Error in age/gender prediction: {str(e)}")
            return "Unknown", "Unknown" 