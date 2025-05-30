import cv2
import numpy as np
from mtcnn import MTCNN
import logging
from typing import List, Tuple

logger = logging.getLogger("face_analysis")

class FaceDetector:
    def __init__(self, min_confidence: float = 0.9):
        """Initialize MTCNN face detector.
        
        Args:
            min_confidence: Minimum confidence threshold for face detection
        """
        self.min_confidence = min_confidence
        try:
            self.detector = MTCNN(min_face_size=20, scale_factor=0.709)
            logger.info("MTCNN face detector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MTCNN: {str(e)}")
            raise

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the input frame.
        
        Args:
            frame: Input BGR image
            
        Returns:
            List of face bounding boxes in format (x, y, width, height)
        """
        try:
            # Convert BGR to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = self.detector.detect_faces(rgb_frame)
            
            # Filter and convert to bounding boxes
            faces = []
            for detection in detections:
                if detection['confidence'] >= self.min_confidence:
                    x, y, w, h = detection['box']
                    faces.append((x, y, w, h))
            
            return faces
            
        except Exception as e:
            logger.error(f"Error during face detection: {str(e)}")
            return []

    def get_face_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract face region from frame using bounding box.
        
        Args:
            frame: Input BGR image
            bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Cropped face image
        """
        x, y, w, h = bbox
        return frame[y:y+h, x:x+w] 