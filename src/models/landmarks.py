import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Tuple, Optional

logger = logging.getLogger("face_analysis")

class FacialLandmarks:
    def __init__(self):
        """Initialize MediaPipe Face Mesh for landmark detection."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        logger.info("MediaPipe Face Mesh initialized successfully")
        
    def detect(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Detect facial landmarks in the face region.
        
        Args:
            frame: Input BGR image
            bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Array of landmark coordinates or None if detection fails
        """
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
                
            # Extract landmarks for the first face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to numpy array
            landmarks = np.array([
                [int(point.x * frame.shape[1]), int(point.y * frame.shape[0])]
                for point in face_landmarks.landmark
            ])
            
            return landmarks
            
        except Exception as e:
            logger.error(f"Error in landmark detection: {str(e)}")
            return None
            
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close() 