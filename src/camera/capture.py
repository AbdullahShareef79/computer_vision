import cv2
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger("face_analysis")

class WebcamCapture:
    def __init__(self, device_id: int = 0):
        """Initialize webcam capture.
        
        Args:
            device_id: Camera device index (default: 0 for primary webcam)
        """
        self.device_id = device_id
        self.cap = None
        self._initialize_capture()

    def _initialize_capture(self) -> None:
        """Initialize the video capture device."""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera device {self.device_id}")
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info(f"Camera initialized successfully (ID: {self.device_id})")
        except Exception as e:
            logger.error(f"Error initializing camera: {str(e)}")
            raise

    def read(self) -> Optional[np.ndarray]:
        """Read a frame from the webcam.
        
        Returns:
            numpy.ndarray: BGR image if successful, None otherwise
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Camera is not initialized")
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to grab frame from camera")
                return None
                
            return frame
        except Exception as e:
            logger.error(f"Error reading frame: {str(e)}")
            return None

    def release(self) -> None:
        """Release the video capture device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released") 