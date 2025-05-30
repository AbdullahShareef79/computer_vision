import cv2
import numpy as np
from typing import Dict, Any

class Visualizer:
    def __init__(self):
        """Initialize visualizer with color schemes and fonts."""
        self.colors = {
            'bbox': (0, 255, 0),  # Green
            'text_bg': (0, 0, 0),  # Black
            'text': (255, 255, 255),  # White
            'landmarks': (255, 0, 0),  # Red
            'fps': (0, 255, 255)  # Yellow
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
    def draw(self, frame: np.ndarray, results: Dict[str, Any]) -> None:
        """Draw detection results on the frame.
        
        Args:
            frame: Input BGR image
            results: Dictionary containing detection results
        """
        if 'bbox' in results:
            self._draw_bbox(frame, results['bbox'])
            
        if 'landmarks' in results and results['landmarks'] is not None:
            self._draw_landmarks(frame, results['landmarks'])
            
        # Draw text results above bbox
        text_results = []
        if 'age' in results:
            text_results.append(f"Age: {results['age']}")
        if 'gender' in results:
            text_results.append(f"Gender: {results['gender']}")
        if 'emotion' in results:
            text_results.append(f"Emotion: {results['emotion']}")
            
        if text_results and 'bbox' in results:
            self._draw_text_block(frame, text_results, results['bbox'])
    
    def _draw_bbox(self, frame: np.ndarray, bbox: tuple) -> None:
        """Draw bounding box on frame."""
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['bbox'], self.thickness)
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> None:
        """Draw facial landmarks on frame."""
        if landmarks is None:
            return
            
        for point in landmarks:
            cv2.circle(frame, tuple(point.astype(int)), 2, self.colors['landmarks'], -1)
    
    def _draw_text_block(self, frame: np.ndarray, text_lines: list, bbox: tuple) -> None:
        """Draw multiple lines of text above bounding box."""
        x, y, _, _ = bbox
        line_height = 25
        
        for i, text in enumerate(text_lines):
            text_y = y - (len(text_lines) - i) * line_height
            
            # Get text size
            (text_w, text_h), _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)
            
            # Draw background rectangle
            cv2.rectangle(frame, 
                        (x, text_y - text_h - 5),
                        (x + text_w + 10, text_y + 5),
                        self.colors['text_bg'],
                        -1)
            
            # Draw text
            cv2.putText(frame,
                       text,
                       (x + 5, text_y),
                       self.font,
                       self.font_scale,
                       self.colors['text'],
                       self.thickness)
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Draw FPS counter in top-left corner."""
        text = f"FPS: {fps:.1f}"
        cv2.putText(frame,
                   text,
                   (10, 30),
                   self.font,
                   self.font_scale,
                   self.colors['fps'],
                   self.thickness) 