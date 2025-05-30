import time
from collections import deque

class FPSCounter:
    def __init__(self, window_size: int = 30):
        """Initialize FPS counter with moving average.
        
        Args:
            window_size: Number of frames to average FPS over
        """
        self.frame_times = deque(maxlen=window_size)
        self.last_time = None
        
    def update(self) -> None:
        """Record the time of a new frame."""
        current_time = time.time()
        if self.last_time is not None:
            self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        
    def get_fps(self) -> float:
        """Calculate current FPS based on moving average.
        
        Returns:
            Current FPS value
        """
        if not self.frame_times:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0 