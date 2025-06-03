import argparse
import sys
import cv2
import logging
import time
import psutil
import threading
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from rich.logging import RichHandler
from queue import Queue
from typing import Optional, Dict, Any

# Local imports
from .camera.capture import WebcamCapture
from .models.face_detector import FaceDetector
from .models.age_gender import AgeGenderPredictor
from .models.emotion import EmotionDetector
from .models.landmarks import FacialLandmarks
from .utils.fps_counter import FPSCounter
from .utils.visualization import Visualizer

# Set up logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True),
        RotatingFileHandler(
            'face_analysis.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
    ]
)
logger = logging.getLogger("face_analysis")

class PerformanceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.frame_count = 0
        self.metrics: Dict[str, float] = {}
        
    def update(self):
        self.frame_count += 1
        self.metrics.update({
            'cpu_percent': self.process.cpu_percent(),
            'memory_percent': self.process.memory_percent(),
            'fps': self.frame_count / (time.time() - self.start_time)
        })
        
    def get_metrics(self) -> Dict[str, float]:
        return self.metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced Face Analysis System")
    parser.add_argument("--no-age", action="store_true", help="Disable age prediction")
    parser.add_argument("--no-gender", action="store_true", help="Disable gender prediction")
    parser.add_argument("--no-emotion", action="store_true", help="Disable emotion detection")
    parser.add_argument("--no-landmarks", action="store_true", help="Disable facial landmarks")
    parser.add_argument("--gui", action="store_true", help="Enable GUI controls")
    return parser.parse_args()

class FaceAnalysisApp:
    def __init__(self, args):
        self.args = args
        self.fps_counter = FPSCounter()
        self.visualizer = Visualizer()
        self.performance_monitor = PerformanceMonitor()
        self.frame_queue = Queue(maxsize=30)  # Buffer for 1 second at 30 fps
        self.running = False
        
        # Initialize components with retry mechanism
        self._init_components()

    def _init_components(self, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                logger.info("Initializing camera...")
                self.camera = WebcamCapture()
                
                logger.info("Loading face detector...")
                self.face_detector = FaceDetector()
                
                if not self.args.no_age or not self.args.no_gender:
                    logger.info("Loading age/gender predictor...")
                    self.age_gender_predictor = AgeGenderPredictor()
                    
                if not self.args.no_emotion:
                    logger.info("Loading emotion detector...")
                    self.emotion_detector = EmotionDetector()
                    
                if not self.args.no_landmarks:
                    logger.info("Loading facial landmarks detector...")
                    self.landmarks_detector = FacialLandmarks()
                    
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise RuntimeError("Failed to initialize components")
                time.sleep(1)

    @contextmanager
    def _frame_processing_timeout(self, timeout: float = 1.0):
        """Context manager for timing out frame processing."""
        timer = threading.Timer(timeout, lambda: sys.exit(1))
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

    def _process_frame(self, frame) -> Optional[Dict[str, Any]]:
        """Process a single frame with timeout protection."""
        try:
            with self._frame_processing_timeout():
                faces = self.face_detector.detect(frame)
                results = []
                
                for face_box in faces:
                    face_results = {"bbox": face_box}
                    
                    if not self.args.no_age or not self.args.no_gender:
                        age, gender = self.age_gender_predictor.predict(frame, face_box)
                        face_results.update({"age": age, "gender": gender})
                    
                    if not self.args.no_emotion:
                        emotion = self.emotion_detector.detect(frame, face_box)
                        face_results.update({"emotion": emotion})
                    
                    if not self.args.no_landmarks:
                        landmarks = self.landmarks_detector.detect(frame, face_box)
                        face_results.update({"landmarks": landmarks})
                    
                    results.append(face_results)
                
                return results
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            return None

    def run(self):
        logger.info("Starting video capture...")
        self.running = True
        
        try:
            while self.running:
                frame = self.camera.read()
                if frame is None:
                    logger.warning("Failed to capture frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                self.fps_counter.update()
                self.performance_monitor.update()
                
                # Process frame
                results = self._process_frame(frame)
                if results:
                    for result in results:
                        self.visualizer.draw(frame, result)
                
                # Add monitoring information
                metrics = self.performance_monitor.get_metrics()
                self.visualizer.draw_fps(frame, metrics['fps'])
                self.visualizer.draw_metrics(frame, metrics)
                
                # Display the frame
                cv2.imshow("Face Analysis", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.exception("An error occurred")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.running = False
        
        # Release camera
        if hasattr(self, 'camera'):
            try:
                self.camera.release()
            except Exception as e:
                logger.error(f"Error releasing camera: {str(e)}")
            
        # Clean up predictors
        for component in ['age_gender_predictor', 'emotion_detector']:
            if hasattr(self, component):
                try:
                    getattr(self, component).cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up {component}: {str(e)}")
        
        # Clean up windows
        cv2.destroyAllWindows()

def main():
    args = parse_args()
    app = FaceAnalysisApp(args)
    app.run()

if __name__ == "__main__":
    main() 