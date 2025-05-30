import argparse
import sys
import cv2
import logging
from rich.logging import RichHandler

# Local imports
from camera.capture import WebcamCapture
from models.face_detector import FaceDetector
from models.age_gender import AgeGenderPredictor
from models.emotion import EmotionDetector
from models.landmarks import FacialLandmarks
from utils.fps_counter import FPSCounter
from utils.visualization import Visualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("face_analysis")

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
        
        # Initialize components
        logger.info("Initializing camera...")
        self.camera = WebcamCapture()
        
        logger.info("Loading face detector...")
        self.face_detector = FaceDetector()
        
        if not args.no_age or not args.no_gender:
            logger.info("Loading age/gender predictor...")
            self.age_gender_predictor = AgeGenderPredictor()
            
        if not args.no_emotion:
            logger.info("Loading emotion detector...")
            self.emotion_detector = EmotionDetector()
            
        if not args.no_landmarks:
            logger.info("Loading facial landmarks detector...")
            self.landmarks_detector = FacialLandmarks()

    def run(self):
        logger.info("Starting video capture...")
        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    break
                
                self.fps_counter.update()
                
                # Detect faces
                faces = self.face_detector.detect(frame)
                
                # Process each face
                for face_box in faces:
                    # Run enabled analyses
                    results = {"bbox": face_box}
                    
                    if not self.args.no_age or not self.args.no_gender:
                        age, gender = self.age_gender_predictor.predict(frame, face_box)
                        results.update({"age": age, "gender": gender})
                    
                    if not self.args.no_emotion:
                        emotion = self.emotion_detector.detect(frame, face_box)
                        results.update({"emotion": emotion})
                    
                    if not self.args.no_landmarks:
                        landmarks = self.landmarks_detector.detect(frame, face_box)
                        results.update({"landmarks": landmarks})
                    
                    # Visualize results
                    self.visualizer.draw(frame, results)
                
                # Add FPS counter
                self.visualizer.draw_fps(frame, self.fps_counter.get_fps())
                
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
        logger.info("Cleaning up...")
        self.camera.release()
        cv2.destroyAllWindows()

def main():
    args = parse_args()
    app = FaceAnalysisApp(args)
    app.run()

if __name__ == "__main__":
    main() 