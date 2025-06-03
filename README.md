# Advanced Face Analysis System

A real-time face analysis system that can detect faces and analyze various facial attributes including age, gender, and emotions. Built with Python and OpenCV, this system provides a lightweight and efficient solution for facial analysis without requiring complex deep learning models.

## Features

- **Face Detection**: Robust face detection using MTCNN
- **Age Estimation**: Predicts age ranges (15-25, 25-35, 35-45, 45-55, 55+)
- **Gender Detection**: Determines gender using multiple facial features
- **Emotion Recognition**: Detects 5 basic emotions:
  - Happy
  - Sad
  - Angry
  - Surprised
  - Neutral
- **Real-time Performance**: Optimized for real-time processing
- **Temporal Smoothing**: Reduces prediction jitter using historical data
- **Performance Monitoring**: Built-in FPS counter and system metrics

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- MTCNN
- MediaPipe

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install opencv-python numpy mtcnn mediapipe
```

## Usage

Run the main application:
```bash
python run.py
```

### Command Line Options

- `--no-age`: Disable age prediction
- `--no-gender`: Disable gender prediction
- `--no-emotion`: Disable emotion detection
- `--no-landmarks`: Disable facial landmarks
- `--gui`: Enable GUI controls

Example:
```bash
python run.py --no-landmarks --gui
```

## How It Works

### Face Detection
Uses MTCNN (Multi-task Cascaded Convolutional Networks) for reliable face detection across different poses and lighting conditions.

### Age & Gender Detection
Employs advanced image processing techniques:
- Texture analysis using gradient features
- Region-specific feature extraction (forehead, cheeks, jaw)
- Facial geometry measurements
- Temporal smoothing for stable predictions

### Emotion Detection
Uses a combination of:
- Facial feature gradients
- Intensity analysis
- Region-specific measurements
- Temporal pattern analysis

## Performance Considerations

- CPU usage is optimized for real-time processing
- Memory footprint is kept minimal
- Adjustable processing parameters for different hardware capabilities

## Known Limitations

- Age prediction accuracy may vary with lighting conditions
- Best results are achieved with front-facing poses
- Performance may vary based on hardware capabilities

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project Structure

```
src/
├── camera/         # Camera handling and video stream
├── models/         # AI models and inference
├── utils/          # Helper functions and utilities
└── app.py         # Main application entry point
```

## Acknowledgments

- MTCNN for face detection
- MediaPipe for facial landmarks
- TensorFlow and PyTorch communities
- OpenCV team

## Performance

The system is optimized to run on CPU while maintaining real-time performance:
- Face Detection: ~30 FPS
- Full Analysis: ~15-20 FPS (depending on hardware)
- Memory Usage: ~500MB-1GB

## Future Improvements

- [ ] Add face recognition capabilities
- [ ] Implement face tracking
- [ ] Add more emotion categories
- [ ] Support for multiple faces
- [ ] GPU acceleration support 