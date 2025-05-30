# Advanced Face Analysis System

A real-time facial analysis system that demonstrates advanced computer vision and deep learning capabilities. This project showcases integration of multiple AI models for face detection, age estimation, gender prediction, emotion recognition, and facial landmark detection.

## 🌟 Features

- **Real-time Face Detection**: Using MTCNN/MediaPipe for accurate face detection
- **Age Estimation**: Deep learning-based age prediction
- **Gender Classification**: Real-time gender prediction
- **Emotion Recognition**: Detection of 7 basic emotions
- **Facial Landmarks**: 468-point facial landmark detection and visualization
- **Performance Metrics**: Real-time FPS counter and model inference time
- **Modular Design**: Select which analyses to run via command line or GUI

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-analysis-system.git
   cd face-analysis-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. Run the main application:
   ```bash
   python src/app.py
   ```

2. Command line options:
   ```bash
   python src/app.py --no-age --no-gender  # Disable specific analyses
   python src/app.py --gui  # Launch with GUI interface
   python src/app.py --help  # Show all options
   ```

## 🏗️ Project Structure

```
src/
├── camera/         # Camera handling and video stream
├── models/         # AI models and inference
├── utils/          # Helper functions and utilities
└── app.py         # Main application entry point
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MTCNN for face detection
- MediaPipe for facial landmarks
- TensorFlow and PyTorch communities
- OpenCV team

## 📊 Performance

The system is optimized to run on CPU while maintaining real-time performance:
- Face Detection: ~30 FPS
- Full Analysis: ~15-20 FPS (depending on hardware)
- Memory Usage: ~500MB-1GB

## 🔮 Future Improvements

- [ ] Add face recognition capabilities
- [ ] Implement face tracking
- [ ] Add more emotion categories
- [ ] Support for multiple faces
- [ ] GPU acceleration support 