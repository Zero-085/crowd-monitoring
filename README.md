# AI-Powered Crowd Density & Safety Monitoring System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)

A real-time computer vision system for monitoring crowd density and detecting safety violations in public spaces using state-of-the-art deep learning.

## Overview

This project implements an intelligent crowd monitoring solution designed to enhance public safety in high-traffic areas such as railway stations, metro platforms, airports, and event venues. The system leverages YOLOv8 object detection to provide real-time people counting, zone-based risk assessment, and boundary violation detection.

### Key Features

- **Real-Time Detection**: Processes video streams at 30+ FPS with instant people detection and counting
- **Intelligent Risk Assessment**: Dynamic zone-based classification (Safe/Moderate/Danger) with configurable thresholds
- **Boundary Monitoring**: Virtual line detection for restricted area violations
- **Interactive Dashboard**: Web-based interface with live statistics, alert logging, and historical analytics
- **Multi-Source Support**: Compatible with webcams, IP cameras, and video files
- **Edge Computing Ready**: Optimized for deployment on affordable hardware (Raspberry Pi, Jetson Nano)

## Technical Architecture

```
┌─────────────────┐
│  Video Source   │ (Webcam/IP Camera/File)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  YOLOv8 Engine  │ (Object Detection)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analysis Module │ (Counting, Classification)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Dashboard     │ (Visualization & Alerts)
└─────────────────┘
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Webcam or video input source
- 4GB+ RAM recommended

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/crowd-monitoring.git
cd crowd-monitoring
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

The system will automatically download the YOLOv8 model (~6MB) on first run.

## Usage

### Quick Start Test

Verify your installation with a basic detection test:

```bash
python test_simple.py
```

### Core Detection Engine

Run the standalone detection system with visual interface:

```bash
python crowd_detector.py
```

**Controls:**
- `q`: Quit application
- `s`: Save screenshot

### Interactive Dashboard

Launch the full-featured web dashboard:

```bash
streamlit run app.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`

## Configuration

### Zone Thresholds

Adjust crowd density thresholds in `crowd_detector.py` or via the dashboard sidebar:

```python
self.green_threshold = 10   # Safe zone limit
self.yellow_threshold = 20  # Moderate zone limit
```

### Boundary Detection

Configure the virtual boundary line position (0.0 to 1.0, representing screen height percentage):

```python
self.set_boundary_line(frame_height, position=0.70)
```

### Video Source

```python
# Default webcam
detector.process_video(video_source=0)

# External USB camera
detector.process_video(video_source=1)

# Video file
detector.process_video(video_source='path/to/video.mp4')

# IP camera stream
detector.process_video(video_source='rtsp://camera_ip:port/stream')
```

## Project Structure

```
crowd-monitoring/
├── test_simple.py         # Basic YOLOv8 functionality test
├── crowd_detector.py      # Core detection engine with OpenCV interface
├── app.py                 # Streamlit web dashboard
├── requirements.txt       # Python dependencies
└── README.md             # Documentation
```

## Use Cases

- **Railway Stations**: Platform overcrowding detection and prevention
- **Metro Systems**: Real-time passenger density monitoring
- **Event Venues**: Crowd flow management at concerts and festivals
- **Airports**: Queue management and security monitoring
- **Shopping Malls**: Peak hour traffic analysis
- **Emergency Management**: Evacuation monitoring and assistance

## Performance Optimization

For improved performance on resource-constrained devices:

- Use YOLOv8n (nano) variant for maximum speed
- Process alternate frames (`skip_frames=2`)
- Reduce input resolution
- Enable GPU acceleration if available

## Technical Stack

| Component | Technology |
|-----------|------------|
| Object Detection | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV 4.8+ |
| Web Framework | Streamlit |
| Language | Python 3.10+ |
| Deep Learning | PyTorch |

## System Requirements

### Minimum
- CPU: Dual-core 2.0 GHz
- RAM: 4GB
- Storage: 500MB
- Camera: 720p webcam

### Recommended
- CPU: Quad-core 2.5 GHz or GPU
- RAM: 8GB
- Storage: 1GB
- Camera: 1080p IP camera

## Troubleshooting

### Camera Access Issues
```bash
# Test different camera indices
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Module Import Errors
Ensure virtual environment is activated (you should see `(venv)` in terminal):
```bash
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Port Conflicts (Streamlit)
```bash
streamlit run app.py --server.port 8502
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

⚠️ Known Limitations

1. Accuracy may reduce in extremely dense crowds or poor lighting
2. Performance depends on hardware + camera quality
3. No facial recognition (privacy-safe but limits identification)

## Future Enhancements

- [ ] Multi-camera synchronization
- [ ] Heatmap visualization
- [ ] SMS/Email alert integration
- [ ] Cloud deployment support
- [ ] Mobile app integration
- [ ] Historical data analytics
- [ ] AI-powered crowd flow prediction

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for state-of-the-art object detection
- [Streamlit](https://streamlit.io/) for rapid dashboard development
- [OpenCV](https://opencv.org/) for computer vision capabilities

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub or reach out to the development team.

---

**Built with ❤️ for public safety innovation**
