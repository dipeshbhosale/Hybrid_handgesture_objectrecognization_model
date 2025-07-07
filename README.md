# Hybrid Ultimate Detection System

## Overview
This is a streamlined version of the Hybrid Ultimate Detection System that combines ultra-accurate object detection with performance optimization for the ultimate AI vision experience. The system features adaptive performance modes, auto-configuration, and seamless multi-modal detection.

## Files in this Package
- **hybrid_ultimate_detection.py**: The core detection system integrating YOLOv8 models and gesture recognition
- **main.py**: Simple menu system to access different features
- **hybrid_web_interface.py**: Browser-based interface using Gradio
- **launch_hybrid.py**: Command-line launcher with configuration options
- **gesture_model.pkl**: Pre-trained gesture recognition model
- **YOLOv8 models**: Various sized models (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- **requirements.txt**: Required Python packages

## Installation
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script to access all features:
```bash
python main.py
```

### Available Options:
1. **Run Hybrid Ultimate Detection (Fullscreen Mode)**: Launches the detection system in fullscreen
2. **Launch Web Interface**: Opens a browser-based interface
3. **Run with Custom Options**: Allows setting specific configuration options
4. **Train Gesture Model**: Train or retrain the gesture recognition model
5. **Exit**: Quit the application

## Direct Access to Specific Interfaces
- Fullscreen detection: `python hybrid_ultimate_detection.py`
- Web interface: `python hybrid_web_interface.py` (accessible at http://127.0.0.1:7860 or http://127.0.0.1:7861)
- Command-line options: `python launch_hybrid.py --help`
- Web detection verification: `python verify_web_detection.py`

## Features
- Auto-optimized performance based on hardware capabilities
- Ultra-accurate object detection with YOLOv8 ensemble
- Real-time hand gesture recognition
- Adaptive performance modes (efficient, fast, balanced, ultra_accurate)
- Full screen display mode for production environments
- Browser-based interface option
- Command-line configuration options

## Controls
- **q**: Quit
- **s**: Save screenshot and detection data
- **r**: Reset statistics
- **1-4**: Switch performance mode (1=efficient, 2=fast, 3=balanced, 4=ultra_accurate)
- **c**: Cycle confidence threshold
