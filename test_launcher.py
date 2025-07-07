#!/usr/bin/env python3
"""
Test launcher to debug and run the hybrid detection system
"""

print("🔍 Testing environment and dependencies...")

# Test basic imports
try:
    import sys
    print(f"✅ Python version: {sys.version}")
except:
    print("❌ Python import failed")

try:
    import os
    print(f"✅ Working directory: {os.getcwd()}")
except:
    print("❌ OS import failed")

try:
    import cv2
    print(f"✅ OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    import numpy as np
    print(f"✅ NumPy version: {np.__version__}")
except Exception as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import mediapipe as mp
    print(f"✅ MediaPipe version: {mp.__version__}")
except Exception as e:
    print(f"❌ MediaPipe import failed: {e}")

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
except Exception as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLO available")
except Exception as e:
    print(f"❌ Ultralytics import failed: {e}")

print("\n🚀 Attempting to import and run hybrid detection system...")

try:
    # Import the main module
    import hybrid_ultimate_detection
    print("✅ Hybrid detection module imported successfully")
    
    # Run the main function
    print("🎬 Starting hybrid detection system...")
    hybrid_ultimate_detection.run_hybrid_ultimate_detection_system()
    
except Exception as e:
    print(f"❌ Error running hybrid detection: {e}")
    import traceback
    traceback.print_exc()
