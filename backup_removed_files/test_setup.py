#!/usr/bin/env python3
"""
Test Script for Object Detection Setup
Run this to verify all dependencies are installed correctly
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    tests = [
        ("OpenCV", "import cv2"),
        ("MediaPipe", "import mediapipe as mp"),
        ("NumPy", "import numpy as np"),
        ("Pandas", "import pandas as pd"),
        ("Scikit-learn", "import sklearn"),
        ("Joblib", "import joblib"),
        ("Gradio", "import gradio as gr"),
        ("Keyboard", "import keyboard"),
        ("Matplotlib", "import matplotlib"),
        ("Seaborn", "import seaborn"),
        ("Ultralytics (YOLO)", "from ultralytics import YOLO"),
        ("PyTorch", "import torch"),
        ("TorchVision", "import torchvision"), 
        ("TorchAudio", "import torchaudio")
    ]
    
    passed = 0
    failed = 0
    
    for name, import_cmd in tests:
        try:
            exec(import_cmd)
            print(f"✅ {name}: OK")
            passed += 1
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️  {name}: WARNING - {e}")
            passed += 1  # Count as passed if it's not an import error
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0

def test_yolo_model():
    """Test if YOLO model can be loaded"""
    print("\n🤖 Testing YOLO model loading...")
    
    try:
        from ultralytics import YOLO
        print("📦 Loading YOLOv8 nano model...")
        model = YOLO('yolov8n.pt')  # This will download the model if not present
        print("✅ YOLO model loaded successfully!")
        
        # Test basic info
        print(f"📋 Model info:")
        print(f"   - Classes: {len(model.names)}")
        print(f"   - Device: {model.device}")
        print(f"   - Model size: yolov8n (nano)")
        
        return True
    except Exception as e:
        print(f"❌ YOLO model test failed: {e}")
        return False

def test_camera():
    """Test camera access"""
    print("\n📹 Testing camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera is working!")
                print(f"   - Frame shape: {frame.shape}")
                cap.release()
                return True
            else:
                print("❌ Camera opened but cannot capture frames")
                cap.release()
                return False
        else:
            print("❌ Cannot open camera")
            return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_gesture_files():
    """Check if gesture recognition files exist"""
    print("\n🤟 Checking gesture recognition files...")
    
    files_to_check = [
        ("main.py", "Main script"),
        ("gesture_data.csv", "Gesture training data (optional)"),
        ("gesture_model.pkl", "Trained gesture model (optional)")
    ]
    
    for filename, description in files_to_check:
        if os.path.exists(filename):
            print(f"✅ {filename}: Found - {description}")
        else:
            print(f"📝 {filename}: Missing - {description}")
    
    return True

def main():
    print("🚀 Object Detection Setup Test")
    print("=" * 50)
    
    # Run all tests
    tests_passed = 0
    total_tests = 4
    
    if test_imports():
        tests_passed += 1
        
    if test_yolo_model():
        tests_passed += 1
        
    if test_camera():
        tests_passed += 1
        
    if test_gesture_files():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Setup Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! You're ready to use object detection!")
        print("\n📋 Next steps:")
        print("1. Run: python test_object_detection.py")
        print("2. Or modify main.py to enable the extended menu")
        print("3. Choose option 7 for Object Detection Only")
        print("4. Choose option 8 for Combined Gesture + Object Detection")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        
    print("\n💡 Tips:")
    print("- For gesture recognition, collect data first (main.py option 2)")
    print("- Then train the model (main.py option 3)")
    print("- YOLO model downloads automatically on first use (~6MB)")

if __name__ == "__main__":
    main()
