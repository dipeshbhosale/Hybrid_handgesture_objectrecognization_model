#!/usr/bin/env python3
"""
Quick Launcher for Combined AI Vision
Choose how you want to run the combined gesture + object detection
"""

def main():
    print("🚀 Combined AI Vision Launcher")
    print("=" * 40)
    
    choice = input("""
Choose interface:
1. 🌐 Web Interface (Gradio) - Browser-based
2. 🖥️  Desktop Interface (OpenCV) - Native app  
3. 📋 Show Requirements
4. 🔧 Test Setup

Enter (1-4): """)
    
    if choice == "1":
        print("\n🌐 Starting Web Interface...")
        print("📱 This will open in your browser at http://localhost:7860")
        print("⚡ Features: Gesture + Object detection in web browser")
        
        import subprocess
        import sys
        subprocess.run([sys.executable, "combined_gradio.py"])
        
    elif choice == "2":
        print("\n🖥️ Starting Desktop Interface...")
        print("🎮 Controls: 'g' toggle gestures, 'o' toggle objects, 'q' quit")
        
        try:
            from main import combined_gesture_and_object_detection
            combined_gesture_and_object_detection()
        except ImportError:
            print("❌ Error importing from main.py")
            print("💡 Try running: python main.py and choose option 8")
            
    elif choice == "3":
        print("\n📋 Requirements Check:")
        
        requirements = [
            ("gesture_model.pkl", "Gesture recognition model"),
            ("main.py", "Main application script"),
            ("combined_gradio.py", "Web interface script")
        ]
        
        import os
        for file, desc in requirements:
            if os.path.exists(file):
                print(f"✅ {file} - {desc}")
            else:
                print(f"❌ {file} - {desc} (MISSING)")
        
        print("\n📦 Required packages:")
        packages = ["opencv-python", "mediapipe", "gradio", "ultralytics", "torch"]
        for pkg in packages:
            try:
                __import__(pkg.replace('-', '_'))
                print(f"✅ {pkg}")
            except ImportError:
                print(f"❌ {pkg} (run: pip install {pkg})")
                
    elif choice == "4":
        print("\n🔧 Testing Setup...")
        
        # Test camera
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("✅ Camera working")
                cap.release()
            else:
                print("❌ Camera not accessible")
        except Exception as e:
            print(f"❌ Camera test failed: {e}")
        
        # Test models
        import os
        if os.path.exists("gesture_model.pkl"):
            print("✅ Gesture model found")
        else:
            print("❌ Gesture model missing (train with main.py option 3)")
            
        try:
            from ultralytics import YOLO
            print("✅ YOLO available")
        except ImportError:
            print("❌ YOLO not installed (run: pip install ultralytics)")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
