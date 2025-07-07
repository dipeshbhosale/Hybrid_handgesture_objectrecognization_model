#!/usr/bin/env python3
"""
Quick Test - Object Detection Integration
Simple test to verify the object detection works
"""
import sys
import os

def quick_test():
    """Quick test of object detection functionality"""
    print("🚀 Quick Object Detection Test")
    print("=" * 40)
    
    try:
        # Test basic imports
        print("📦 Testing imports...")
        import cv2
        from ultralytics import YOLO
        import torch
        print("✅ All imports successful!")
        
        # Test camera
        print("\n📹 Testing camera...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera working!")
                cap.release()
            else:
                print("❌ Camera issues")
                cap.release()
                return False
        else:
            print("❌ Cannot open camera")
            return False
        
        # Load YOLO with compatibility fix
        print("\n🤖 Testing YOLO (this may take a moment)...")
        
        # Temporary fix for PyTorch compatibility
        import warnings
        warnings.filterwarnings('ignore')
        
        # Try loading with weights_only=False
        old_weights_only = None
        try:
            import torch
            old_load = torch.load
            def safe_load(*args, **kwargs):
                kwargs.pop('weights_only', None)  # Remove if present
                return old_load(*args, **kwargs, weights_only=False)
            torch.load = safe_load
            
            model = YOLO('yolov8n.pt')
            
            # Restore original function
            torch.load = old_load
            
            print("✅ YOLO model loaded successfully!")
            print(f"   Classes available: {len(model.names)}")
            
            return True
            
        except Exception as e:
            print(f"❌ YOLO loading failed: {str(e)[:100]}...")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_available_options():
    """Show what the user can do next"""
    print("\n🎯 Available Options:")
    print("=" * 40)
    
    print("1. 🤟 Gesture Recognition Only:")
    print("   python main.py  # Choose option 6")
    
    print("\n2. 🎯 Object Detection Only:")
    print("   python test_object_detection.py  # Choose option 1")
    
    print("\n3. 🤖 Combined AI Vision:")
    print("   python test_object_detection.py  # Choose option 2")
    print("   (Requires gesture model to be trained first)")
    
    print("\n4. 🌐 Web Interface:")
    print("   python main.py  # Choose option 4 or 9")
    
    print("\n5. 🎯 Advanced Detection (15 FPS):")
    print("   python main.py  # Choose option 11")
    
    print("\n6. 🏆 ULTRA-ACCURATE Detection (MAXIMUM PRECISION):")
    print("   python main.py  # Choose option 14")
    print("   ⚠️  Warning: Slower but highest accuracy possible!")
    
    print("\n📋 Setup Steps:")
    if not os.path.exists("gesture_model.pkl"):
        print("   ⚠️  Missing: gesture_model.pkl")
        print("   • Run: python main.py (option 2 to collect data)")
        print("   • Then: python main.py (option 3 to train model)")
    else:
        print("   ✅ gesture_model.pkl found - ready for combined detection!")
    
    print("\n💡 Pro Tips:")
    print("   • Use good lighting for better detection")
    print("   • Object detection works without gesture model")
    print("   • Combined mode needs both models")
    print("   • Ultra-accurate mode uses ALL YOLO models for best results")
    print("   • Test ultra-accurate mode: python test_ultra_accurate_detection.py")

if __name__ == "__main__":
    if quick_test():
        print("\n🎉 Setup Complete! Object detection is ready to use.")
        show_available_options()
    else:
        print("\n❌ Setup incomplete. Please check the errors above.")
        print("\n🔧 Troubleshooting Tips:")
        print("   • Make sure camera is not used by other apps")
        print("   • Try updating PyTorch: pip install --upgrade torch")
        print("   • Check internet connection for model download")
