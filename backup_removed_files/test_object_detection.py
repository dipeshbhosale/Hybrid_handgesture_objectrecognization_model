#!/usr/bin/env python3
"""
Test Object Detection Integration
Run this script to test the new object detection features
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main module
try:
    from main import (
        object_detection_only, 
        combined_gesture_and_object_detection,
        run_with_object_detection_options
    )
    print("✅ Successfully imported object detection functions")
except ImportError as e:
    print(f"❌ Error importing functions: {e}")
    exit(1)

def main():
    print("🚀 Object Detection Test Menu")
    print("="*40)
    
    choice = input("""
Choose test option:
1. Test Object Detection Only
2. Test Combined Gesture + Object Detection  
3. Show Extended Main Menu
4. Check Requirements

Enter (1-4): """)
    
    if choice == "1":
        print("\n🎯 Testing Object Detection Only...")
        print("Note: This will download YOLOv8 model on first run (~6MB)")
        input("Press Enter to continue or Ctrl+C to cancel...")
        object_detection_only()
        
    elif choice == "2":
        print("\n🤖 Testing Combined AI Vision...")
        print("Requirements:")
        print("- gesture_model.pkl must exist (train gesture model first)")
        print("- YOLOv8 will download on first run (~6MB)")
        
        if not os.path.exists("gesture_model.pkl"):
            print("❌ gesture_model.pkl not found!")
            print("Run main.py option 3 to train gesture model first")
            return
            
        input("Press Enter to continue or Ctrl+C to cancel...")
        combined_gesture_and_object_detection()
        
    elif choice == "3":
        print("\n📋 Extended Main Menu:")
        run_with_object_detection_options()
        
    elif choice == "4":
        print("\n🔍 Checking Requirements...")
        check_requirements()
        
    else:
        print("Invalid choice!")

def check_requirements():
    """Check if required packages are installed"""
    print("Checking object detection requirements...")
    
    # Check ultralytics
    try:
        import ultralytics
        print("✅ ultralytics: Installed")
    except ImportError:
        print("❌ ultralytics: Missing")
        print("   Install with: pip install ultralytics")
    
    # Check torch
    try:
        import torch
        print("✅ torch: Installed")
        if torch.cuda.is_available():
            print("   🚀 CUDA available for GPU acceleration")
        else:
            print("   💻 Using CPU (still works fine)")
    except ImportError:
        print("❌ torch: Missing")
        print("   Install with: pip install torch")
    
    # Check existing files
    print("\nChecking existing files...")
    
    if os.path.exists("gesture_data.csv"):
        print("✅ gesture_data.csv: Found")
    else:
        print("❌ gesture_data.csv: Missing (collect gesture data first)")
    
    if os.path.exists("gesture_model.pkl"):
        print("✅ gesture_model.pkl: Found")
    else:
        print("❌ gesture_model.pkl: Missing (train model first)")
    
    print("\n📋 Installation Commands:")
    print("pip install ultralytics torch")
    print("\n🎯 To use combined features:")
    print("1. Collect gesture data (main.py option 2)")
    print("2. Train gesture model (main.py option 3)")
    print("3. Run combined detection (this script option 2)")

if __name__ == "__main__":
    main()
