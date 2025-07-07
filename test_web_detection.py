#!/usr/bin/env python3
"""
Test script for debugging web interface detection issues
"""

import cv2
import numpy as np
import time
import os
import sys

def test_detection_directly():
    """Test the detection system directly without the web interface"""
    try:
        from hybrid_ultimate_detection import HybridUltimateDetectionSystem
        
        print("=== Testing Direct Detection ===")
        print("Initializing detection system...")
        detector = HybridUltimateDetectionSystem(auto_configure=True)
        detector.initialize_models()
        
        print("Opening camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return False
        
        # Process 10 frames to test detection
        print("Testing with 10 frames...")
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                print(f"ERROR: Could not read frame {i}")
                continue
                
            print(f"Processing frame {i+1}...")
            start_time = time.time()
            annotated_frame, results = detector.process_frame(frame)
            process_time = time.time() - start_time
            
            # Print detection results
            objects = results.get('objects', [])
            gestures = results.get('gestures', [])
            
            print(f"Frame {i+1} results:")
            print(f"  Processing time: {process_time:.3f} seconds")
            print(f"  Objects detected: {len(objects)}")
            for obj in objects[:3]:  # Show first 3 objects
                print(f"    - {obj.get('class_name', 'unknown')}: {obj.get('confidence', 0):.2f}")
            
            print(f"  Gestures detected: {len(gestures)}")
            for gesture in gestures:
                print(f"    - {gesture.get('gesture_name', 'unknown')}: {gesture.get('confidence', 0):.2f}")
            
            # Save last frame for verification
            if i == 9:
                cv2.imwrite("test_detection_frame.jpg", annotated_frame)
                print("Saved test frame to test_detection_frame.jpg")
        
        # Release the camera
        cap.release()
        print("Test completed successfully")
        return True
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {str(e)}")
        return False

def test_web_interface():
    """Run a test of the web interface"""
    try:
        import gradio as gr
        from hybrid_web_interface import create_web_interface, HybridWebInterface
        
        print("=== Testing Web Interface ===")
        print("Creating a test instance...")
        interface = HybridWebInterface()
        
        print("Testing detector initialization...")
        result = interface.initialize_detector()
        print(f"Initialization result: {result}")
        
        print("\nWebcam test:")
        print("1. A blank image will be passed to the detector")
        print("2. The processing result will be shown")
        
        # Create a test frame
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Process the test frame
        print("Processing test frame...")
        try:
            output_frame, status = interface.process_video_frame(test_frame)
            print(f"Status: {status}")
            cv2.imwrite("test_web_output.jpg", cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
            print("Saved test output to test_web_output.jpg")
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
        
        print("\nTest completed")
        return True
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("==== HYBRID DETECTION SYSTEM WEB INTERFACE - DIAGNOSTIC TEST ====")
    print("This test checks if detection works correctly both directly and through the web interface")
    
    print("\n=== STEP 1: Direct Detection Test ===")
    direct_test = test_detection_directly()
    
    print("\n=== STEP 2: Web Interface Test ===")
    web_test = test_web_interface()
    
    print("\n=== TEST SUMMARY ===")
    print(f"Direct Detection Test: {'✅ PASSED' if direct_test else '❌ FAILED'}")
    print(f"Web Interface Test: {'✅ PASSED' if web_test else '❌ FAILED'}")
    
    if direct_test and not web_test:
        print("\n🔍 DIAGNOSIS: The detection system works correctly, but there's an issue with the web interface.")
    elif not direct_test:
        print("\n🔍 DIAGNOSIS: There's a fundamental issue with the detection system.")
    else:
        print("\n🔍 DIAGNOSIS: Both systems appear to work correctly in diagnostic tests.")
        print("If you're still having issues in the live web interface, it could be related to:")
        print("1. Webcam configuration in Gradio")
        print("2. Frame conversion between RGB and BGR formats")
        print("3. Browser permissions for webcam access")
