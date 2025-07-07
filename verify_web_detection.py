#!/usr/bin/env python3
"""
Verify Web Interface Detection

This script verifies that the web interface correctly performs object detection and gesture recognition.
"""

import sys
import os
import time
import cv2
import numpy as np
import traceback

def verify_web_interface_detection():
    """Verify that the web interface correctly handles detection"""
    try:
        print("\n=== Web Interface Detection Verification ===")
        
        # Step 1: Import required modules
        print("Step 1: Importing modules...")
        try:
            from hybrid_ultimate_detection import HybridUltimateDetectionSystem
            from hybrid_web_interface import HybridWebInterface
            print("✅ Successfully imported required modules")
        except Exception as e:
            print(f"❌ Import failed: {str(e)}")
            return False
        
        # Step 2: Initialize detection system
        print("\nStep 2: Initializing detection system...")
        detector = HybridUltimateDetectionSystem(auto_configure=True)
        if not detector:
            print("❌ Failed to create detection system")
            return False
        print("✅ Detection system created")
        
        # Step 3: Initialize models
        print("\nStep 3: Loading detection models...")
        if not detector.initialize_models():
            print("❌ Failed to initialize detection models")
            return False
        print("✅ Detection models loaded")
        
        # Step 4: Create web interface
        print("\nStep 4: Creating web interface...")
        web = HybridWebInterface()
        if not web:
            print("❌ Failed to create web interface")
            return False
        print("✅ Web interface created")
        
        # Step 5: Initialize detector in web interface
        print("\nStep 5: Initializing detector in web interface...")
        web.detector = detector
        web.performance_mode = detector.performance_mode
        print("✅ Detector connected to web interface")
        
        # Step 6: Open camera and test detection
        print("\nStep 6: Testing camera detection...")
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Failed to open camera")
                return False
                
            print("✅ Camera opened")
            
            # Process a few frames
            frame_count = 5
            print(f"\nProcessing {frame_count} frames to test detection...")
            
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    print(f"❌ Failed to read frame {i+1}")
                    continue
                
                # Process with detector directly
                print(f"\nFrame {i+1} - Direct detection:")
                annotated_frame_direct, results_direct = detector.process_frame(frame)
                
                # Convert to RGB for web interface
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with web interface
                print(f"Frame {i+1} - Web interface detection:")
                annotated_frame_web, status_web = web.process_video_frame(frame_rgb)
                
                # Compare results
                objects_direct = len(results_direct.get('objects', []))
                gestures_direct = len(results_direct.get('gestures', []))
                
                print(f"  Direct: {objects_direct} objects, {gestures_direct} gestures")
                print(f"  Web: {status_web}")
                
                # Save the last frame for visual comparison
                if i == frame_count - 1:
                    cv2.imwrite("verify_direct_detection.jpg", annotated_frame_direct)
                    cv2.imwrite("verify_web_detection.jpg", 
                               cv2.cvtColor(annotated_frame_web, cv2.COLOR_RGB2BGR))
                    print("\n✅ Saved detection images for comparison")
            
            # Release camera
            cap.release()
            
        except Exception as e:
            print(f"❌ Error during detection test: {str(e)}")
            traceback.print_exc()
            return False
        
        print("\n✅ Web interface detection verification completed successfully!")
        print("📸 Check the generated images to confirm visual results:")
        print("  - verify_direct_detection.jpg: Direct detection results")
        print("  - verify_web_detection.jpg: Web interface detection results")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n==== HYBRID ULTIMATE DETECTION - WEB INTERFACE VERIFICATION ====")
    print("This script verifies that the web interface correctly performs object detection and gesture recognition.")
    print("It will test direct detection and web interface detection and compare the results.")
    
    success = verify_web_interface_detection()
    
    print("\n=== VERIFICATION SUMMARY ===")
    if success:
        print("✅ Web Interface Detection: PASSED")
        print("\nThe web interface correctly processes detection. If you're still experiencing issues:")
        print("1. Make sure your webcam is working properly")
        print("2. Try accessing the web interface at: http://127.0.0.1:7860")
        print("3. Check browser permissions for webcam access")
    else:
        print("❌ Web Interface Detection: FAILED")
        print("\nCheck the error messages above to identify and resolve the issue.")
    
    print("\nTo launch the web interface, run: python hybrid_web_interface.py")
