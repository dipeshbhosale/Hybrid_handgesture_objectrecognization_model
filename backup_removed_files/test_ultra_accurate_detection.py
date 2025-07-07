#!/usr/bin/env python3
"""
Test Ultra-Accurate Detection System
==================================
Test and validate the ultra-accurate multi-object detection system
"""

import cv2
import time
import os
import sys
from datetime import datetime

def test_ultra_accurate_system():
    """Test the ultra-accurate detection system"""
    print("🎯 Testing Ultra-Accurate Detection System")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from ultra_accurate_detection import UltraAccurateDetectionSystem, DETECTION_AVAILABLE
        
        if not DETECTION_AVAILABLE:
            print("❌ Detection dependencies not available")
            return False
        
        print("✅ All imports successful!")
        
        # Initialize system
        print("\n🔄 Initializing ultra-accurate detection system...")
        detector = UltraAccurateDetectionSystem()
        
        # Load models
        print("📥 Loading all YOLO models for maximum accuracy...")
        success = detector.load_all_models()
        
        if not success or not detector.models:
            print("❌ Failed to load models")
            print("💡 Try running: python setup_enhanced_detection.py")
            return False
        
        print(f"✅ Successfully loaded {len(detector.models)} models")
        for model_name in detector.models.keys():
            print(f"   • {model_name}")
        
        # Test camera
        print("\n📹 Testing camera with ultra-accurate detection...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return False
        
        # Set high resolution for maximum accuracy
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✅ Camera initialized at high resolution")
        
        # Test detection on a few frames
        print("\n🎯 Testing ultra-accurate detection...")
        test_frames = 3
        total_detections = 0
        
        for i in range(test_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Failed to capture frame {i+1}")
                continue
            
            print(f"🔍 Processing frame {i+1}/{test_frames}...")
            start_time = time.time()
            
            # Run ultra-accurate detection
            detections, accuracy_stats = detector.detect_ultra_accurate(frame)
            
            process_time = time.time() - start_time
            total_detections += len(detections)
            
            print(f"   ✅ Found {len(detections)} objects in {process_time:.2f}s")
            
            # Test drawing
            annotated_frame = detector.draw_ultra_accurate_detections_with_stats(
                frame, detections, accuracy_stats
            )
            
            # Save test frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ultra_accurate_test_{i+1}_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"   💾 Saved test result: {filename}")
        
        cap.release()
        
        # Print test summary
        print(f"\n📊 Test Summary:")
        print(f"   • Models loaded: {len(detector.models)}")
        print(f"   • Frames processed: {test_frames}")
        print(f"   • Total detections: {total_detections}")
        print(f"   • Average detections per frame: {total_detections/test_frames:.1f}")
        
        # Test statistics
        stats = detector.get_accuracy_statistics()
        print(f"   • Unique object types detected: {stats.get('unique_classes_detected', 0)}")
        
        print("\n✅ Ultra-accurate detection system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_ensemble():
    """Test ensemble functionality specifically"""
    print("\n🔬 Testing Model Ensemble Functionality")
    print("=" * 40)
    
    try:
        from ultra_accurate_detection import UltraAccurateDetectionSystem
        
        detector = UltraAccurateDetectionSystem()
        detector.load_all_models()
        
        if len(detector.models) < 2:
            print("⚠️ Need at least 2 models for ensemble testing")
            return True
        
        print(f"🎯 Testing ensemble with {len(detector.models)} models:")
        for name, info in detector.models.items():
            print(f"   • {name} (priority: {info['priority']}, weight: {info['weight']})")
        
        # Create test image
        test_frame = cv2.imread("advanced_detection_20250703_124809.jpg")
        if test_frame is None:
            # Create synthetic test image if no image available
            test_frame = cv2.rectangle(
                np.zeros((480, 640, 3), dtype=np.uint8),
                (100, 100), (200, 200), (255, 255, 255), -1
            )
            print("📝 Using synthetic test image")
        else:
            print("📸 Using existing test image")
        
        # Test ensemble detection
        print("🔍 Running ensemble detection...")
        start_time = time.time()
        
        detections, stats = detector.detect_ultra_accurate(test_frame)
        
        ensemble_time = time.time() - start_time
        
        print(f"✅ Ensemble detection completed in {ensemble_time:.2f}s")
        print(f"🎯 Found {len(detections)} objects with ensemble method")
        
        if detections:
            print("🏆 Top detections:")
            for i, detection in enumerate(detections[:5]):
                print(f"   {i+1}. {detection['class_name']}: {detection['confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Ultra-Accurate Detection System Test Suite")
    print("=" * 60)
    
    # Run basic system test
    basic_test = test_ultra_accurate_system()
    
    if basic_test:
        # Run ensemble test
        ensemble_test = test_model_ensemble()
        
        if ensemble_test:
            print("\n🎉 All tests passed! Ultra-accurate detection system is ready.")
            print("\n🎯 To use the system:")
            print("   python main.py")
            print("   Then select option 14: Ultra-Accurate Multi-Object Detection")
        else:
            print("\n⚠️ Basic test passed but ensemble test failed")
    else:
        print("\n❌ Tests failed. Please check the setup:")
        print("   1. Ensure all dependencies are installed")
        print("   2. Run: python setup_enhanced_detection.py")
        print("   3. Check camera connection")
        print("   4. Verify YOLO model files are present")
