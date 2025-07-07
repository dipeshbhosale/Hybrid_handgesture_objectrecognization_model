#!/usr/bin/env python3
"""
15 FPS Performance Test for Advanced Detection System
===================================================
Tests the optimized advanced detection system to verify 15 FPS target is achieved.
"""

import cv2
import time
import numpy as np
from collections import deque
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_detection_system import get_advanced_detector
    ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"❌ Advanced detection system not available: {e}")
    ADVANCED_AVAILABLE = False

def test_15fps_performance():
    """Test the advanced detection system for 15 FPS performance"""
    
    if not ADVANCED_AVAILABLE:
        print("❌ Cannot run test - advanced detection system not available")
        return False
    
    print("\n" + "="*60)
    print("🚀 15 FPS PERFORMANCE TEST - Advanced Detection System")
    print("="*60)
    print("🎯 Target: Stable 15 FPS with real-time object detection")
    print("📊 Test duration: 30 seconds")
    print("🔄 Testing both single model and ensemble modes")
    print("="*60)
    
    # Initialize detector
    detector = get_advanced_detector()
    if not detector or not detector.models:
        print("❌ Failed to initialize detector")
        return False
    
    print(f"✅ Detector initialized with {len(detector.models)} models")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return False
    
    # Set optimal camera settings for 15 FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Test parameters
    test_duration = 30  # seconds
    fps_history = deque(maxlen=100)
    detection_times = deque(maxlen=100)
    modes_to_test = [
        ("Single Model (Fast)", False),
        ("Ensemble Mode", True)
    ]
    
    try:
        for mode_name, ensemble_mode in modes_to_test:
            print(f"\n🧪 Testing {mode_name}...")
            
            frame_count = 0
            start_time = time.time()
            mode_fps = deque(maxlen=50)
            mode_detection_times = deque(maxlen=50)
            
            # Set optimized confidence for the mode
            if ensemble_mode:
                detector.confidence_threshold = 0.5  # Higher for ensemble
            else:
                detector.confidence_threshold = 0.4  # Balanced for single
            
            while time.time() - start_time < test_duration / 2:  # 15 seconds per mode
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_start = time.time()
                
                # Process frame
                detection_start = time.time()
                processed_frame, detections = detector.process_frame_with_tracking(
                    frame, use_ensemble=ensemble_mode, draw_stats=True
                )
                detection_time = time.time() - detection_start
                
                frame_time = time.time() - frame_start
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                
                # Record metrics
                mode_fps.append(current_fps)
                mode_detection_times.append(detection_time)
                frame_count += 1
                
                # Display frame with test info
                test_info = [
                    f"Mode: {mode_name}",
                    f"FPS: {current_fps:.1f}",
                    f"Detection Time: {detection_time*1000:.1f}ms",
                    f"Objects: {len(detections)}",
                    f"Frame: {frame_count}"
                ]
                
                y_offset = 30
                for info in test_info:
                    cv2.putText(processed_frame, info, (400, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    y_offset += 25
                
                cv2.imshow(f'15 FPS Test - {mode_name}', processed_frame)
                
                # Break on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Calculate results for this mode
            avg_fps = np.mean(mode_fps) if mode_fps else 0
            avg_detection_time = np.mean(mode_detection_times) * 1000 if mode_detection_times else 0
            min_fps = np.min(mode_fps) if mode_fps else 0
            max_fps = np.max(mode_fps) if mode_fps else 0
            
            # Performance analysis
            fps_above_15 = sum(1 for fps in mode_fps if fps >= 15)
            fps_percentage = (fps_above_15 / len(mode_fps) * 100) if mode_fps else 0
            
            print(f"\n📊 {mode_name} Results:")
            print(f"   📈 Average FPS: {avg_fps:.2f}")
            print(f"   🎯 Min FPS: {min_fps:.2f}")
            print(f"   🚀 Max FPS: {max_fps:.2f}")
            print(f"   ⏱️  Avg Detection Time: {avg_detection_time:.1f}ms")
            print(f"   ✅ Frames ≥15 FPS: {fps_percentage:.1f}%")
            
            # Performance verdict
            if avg_fps >= 15:
                print(f"   🎉 PASSED: Average FPS meets 15 FPS target!")
            else:
                print(f"   ⚠️  NEEDS IMPROVEMENT: Average FPS below 15")
            
            if fps_percentage >= 80:
                print(f"   🎯 EXCELLENT: 80%+ frames at 15+ FPS")
            elif fps_percentage >= 60:
                print(f"   👍 GOOD: 60%+ frames at 15+ FPS")
            else:
                print(f"   📈 NEEDS OPTIMIZATION: Less than 60% frames at 15+ FPS")
    
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Final recommendations
    print(f"\n" + "="*60)
    print("🎯 15 FPS PERFORMANCE TEST SUMMARY")
    print("="*60)
    print("💡 Recommendations for 15 FPS:")
    print("   • Use Single Model mode for consistent 15+ FPS")
    print("   • Set confidence threshold to 0.5+ for better performance")
    print("   • Ensure camera resolution is 640x480 or lower")
    print("   • Use fast mode ('F' key) during detection")
    print("   • Consider ensemble mode only when accuracy is critical")
    print("="*60)
    
    return True

def quick_fps_benchmark():
    """Quick benchmark without camera for testing detection speed"""
    
    if not ADVANCED_AVAILABLE:
        print("❌ Cannot run benchmark - advanced detection system not available")
        return
    
    print("\n🏃‍♂️ QUICK FPS BENCHMARK (No Camera)")
    print("="*50)
    
    detector = get_advanced_detector()
    if not detector or not detector.models:
        print("❌ Failed to initialize detector")
        return
    
    # Create test frames
    test_frames = []
    for i in range(10):
        # Random frame with some patterns
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add some shapes that might be detected
        cv2.rectangle(frame, (100+i*30, 100), (200+i*30, 200), (255, 255, 255), -1)
        cv2.circle(frame, (400, 300), 50+i*5, (128, 128, 128), -1)
        test_frames.append(frame)
    
    # Test single model
    print("🧪 Testing Single Model Mode...")
    times = []
    for frame in test_frames:
        start = time.time()
        detections = detector.ensemble_detect(frame, use_ensemble=False)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    avg_fps = 1.0 / avg_time
    print(f"   ⏱️  Average time: {avg_time*1000:.1f}ms")
    print(f"   📈 Theoretical FPS: {avg_fps:.1f}")
    
    # Test ensemble
    if len(detector.models) > 1:
        print("🧪 Testing Ensemble Mode...")
        times = []
        for frame in test_frames:
            start = time.time()
            detections = detector.ensemble_detect(frame, use_ensemble=True)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        avg_fps = 1.0 / avg_time
        print(f"   ⏱️  Average time: {avg_time*1000:.1f}ms")
        print(f"   📈 Theoretical FPS: {avg_fps:.1f}")
    
    print("="*50)

if __name__ == "__main__":
    choice = input("""
🚀 Choose test type:
1. Full 15 FPS Performance Test (with camera)
2. Quick Benchmark (no camera required)

Enter (1-2): """).strip()
    
    if choice == "1":
        test_15fps_performance()
    elif choice == "2":
        quick_fps_benchmark()
    else:
        print("🚀 Running full performance test...")
        test_15fps_performance()
