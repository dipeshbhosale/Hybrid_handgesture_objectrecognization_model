#!/usr/bin/env python3
"""
Quick test script to verify the optimized advanced detection system FPS
Tests both single model and ensemble modes
"""

import time
import sys
import os
sys.path.append('.')

def test_optimized_fps_single_model():
    """Test the optimized advanced detection system performance with single model"""
    try:
        from main import ADVANCED_DETECTION_AVAILABLE
        
        if not ADVANCED_DETECTION_AVAILABLE:
            print("❌ Advanced detection system not available")
            return False
        
        from advanced_detection_system import get_advanced_detector
        import numpy as np
        
        print("🧪 Testing OPTIMIZED Advanced Detection System (Single Model Mode)")
        print("="*70)
        
        # Initialize detector
        detector = get_advanced_detector()
        
        # Set optimized parameters (same as in main.py)
        detector.confidence_threshold = 0.4  # Higher confidence for better performance
        detector.nms_threshold = 0.5
        
        if not detector.models:
            print("❌ No models loaded")
            return False
        
        print(f"✅ Loaded {len(detector.models)} models")
        print(f"🔧 Confidence threshold: {detector.confidence_threshold}")
        print(f"🔧 NMS threshold: {detector.nms_threshold}")
        
        # Test with optimized settings
        width, height = 640, 480
        print(f"\n📊 Testing Optimized Settings ({width}x{height})")
        
        # Create test frames (smaller for speed)
        num_frames = 10
        test_frames = []
        for i in range(num_frames):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            # Add some simple shapes to potentially detect
            if i % 3 == 0:
                frame[100:200, 100:200] = [255, 255, 255]  # White square
            test_frames.append(frame)
        
        # Test Single Model Mode (should be faster)
        print("\n🚀 Testing SINGLE MODEL Mode (Optimized for 15 FPS):")
        start_time = time.time()
        total_detections = 0
        frame_times = []
        
        for i, frame in enumerate(test_frames):
            frame_start = time.time()
            # Use single model mode (ensemble=False)
            processed_frame, detections = detector.process_frame_with_tracking(
                frame, use_ensemble=False, draw_stats=False
            )
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            total_detections += len(detections)
            
            fps = 1.0 / frame_time if frame_time > 0 else 0
            print(f"   Frame {i+1:2d}: {frame_time:.3f}s ({fps:5.1f} FPS) - {len(detections)} objects")
        
        total_time = time.time() - start_time
        average_fps = num_frames / total_time
        avg_frame_time = sum(frame_times) / len(frame_times)
        avg_fps_per_frame = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        print(f"\n📈 SINGLE MODEL RESULTS:")
        print(f"   Average FPS (total): {average_fps:.1f}")
        print(f"   Average FPS (per frame): {avg_fps_per_frame:.1f}")
        print(f"   🎯 Total detections: {total_detections}")
        
        if avg_fps_per_frame >= 15:
            print(f"   ✅ TARGET 15 FPS ACHIEVED! 🎉")
        elif avg_fps_per_frame >= 10:
            print(f"   ⚡ Good performance (>10 FPS)")
        elif avg_fps_per_frame >= 5:
            print(f"   💫 Moderate performance (>5 FPS)")
        else:
            print(f"   ⚠️ Below target FPS")
        
        # Test Ensemble Mode for comparison
        print("\n🎭 Testing ENSEMBLE Mode (for comparison):")
        start_time = time.time()
        total_detections_ensemble = 0
        frame_times_ensemble = []
        
        # Test with fewer frames for ensemble (it's slower)
        test_frames_small = test_frames[:5]
        
        for i, frame in enumerate(test_frames_small):
            frame_start = time.time()
            # Use ensemble mode
            processed_frame, detections = detector.process_frame_with_tracking(
                frame, use_ensemble=True, draw_stats=False
            )
            frame_time = time.time() - frame_start
            frame_times_ensemble.append(frame_time)
            total_detections_ensemble += len(detections)
            
            fps = 1.0 / frame_time if frame_time > 0 else 0
            print(f"   Frame {i+1:2d}: {frame_time:.3f}s ({fps:5.1f} FPS) - {len(detections)} objects")
        
        avg_frame_time_ensemble = sum(frame_times_ensemble) / len(frame_times_ensemble)
        avg_fps_ensemble = 1.0 / avg_frame_time_ensemble if avg_frame_time_ensemble > 0 else 0
        
        print(f"\n📈 ENSEMBLE MODE RESULTS:")
        print(f"   Average FPS (per frame): {avg_fps_ensemble:.1f}")
        print(f"   🎯 Total detections: {total_detections_ensemble}")
        
        # Summary
        print(f"\n🏆 PERFORMANCE COMPARISON:")
        print(f"   Single Model: {avg_fps_per_frame:.1f} FPS")
        print(f"   Ensemble Mode: {avg_fps_ensemble:.1f} FPS")
        improvement = avg_fps_per_frame / avg_fps_ensemble if avg_fps_ensemble > 0 else 1
        print(f"   Speed improvement: {improvement:.1f}x faster with single model")
        
        print("\n💡 OPTIMIZATION TIPS:")
        print("   ✅ Use single model mode for 15 FPS target")
        print("   ✅ Use 640x480 resolution")
        print("   ✅ Higher confidence threshold (0.4-0.6)")
        print("   ✅ Process every 3rd frame in live system")
        print("   💫 Enable ensemble only when accuracy is more important than speed")
        
        return avg_fps_per_frame >= 10  # Consider success if we get >10 FPS
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimized_fps_single_model()
    if success:
        print("\n🎉 Optimization test completed successfully!")
    else:
        print("\n😞 Optimization test encountered issues")
