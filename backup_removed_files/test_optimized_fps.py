#!/usr/bin/env python3
"""
Quick test script to verify the optimized advanced detection system FPS
"""

import time
import sys
import os
sys.path.append('.')

def test_optimized_fps():
    """Test the optimized advanced detection system performance"""
    try:
        from main import ADVANCED_DETECTION_AVAILABLE
        
        if not ADVANCED_DETECTION_AVAILABLE:
            print("❌ Advanced detection system not available")
            return False
        
        from advanced_detection_system import get_advanced_detector
        import numpy as np
        
        print("🧪 Testing Optimized Advanced Detection System Performance")
        print("="*60)
        
        # Initialize detector
        detector = get_advanced_detector()
        
        # Set optimized parameters (same as in main.py)
        detector.confidence_threshold = 0.35
        detector.nms_threshold = 0.5
        
        if not detector.models:
            print("❌ No models loaded")
            return False
        
        print(f"✅ Loaded {len(detector.models)} models")
        print(f"🔧 Confidence threshold: {detector.confidence_threshold}")
        print(f"🔧 NMS threshold: {detector.nms_threshold}")
        
        # Test with multiple frame sizes to simulate optimization
        test_configs = [
            {"size": (640, 480), "name": "Optimized Resolution"},
            {"size": (1280, 720), "name": "High Resolution (comparison)"}
        ]
        
        for config in test_configs:
            width, height = config["size"]
            print(f"\n📊 Testing {config['name']} ({width}x{height})")
            
            # Create test frames
            test_frames = []
            for i in range(5):
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                test_frames.append(frame)
            
            # Measure performance
            start_time = time.time()
            total_detections = 0
            
            for i, frame in enumerate(test_frames):
                frame_start = time.time()
                processed_frame, detections = detector.process_frame_with_tracking(
                    frame, use_ensemble=True, draw_stats=False
                )
                frame_time = time.time() - frame_start
                total_detections += len(detections)
                
                fps = 1.0 / frame_time if frame_time > 0 else 0
                print(f"   Frame {i+1}: {frame_time:.3f}s ({fps:.1f} FPS) - {len(detections)} objects")
            
            total_time = time.time() - start_time
            average_fps = len(test_frames) / total_time
            
            print(f"   📈 Average FPS: {average_fps:.1f}")
            print(f"   🎯 Total detections: {total_detections}")
            
            if average_fps >= 15:
                print(f"   ✅ Target 15 FPS achieved!")
            elif average_fps >= 10:
                print(f"   ⚡ Good performance (>10 FPS)")
            else:
                print(f"   ⚠️ Below target FPS")
        
        print("\n💡 Performance Tips:")
        print("   • Use 640x480 resolution for best performance")
        print("   • Toggle fast mode (F key) in live system")
        print("   • Disable ensemble mode for single model speed")
        print("   • Higher confidence threshold = fewer detections = better FPS")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_optimized_fps()
