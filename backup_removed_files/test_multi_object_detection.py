#!/usr/bin/env python3
"""
Enhanced Multi-Object Detection Test
===================================
Test script to verify multiple objects are being detected properly with the advanced system.
"""

import cv2
import numpy as np
import time
from collections import defaultdict
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

def test_multi_object_detection():
    """Test the system's ability to detect multiple objects"""
    
    if not ADVANCED_AVAILABLE:
        print("❌ Cannot run test - advanced detection system not available")
        return False
    
    print("\n" + "="*60)
    print("🎯 ENHANCED MULTI-OBJECT DETECTION TEST")
    print("="*60)
    print("🔍 Testing ability to detect multiple objects simultaneously")
    print("📊 Will run for 30 seconds and report detailed statistics")
    print("📋 Instructions:")
    print("   • Place multiple objects in camera view")
    print("   • Try phones, bottles, books, keys, etc.")
    print("   • Move objects around to test tracking")
    print("="*60)
    
    # Initialize detector
    detector = get_advanced_detector()
    if not detector or not detector.models:
        print("❌ Failed to initialize detector")
        return False
    
    print(f"✅ Detector initialized with {len(detector.models)} models")
    
    # Set optimized settings for multi-object detection
    detector.confidence_threshold = 0.25  # Lower confidence for more detections
    print(f"🎚️ Confidence threshold set to: {detector.confidence_threshold}")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return False
    
    # Set camera to full resolution for better detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("🎥 Camera initialized at 1280x720 for optimal multi-object detection")
    
    # Test parameters
    test_duration = 30  # seconds
    start_time = time.time()
    frame_count = 0
    
    # Statistics tracking
    object_detection_history = []
    unique_objects_seen = set()
    max_objects_frame = 0
    total_detections = 0
    
    try:
        print(f"\n🧪 Starting {test_duration} second test...")
        print("💡 Try to show multiple objects to the camera!")
        
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Process frame with single model for speed but better multi-object detection
            processed_frame, detections = detector.process_frame_with_tracking(
                frame, use_ensemble=False, draw_stats=True
            )
            
            # Track statistics
            num_objects = len(detections)
            object_detection_history.append(num_objects)
            total_detections += num_objects
            max_objects_frame = max(max_objects_frame, num_objects)
            
            # Track unique object types
            for det in detections:
                unique_objects_seen.add(det['class_name'])
            
            # Display test info
            elapsed = time.time() - start_time
            remaining = test_duration - elapsed
            
            test_info = [
                f"🧪 ENHANCED MULTI-OBJECT TEST",
                f"⏱️ Time: {remaining:.0f}s remaining",
                f"🎯 Objects this frame: {num_objects}",
                f"📊 Max objects seen: {max_objects_frame}",
                f"🔍 Unique types: {len(unique_objects_seen)}",
                f"📈 Total detections: {total_detections}"
            ]
            
            y_offset = 30
            for info in test_info:
                cv2.putText(processed_frame, info, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
            
            # Add current object list
            if detections:
                cv2.putText(processed_frame, "Current objects:", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                y_offset += 25
                
                # Show up to 8 current objects (increased from 5)
                for i, det in enumerate(detections[:8]):
                    obj_text = f"• {det['class_name']} ({det['confidence']:.1f})"
                    cv2.putText(processed_frame, obj_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 20
                
                if len(detections) > 8:
                    cv2.putText(processed_frame, f"... +{len(detections)-8} more", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            cv2.imshow('Enhanced Multi-Object Detection Test', processed_frame)
            
            # Break on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Calculate detailed results
        avg_objects = np.mean(object_detection_history) if object_detection_history else 0
        frames_with_multiple = sum(1 for count in object_detection_history if count >= 2)
        frames_with_many = sum(1 for count in object_detection_history if count >= 5)
        frames_with_lots = sum(1 for count in object_detection_history if count >= 10)
        multi_object_percentage = (frames_with_multiple / len(object_detection_history) * 100) if object_detection_history else 0
        
        print(f"\n📊 ENHANCED MULTI-OBJECT DETECTION TEST RESULTS:")
        print(f"   📈 Frames processed: {len(object_detection_history)}")
        print(f"   🎯 Average objects per frame: {avg_objects:.2f}")
        print(f"   🚀 Maximum objects in single frame: {max_objects_frame}")
        print(f"   🔢 Total detections made: {total_detections}")
        print(f"   🎭 Unique object types seen: {len(unique_objects_seen)}")
        print(f"   📊 Frames with 2+ objects: {frames_with_multiple} ({multi_object_percentage:.1f}%)")
        print(f"   🌟 Frames with 5+ objects: {frames_with_many}")
        print(f"   🎯 Frames with 10+ objects: {frames_with_lots}")
        
        if unique_objects_seen:
            print(f"\n🎭 Object Types Detected ({len(unique_objects_seen)} total):")
            sorted_objects = sorted(unique_objects_seen)
            for i in range(0, len(sorted_objects), 4):
                row = sorted_objects[i:i+4]
                print("   " + " | ".join(f"{obj:<15}" for obj in row))
        
        # Enhanced performance analysis
        print(f"\n🎯 MULTI-OBJECT DETECTION ANALYSIS:")
        if max_objects_frame >= 10:
            print("   🌟 OUTSTANDING: Detected 10+ objects in single frame!")
            grade = "A+"
        elif max_objects_frame >= 7:
            print("   � EXCELLENT: Detected 7+ objects in single frame!")
            grade = "A"
        elif max_objects_frame >= 5:
            print("   👍 VERY GOOD: Detected 5+ objects in single frame")
            grade = "B+"
        elif max_objects_frame >= 3:
            print("   ✅ GOOD: Detected 3+ objects in single frame")
            grade = "B"
        elif max_objects_frame >= 2:
            print("   📈 WORKING: Can detect multiple objects")
            grade = "C"
        else:
            print("   ⚠️ NEEDS IMPROVEMENT: Only single objects detected")
            grade = "D"
        
        if multi_object_percentage >= 70:
            print("   🎉 OUTSTANDING: 70%+ frames had multiple objects")
        elif multi_object_percentage >= 50:
            print("   🌟 EXCELLENT: 50%+ frames had multiple objects")
        elif multi_object_percentage >= 30:
            print("   👌 GOOD: 30%+ frames had multiple objects")
        elif multi_object_percentage >= 15:
            print("   📈 IMPROVING: 15%+ frames had multiple objects")
        else:
            print("   🔧 NEEDS WORK: Less than 15% frames had multiple objects")
        
        if len(unique_objects_seen) >= 15:
            print("   🔍 HIGHLY DIVERSE: Detected 15+ different object types")
        elif len(unique_objects_seen) >= 10:
            print("   🎭 VERY DIVERSE: Detected 10+ different object types")
        elif len(unique_objects_seen) >= 5:
            print("   📋 DIVERSE: Detected 5+ different object types")
        else:
            print("   🎯 LIMITED VARIETY: Few object types detected")
        
        print(f"\n📊 OVERALL GRADE: {grade}")
        
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        if avg_objects < 2:
            print("   • Place more objects in camera view simultaneously")
            print("   • Ensure good lighting for better detection accuracy")
            print("   • Try lowering confidence threshold (current: {:.2f})".format(detector.confidence_threshold))
        
        if max_objects_frame < 5:
            print("   • Use larger, more distinct objects for better detection")
            print("   • Ensure objects are well-separated and visible")
            print("   • Check camera resolution and focus quality")
            print("   • Improve lighting conditions")
        
        if len(unique_objects_seen) < 5:
            print("   • Try different types of objects (phones, bottles, books, etc.)")
            print("   • Use common household items that YOLO recognizes well")
        
        print("   • For optimal results: bright lighting, distinct objects, stable camera")
        print("   • The system now supports up to 100 detections per frame!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return True

if __name__ == "__main__":
    print("🎯 Enhanced Multi-Object Detection Test")
    print("This will test the improved system's ability to detect multiple objects simultaneously.")
    print("The system has been optimized for better multi-object detection while maintaining performance.")
    
    input("\nPress Enter to start the enhanced test...")
    test_multi_object_detection()
            return old_load(*args, **kwargs, weights_only=False)
        torch.load = safe_load
        
        # Load YOLO model
        model = YOLO('yolov8n.pt')
        torch.load = old_load
        print("✅ YOLO model loaded successfully")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n📋 Multi-Object Detection Features:")
        print("• Detects up to 50 objects simultaneously")
        print("• Color-coded bounding boxes for different object types")
        print("• Real-time object counting and statistics")
        print("• Confidence scores for each detection")
        print("\n🎮 Controls:")
        print("• Press 'q' to quit")
        print("• Try placing multiple objects in front of camera")
        print("\n🚀 Starting detection...")
        
        colors = [
            (0, 255, 255),   # Yellow
            (255, 0, 255),   # Magenta  
            (255, 255, 0),   # Cyan
            (0, 255, 0),     # Green
            (255, 0, 0),     # Blue
            (0, 0, 255),     # Red
            (255, 165, 0),   # Orange
            (128, 0, 128),   # Purple
        ]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Run YOLO detection with optimized parameters for multiple objects
            results = model(
                frame, 
                conf=0.4,      # Lower confidence to catch more objects
                iou=0.5,       # Intersection over Union threshold
                max_det=50,    # Maximum detections per image
                verbose=False
            )
            
            detected_objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        
                        detected_objects.append({
                            'class': class_name,
                            'confidence': float(conf),
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'class_id': class_id
                        })
                        
                        # Use different colors for different object types
                        color = colors[class_id % len(colors)]
                        
                        # Draw thick bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                        
                        # Enhanced label with background
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # Background rectangle for label
                        cv2.rectangle(frame, 
                                    (int(x1), int(y1) - label_size[1] - 10),
                                    (int(x1) + label_size[0] + 5, int(y1)),
                                    color, -1)
                        
                        # Label text with contrasting color
                        text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
                        cv2.putText(frame, label, (int(x1) + 2, int(y1) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Display statistics
            if detected_objects:
                # Group objects by class
                object_counts = {}
                for obj in detected_objects:
                    class_name = obj['class']
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
                
                # Show detailed statistics
                y_offset = 30
                cv2.putText(frame, "🎯 DETECTED OBJECTS:", (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_offset += 30
                
                for class_name, count in object_counts.items():
                    text = f"• {class_name}: {count}"
                    cv2.putText(frame, text, (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
                
                # Total count with prominent display
                total_text = f"TOTAL: {len(detected_objects)} objects"
                cv2.putText(frame, total_text, (10, y_offset + 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 3)
                
                # Show unique classes count
                unique_text = f"UNIQUE TYPES: {len(object_counts)}"
                cv2.putText(frame, unique_text, (10, y_offset + 45), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            else:
                cv2.putText(frame, "No objects detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Try placing objects in view", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            # Show instructions
            cv2.putText(frame, "Press 'q' to quit | Multi-Object Detection Test", 
                       (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Multi-Object Detection Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Multi-object detection test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_multi_object_detection()
