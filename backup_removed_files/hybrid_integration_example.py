#!/usr/bin/env python3
"""
HYBRID ULTIMATE DETECTION - INTEGRATION EXAMPLE
===============================================
Example script demonstrating how to integrate the Hybrid Ultimate Detection System
into your own applications for custom AI vision tasks.

Author: AI Vision System
Date: July 3, 2025
Version: 1.0
"""

import cv2
import numpy as np
import time
import os
import argparse
from datetime import datetime

# Import the Hybrid Ultimate Detection System
from hybrid_ultimate_detection import HybridUltimateDetectionSystem

def process_video_file(video_path, output_dir="output", confidence=None):
    """Process a video file with the Hybrid Ultimate Detection System"""
    print(f"🎬 Processing video: {video_path}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the detector
    detector = HybridUltimateDetectionSystem(auto_configure=True)
    if not detector.initialize_models():
        print("❌ Failed to initialize models")
        return False
    
    # Override confidence threshold if provided
    if confidence is not None:
        detector.confidence_threshold = confidence
        print(f"🎯 Set confidence threshold to {confidence}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/processed_{os.path.basename(video_path)}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"🎥 Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
    print(f"📁 Output file: {output_path}")
    
    # Process video frame by frame
    frame_index = 0
    start_time = time.time()
    detection_data = []
    
    print("⏳ Processing frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with detector
        annotated_frame, results = detector.process_frame(frame)
        
        # Add frame information
        results["frame_index"] = frame_index
        results["frame_timestamp"] = frame_index / fps
        detection_data.append(results)
        
        # Write processed frame to output video
        out.write(annotated_frame)
        
        # Display progress
        frame_index += 1
        if frame_index % 30 == 0:
            elapsed = time.time() - start_time
            percent_done = (frame_index / frame_count) * 100
            remaining = (elapsed / frame_index) * (frame_count - frame_index) if frame_index > 0 else 0
            print(f"🔄 Progress: {percent_done:.1f}% ({frame_index}/{frame_count}) | "
                  f"Time: {elapsed:.1f}s | ETA: {remaining:.1f}s | "
                  f"FPS: {results['fps']:.1f}")
    
    # Release resources
    cap.release()
    out.release()
    
    # Save detection data
    json_path = f"{output_dir}/detection_data_{timestamp}.json"
    detector.save_detection_results(json_path, {"frames": detection_data})
    
    # Print summary
    total_time = time.time() - start_time
    avg_fps = frame_index / total_time if total_time > 0 else 0
    print(f"\n✅ Processing complete!")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"🎞️  Processed frames: {frame_index}")
    print(f"⚡ Average processing FPS: {avg_fps:.1f}")
    print(f"📊 Detection statistics:")
    
    # Get and print detection statistics
    stats = detector.get_statistics()
    print(f"   - Total objects detected: {stats['total_object_detections']}")
    print(f"   - Total gestures detected: {stats['total_gesture_detections']}")
    print(f"   - Hardware used: {stats['hardware_device'].upper()}")
    print(f"   - Performance mode: {stats['performance_mode']}")
    
    # Print top detected classes
    if stats['top_detected_objects']:
        print("\n🏆 Top detected objects:")
        for obj, count in stats['top_detected_objects'].items():
            print(f"   - {obj}: {count}")
    
    print(f"\n📁 Processed video saved to: {output_path}")
    print(f"📄 Detection data saved to: {json_path}")
    
    return True

def process_image_file(image_path, output_dir="output", confidence=None):
    """Process a single image file with the Hybrid Ultimate Detection System"""
    print(f"🖼️ Processing image: {image_path}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the detector
    detector = HybridUltimateDetectionSystem(auto_configure=True)
    if not detector.initialize_models():
        print("❌ Failed to initialize models")
        return False
    
    # Override confidence threshold if provided
    if confidence is not None:
        detector.confidence_threshold = confidence
        print(f"🎯 Set confidence threshold to {confidence}")
    
    # Read image file
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not open image: {image_path}")
        return False
    
    # Process image with detector
    start_time = time.time()
    annotated_frame, results = detector.process_frame(frame)
    processing_time = time.time() - start_time
    
    # Save processed image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/processed_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, annotated_frame)
    
    # Save detection data
    json_path = f"{output_dir}/detection_data_{timestamp}.json"
    detector.save_detection_results(json_path, results)
    
    # Print results
    print(f"\n✅ Processing complete!")
    print(f"⏱️  Processing time: {processing_time:.3f}s")
    print(f"🎯 Detection results:")
    print(f"   - Objects detected: {len(results['objects'])}")
    print(f"   - Gestures detected: {len(results['gestures'])}")
    
    # Print detected objects
    if results['objects']:
        print("\n📦 Detected objects:")
        for i, obj in enumerate(results['objects']):
            print(f"   {i+1}. {obj['class_name']} ({obj['confidence']:.2f})")
    
    # Print detected gestures
    if results['gestures']:
        print("\n👋 Detected gestures:")
        for i, gesture in enumerate(results['gestures']):
            print(f"   {i+1}. {gesture['gesture']} ({gesture['confidence']:.2f})")
    
    print(f"\n📁 Processed image saved to: {output_path}")
    print(f"📄 Detection data saved to: {json_path}")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Hybrid Ultimate Detection System - Integration Example")
    
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input file (image or video)")
    
    parser.add_argument("--output", "-o", default="output",
                        help="Output directory for processed files")
    
    parser.add_argument("--confidence", "-c", type=float, default=None,
                        help="Detection confidence threshold (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return
    
    # Check if input is image or video
    file_ext = os.path.splitext(args.input)[1].lower()
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    
    if file_ext in image_extensions:
        process_image_file(args.input, args.output, args.confidence)
    elif file_ext in video_extensions:
        process_video_file(args.input, args.output, args.confidence)
    else:
        print(f"❌ Unsupported file type: {file_ext}")
        print(f"Supported image formats: {', '.join(image_extensions)}")
        print(f"Supported video formats: {', '.join(video_extensions)}")

if __name__ == "__main__":
    print("🚀 HYBRID ULTIMATE DETECTION - INTEGRATION EXAMPLE")
    print("=" * 70)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n✨ Done!")
