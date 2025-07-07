#!/usr/bin/env python3
"""
HYBRID ULTIMATE DETECTION - WEB INTERFACE
=========================================
Browser-based interface for the Hybrid Ultimate Detection System
Provides interactive controls, visualization, and statistics

Author: AI Vision System
Date: July 3, 2025
Version: 1.0
"""

import cv2
import numpy as np
import gradio as gr
import time
from datetime import datetime
import os
from collections import defaultdict, deque
import json
import threading
import warnings
warnings.filterwarnings('ignore')

# Import the Hybrid Ultimate Detection System
from hybrid_ultimate_detection import HybridUltimateDetectionSystem

class HybridWebInterface:
    """
    Web interface for Hybrid Ultimate Detection System
    Features:
    - Live video with detection overlay
    - Performance statistics
    - Mode switching
    - Screenshot capture
    - Detection history
    """
    
    def __init__(self):
        """Initialize the web interface and detection system"""
        self.detector = None
        self.current_frame = None
        self.current_results = None
        self.running = False
        self.frame_count = 0
        self.detection_history = []
        self.history_limit = 100
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.modes = ["efficient", "fast", "balanced", "ultra_accurate"]
        self.performance_mode = "balanced"  # Default mode
        
        # Create screenshots directory if it doesn't exist
        self.screenshots_dir = "detection_screenshots"
        if not os.path.exists(self.screenshots_dir):
            os.makedirs(self.screenshots_dir)
    
    def initialize_detector(self, progress=gr.Progress()):
        """Initialize the detection system with progress updates"""
        try:
            progress(0, desc="Initializing Hybrid Ultimate Detection System...")
            self.detector = HybridUltimateDetectionSystem(auto_configure=True)
            
            progress(0.3, desc="Loading detection models...")
            init_success = self.detector.initialize_models()
            if not init_success:
                return "⚠️ Warning: Some models failed to initialize. Limited functionality available."
            
            progress(0.5, desc="Loading YOLO models...")
            # Ensure YOLO models are initialized
            if not self.detector.yolo_models or len(self.detector.yolo_models) == 0:
                self.detector._load_yolo_models()
            
            progress(0.7, desc="Loading gesture model...")
            # Ensure gesture model is initialized
            if self.detector.gesture_model is None:
                self.detector._load_gesture_model()
                
            progress(0.8, desc="Configuring system...")
            self.performance_mode = self.detector.performance_mode
            
            progress(0.9, desc="Preparing camera...")
            # Pre-warm the models with a dummy frame
            dummy_frame = np.ones((640, 480, 3), dtype=np.uint8) * 128
            self.detector.process_frame(dummy_frame)
            
            progress(1.0, desc="Ready!")
            return f"✅ System initialized in {self.performance_mode} mode! Stream is now active - detection will begin automatically."
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error initializing detector: {str(e)}"
    
    def process_video_frame(self, frame):
        """Process incoming video frames"""
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8), "No video input"
        
        if self.detector is None:
            # Create a message on the frame to initialize the system
            message_frame = frame.copy()
            cv2.putText(message_frame, "Click 'Initialize System' first",
                       (50, message_frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return message_frame, "System not initialized"
        
        try:
            # Convert from RGB to BGR for processing
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Process the frame
            annotated_frame, results = self.detector.process_frame(frame_bgr)
            
            # Convert back to RGB for Gradio
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Keep track of current frame and results
            self.current_frame = annotated_frame.copy()
            self.current_results = results
            
            # Update history (limit size)
            self.frame_count += 1
            if self.frame_count % 5 == 0:  # Store every 5th frame to avoid too much data
                if len(self.detection_history) >= self.history_limit:
                    self.detection_history.pop(0)
                
                # Add timestamp
                results_with_time = results.copy()
                results_with_time['timestamp'] = time.time()
                self.detection_history.append(results_with_time)
            
            # Get FPS and stats
            current_fps = results.get('fps', 0)
            
            # Create status text
            objects_count = len(results.get('objects', []))
            gestures_count = len(results.get('gestures', []))
            status_text = f"FPS: {current_fps:.1f} | Mode: {self.detector.performance_mode} | Objects: {objects_count} | Gestures: {gestures_count}"
            
            # Add some detection info to the frame for better visibility
            if objects_count > 0 or gestures_count > 0:
                detection_summary = []
                
                # Get top 3 objects
                if objects_count > 0:
                    object_counts = {}
                    for obj in results.get('objects', []):
                        obj_class = obj.get('class_name', 'unknown')
                        if obj_class in object_counts:
                            object_counts[obj_class] += 1
                        else:
                            object_counts[obj_class] = 1
                    
                    top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    detection_summary.extend([f"{count}× {name}" for name, count in top_objects])
                
                # Get gestures
                if gestures_count > 0:
                    gesture_names = [g.get('gesture_name', 'unknown') for g in results.get('gestures', [])]
                    detection_summary.extend(gesture_names)
                
                status_text += f" | Detected: {', '.join(detection_summary)}"
            
            return annotated_frame_rgb, status_text
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in video processing: {str(e)}")
            
            # Return original frame with error message
            error_frame = frame.copy()
            cv2.putText(error_frame, f"Error: {str(e)}",
                       (30, error_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            return error_frame, f"Error: {str(e)}"
    
    def take_screenshot(self):
        """Save current frame with detections as screenshot"""
        if self.current_frame is None:
            return "No frame available to capture"
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.screenshots_dir}/detection_{timestamp}.jpg"
            
            # Save image
            cv2.imwrite(filename, self.current_frame)
            
            # Save detection data if available
            if self.current_results:
                json_filename = f"{self.screenshots_dir}/detection_{timestamp}.json"
                
                # Convert results to serializable format
                serializable_results = self._make_json_serializable(self.current_results)
                
                with open(json_filename, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
            
            return f"✅ Screenshot saved: {filename}"
            
        except Exception as e:
            return f"❌ Error saving screenshot: {str(e)}"
    
    def _make_json_serializable(self, obj):
        """Make complex objects JSON serializable"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def get_detection_stats(self):
        """Generate detection statistics and charts"""
        if not self.detector or not self.detection_history:
            return "No detection data available yet. Start detection first."
        
        # Calculate statistics
        stats = self.detector.get_statistics()
        
        # Format into human-readable text
        runtime = time.time() - self.start_time
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        stats_text = f"## 📊 Detection Statistics\n\n"
        stats_text += f"**Session Duration:** {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}\n\n"
        stats_text += f"**System Performance**\n"
        stats_text += f"- Performance Mode: {stats['performance_mode']}\n"
        stats_text += f"- Average FPS: {stats['average_fps']:.2f}\n"
        stats_text += f"- Hardware: {stats['hardware_device'].upper()}\n"
        stats_text += f"- YOLO Models: {stats['yolo_models_loaded']}\n\n"
        
        stats_text += f"**Detection Counts**\n"
        stats_text += f"- Total Object Detections: {stats['total_object_detections']}\n"
        stats_text += f"- Total Gesture Detections: {stats['total_gesture_detections']}\n\n"
        
        # Top detected objects
        if stats['top_detected_objects']:
            stats_text += f"**Top Detected Objects**\n"
            for obj, count in stats['top_detected_objects'].items():
                stats_text += f"- {obj}: {count}\n"
            stats_text += "\n"
        
        # Top detected gestures
        if stats['top_detected_gestures']:
            stats_text += f"**Top Detected Gestures**\n"
            for gesture, count in stats['top_detected_gestures'].items():
                stats_text += f"- {gesture}: {count}\n"
        
        return stats_text
    
    def switch_performance_mode(self, mode):
        """Switch the detection system's performance mode"""
        if not self.detector:
            return f"⚠️ Detector not initialized"
        
        try:
            self.detector.adjust_performance_mode(mode)
            return f"✅ Switched to {mode} mode"
        except Exception as e:
            return f"❌ Failed to switch mode: {str(e)}"

def create_web_interface():
    """Create and launch the Gradio web interface"""
    interface = HybridWebInterface()
    
    with gr.Blocks(title="Hybrid Ultimate Detection System", theme="soft", css="""
        .detection-container {border: 2px solid #4CAF50; border-radius: 8px; padding: 10px;}
        .detection-title {color: #4CAF50; font-weight: bold;}
        .webcam-feed {border: 1px solid #ddd; border-radius: 5px;}
        .status-box {background-color: #f8f9fa; padding: 5px; border-radius: 5px; font-family: monospace;}
    """) as app:
        gr.Markdown("""
        # 🚀 Hybrid Ultimate Detection System
        
        Real-time object detection and gesture recognition with adaptive performance modes.
        
        Features:
        - Multi-model YOLOv8 object detection
        - Hand gesture recognition
        - Auto-optimization for best performance
        - Multiple performance modes
        
        ## Quick Start:
        1. Click "Initialize System" button below
        2. Wait for system initialization to complete
        3. Detection will begin automatically on webcam feed
        4. Make hand gestures in front of camera to test recognition
        """)
        
        # Initialize system button (do this first)
        with gr.Row():
            init_button = gr.Button("1️⃣ Initialize System", variant="primary", size="lg")
            init_output = gr.Textbox(label="System Status", interactive=False)
        
        # Main video display area
        with gr.Row():
            with gr.Column(scale=3):
                # Use larger video displays for better visibility
                video_input = gr.Image(sources="webcam", streaming=True, label="Camera Input", 
                                      height=400, width=600, scale=1, type="numpy")
                video_output = gr.Image(label="Detection Output", 
                                       height=400, width=600, type="numpy", interactive=False)
                status_text = gr.Textbox(label="Detection Status", interactive=False)
            
            with gr.Column(scale=1):
                # Performance mode selector
                mode_dropdown = gr.Dropdown(
                    choices=interface.modes,
                    value="balanced",
                    label="Performance Mode",
                    interactive=True,
                    info="Change detection speed vs. accuracy"
                )
                
                # Screenshots and stats
                screenshot_button = gr.Button("📸 Take Screenshot")
                screenshot_result = gr.Textbox(label="Screenshot Result")
                stats_button = gr.Button("📊 Show Statistics")
                stats_output = gr.Markdown(label="Detection Statistics")
                
                # Video is automatically captured while this is active
                video_active = gr.Checkbox(label="Video Active", visible=False, value=False)
        
        # Event handlers
        init_button.click(
            fn=interface.initialize_detector,
            outputs=[init_output]
        )
        
        # Mode switching
        mode_dropdown.change(
            fn=interface.switch_performance_mode,
            inputs=[mode_dropdown],
            outputs=[status_text]
        )
        
        # Video processing - use stream instead of change for real-time processing
        video_input.stream(
            fn=interface.process_video_frame,
            inputs=[video_input],
            outputs=[video_output, status_text],
            show_progress=False
        )
        
        # Screenshot button
        screenshot_button.click(
            fn=interface.take_screenshot,
            outputs=[screenshot_result]
        )
        
        # Statistics button
        stats_button.click(
            fn=interface.get_detection_stats,
            outputs=[stats_output]
        )
    
    # Launch the interface
    app.launch(share=False, server_name="127.0.0.1")

if __name__ == "__main__":
    print("🚀 Starting Hybrid Ultimate Detection Web Interface...")
    print("=" * 60)
    print("🌐 The system will open in your web browser")
    print("📋 System Requirements:")
    print("  • Python 3.8+ with required packages")
    print("  • Webcam or video source")
    print("  • YOLO models in current directory")
    print("=" * 60)
    print("📌 Access the interface at: http://127.0.0.1:7860")
    
    create_web_interface()
