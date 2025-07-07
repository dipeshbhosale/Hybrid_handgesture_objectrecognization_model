import os
import cv2
import time
import numpy as np
from datetime import datetime

# Check if DeepFace is available
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# Import local emotion detection module
try:
    from emotion_detection import EmotionDetector
    EMOTION_DETECTOR_AVAILABLE = True
except ImportError:
    EMOTION_DETECTOR_AVAILABLE = False

# Try to import hybrid detection system
try:
    from hybrid_ultimate_detection import HybridUltimateDetectionSystem
    HYBRID_SYSTEM_AVAILABLE = True
except ImportError:
    HYBRID_SYSTEM_AVAILABLE = False

class EmotionHybridIntegrator:
    """
    Integrates emotion detection with the Hybrid Ultimate Detection System.
    Acts as a bridge between the two systems without modifying their code.
    """
    def __init__(self):
        self.emotion_detector = None
        self.hybrid_detector = None
        
        # Configuration
        self.emotion_enabled = True
        self.use_yolo_faces = True  # Use YOLO face detection when available
        
        # Initialize emotion detection if available
        if EMOTION_DETECTOR_AVAILABLE:
            try:
                self.emotion_detector = EmotionDetector()
                print("✅ Emotion detection system initialized")
            except Exception as e:
                print(f"❌ Failed to initialize emotion detector: {e}")
                self.emotion_detector = None
        else:
            print("❌ Emotion detection module not available")
            print("💡 Make sure emotion_detection.py is in the current directory")
        
        # Initialize hybrid detection if available
        if HYBRID_SYSTEM_AVAILABLE:
            try:
                self.hybrid_detector = HybridUltimateDetectionSystem()
                print("✅ Hybrid detection system initialized")
            except Exception as e:
                print(f"❌ Failed to initialize hybrid detector: {e}")
                self.hybrid_detector = None
        else:
            print("❌ Hybrid detection system not available")
            print("💡 Make sure hybrid_ultimate_detection.py is in the current directory")
    
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        status = {
            "emotion_detection": EMOTION_DETECTOR_AVAILABLE,
            "deepface": DEEPFACE_AVAILABLE,
            "hybrid_system": HYBRID_SYSTEM_AVAILABLE,
            "opencv": cv2.__version__
        }
        
        print("\n📋 DEPENDENCY STATUS:")
        for dep, avail in status.items():
            status_icon = "✅" if avail else "❌"
            print(f"  {status_icon} {dep}: {avail}")
        
        if not DEEPFACE_AVAILABLE:
            print("\n💡 To install DeepFace: pip install deepface")
        
        return status
    
    def toggle_emotion_detection(self):
        """Toggle emotion detection on/off"""
        if self.emotion_detector:
            self.emotion_enabled = self.emotion_detector.toggle_emotion_detection()
        else:
            self.emotion_enabled = False
            print("❌ Emotion detection not available")
        
        return self.emotion_enabled
    
    def toggle_emotion_logging(self):
        """Toggle emotion logging to CSV"""
        if self.emotion_detector:
            return self.emotion_detector.toggle_logging()
        return False
    
    def process_frame(self, frame):
        """
        Process a frame with both systems.
        First runs hybrid detection, then adds emotion detection.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Processed frame with both detection results
            Dictionary with detection results
        """
        # Initialize result container
        results = {
            "hybrid_detections": None,
            "emotions": None,
            "performance": {
                "hybrid_time": 0,
                "emotion_time": 0,
                "total_time": 0
            }
        }
        
        if frame is None:
            return frame, results
        
        # Make a copy of the frame to avoid modifying the original
        processed_frame = frame.copy()
        start_time = time.time()
        
        # Step 1: Run hybrid detection if available
        faces = []
        if self.hybrid_detector:
            try:
                hybrid_start = time.time()
                # Process frame with hybrid detector
                processed_frame, hybrid_detections = self.hybrid_detector.process_frame(frame)
                results["hybrid_detections"] = hybrid_detections
                
                # Extract face detections if using YOLO for face detection
                if self.use_yolo_faces and hybrid_detections:
                    for detection in hybrid_detections:
                        if detection.get('class_name', '').lower() == 'person' or detection.get('class_name', '').lower() == 'face':
                            x, y, w, h = detection['box']
                            # Focus on the face region (upper 1/3 of person detection)
                            if detection.get('class_name', '').lower() == 'person':
                                face_h = h // 3  # Top third is likely the face
                                faces.append((x, y, w, face_h))
                            else:
                                faces.append((x, y, w, h))
                
                results["performance"]["hybrid_time"] = time.time() - hybrid_start
                
            except Exception as e:
                print(f"⚠️ Error in hybrid detection: {e}")
        
        # Step 2: Run emotion detection if enabled and available
        if self.emotion_enabled and self.emotion_detector:
            try:
                emotion_start = time.time()
                
                # If we have face detections from YOLO, use them
                # Otherwise let the emotion detector handle face detection
                if faces and self.use_yolo_faces:
                    processed_frame, emotions = self.emotion_detector.detect_emotions(
                        processed_frame, faces=faces
                    )
                else:
                    processed_frame, emotions = self.emotion_detector.detect_emotions(
                        processed_frame
                    )
                
                results["emotions"] = emotions
                results["performance"]["emotion_time"] = time.time() - emotion_start
                
                # Add emotion info to the processed frame
                if emotions:
                    y_pos = 100  # Starting position for emotion summary
                    cv2.putText(
                        processed_frame,
                        "🎭 EMOTIONS:",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                    )
                    y_pos += 25
                    
                    for face_id, data in emotions.items():
                        emoji = self.emotion_detector.emoji_map.get(data['emotion'], "")
                        text = f"{face_id}: {data['emotion']} {emoji}"
                        cv2.putText(
                            processed_frame,
                            text,
                            (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                        )
                        y_pos += 20
                
            except Exception as e:
                print(f"⚠️ Error in emotion detection: {e}")
        
        # Calculate total processing time
        results["performance"]["total_time"] = time.time() - start_time
        
        # Add system status overlay
        self._add_status_overlay(processed_frame, results)
        
        return processed_frame, results
    
    def _add_status_overlay(self, frame, results):
        """Add status information overlay to frame"""
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # System status panel (top-right corner)
        panel_width = 200
        panel_height = 120
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (panel_x, panel_y), 
            (panel_x + panel_width, panel_y + panel_height),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Panel border
        cv2.rectangle(
            frame, (panel_x, panel_y), 
            (panel_x + panel_width, panel_y + panel_height),
            (255, 255, 0), 2
        )
        
        # Title
        cv2.putText(
            frame, "SYSTEM STATUS", (panel_x + 10, panel_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
        )
        
        # Hybrid detection status
        hybrid_status = "✅" if self.hybrid_detector else "❌"
        cv2.putText(
            frame, f"Hybrid: {hybrid_status}", (panel_x + 10, panel_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
        
        # Emotion detection status
        emotion_status = "✅" if self.emotion_detector and self.emotion_enabled else "❌"
        cv2.putText(
            frame, f"Emotion: {emotion_status}", (panel_x + 10, panel_y + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
        
        # Performance info
        total_ms = results["performance"]["total_time"] * 1000
        cv2.putText(
            frame, f"Process: {total_ms:.1f}ms", (panel_x + 10, panel_y + 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
        
        # Controls info
        cv2.putText(
            frame, "E=Toggle Emotion L=Log", (panel_x + 10, panel_y + 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
        )
    
    def run_camera_detection(self):
        """Run real-time detection with webcam input"""
        if not self.emotion_detector:
            print("❌ Emotion detector not available")
            return
        
        print("\n" + "="*60)
        print("🚀 EMOTION-HYBRID INTEGRATION SYSTEM")
        print("="*60)
        print("📋 FEATURES:")
        print("  • Combined emotion and object detection")
        print("  • Real-time processing at 8-15 FPS")
        print("  • Emotion trend logging capability")
        print("  • YOLOv8 integration for better face detection")
        print("\n📋 CONTROLS:")
        print("  • Press 'q' to quit")
        print("  • Press 'e' to toggle emotion detection")
        print("  • Press 'l' to toggle emotion logging")
        print("  • Press 's' to save current frame")
        print("="*60)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()
        fps_history = []
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                frame_count += 1
                frame_start = time.time()
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                
                # Add FPS info
                cv2.putText(
                    processed_frame, f"FPS: {avg_fps:.1f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                
                # Display the frame
                cv2.imshow("Emotion-Hybrid Integration", processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    print("👋 Exiting...")
                    break
                elif key & 0xFF == ord('e'):
                    self.toggle_emotion_detection()
                elif key & 0xFF == ord('l'):
                    self.toggle_emotion_logging()
                elif key & 0xFF == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"emotion_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"📸 Frame saved as {filename}")
                
                # Adjust emotion detector frame skipping periodically
                if self.emotion_detector and frame_count % 30 == 0:
                    self.emotion_detector.adjust_frame_skip(target_fps=15)
        
        except Exception as e:
            print(f"❌ Error in detection loop: {e}")
        
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            elapsed_time = time.time() - start_time
            print(f"\n📊 FINAL STATISTICS:")
            print(f"  • Frames processed: {frame_count}")
            print(f"  • Total time: {elapsed_time:.2f}s")
            print(f"  • Average FPS: {frame_count/elapsed_time:.2f}")
            
            if self.emotion_detector:
                stats = self.emotion_detector.get_performance_stats()
                print(f"  • Emotion detection time: {stats['avg_process_time']*1000:.2f}ms")
                print(f"  • Frame skip: {stats['frame_skip']}")

def run_emotion_detection_system():
    """
    Run the emotion detection system integrated with hybrid detection if available
    """
    # Check if DeepFace is installed
    if not DEEPFACE_AVAILABLE:
        print("❌ DeepFace library not installed.")
        print("💡 Please install it using: pip install deepface")
        return
    
    # Initialize the integrator
    integrator = EmotionHybridIntegrator()
    
    # Check dependencies
    dependency_status = integrator.check_dependencies()
    
    # If both systems available, run the integrated version
    if dependency_status["emotion_detection"] and dependency_status["hybrid_system"]:
        print("\n✅ Both emotion and hybrid detection systems available.")
        print("🚀 Starting integrated detection system...")
        integrator.run_camera_detection()
    
    # If only emotion detection is available, run standalone mode
    elif dependency_status["emotion_detection"]:
        print("\n⚠️ Hybrid detection not available, running emotion detection only.")
        if integrator.emotion_detector:
            # Run the standalone emotion detector demo
            print("🚀 Starting standalone emotion detection...")
            import emotion_detection
            emotion_detection.main()
        else:
            print("❌ Failed to initialize emotion detector.")
    
    # If nothing is available, suggest installation
    else:
        print("\n❌ Neither system is available.")
        print("💡 Please ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        print("   pip install deepface")

if __name__ == "__main__":
    print("🎭 EMOTION-HYBRID INTEGRATION")
    print("="*40)
    run_emotion_detection_system()
