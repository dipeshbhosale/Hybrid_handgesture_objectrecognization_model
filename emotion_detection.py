import os
import cv2
import numpy as np
import time
import datetime
import pandas as pd
from deepface import DeepFace

class EmotionDetector:
    """
    Emotion detection system using DeepFace library.
    Can be integrated with existing detection systems.
    """
    def __init__(self):
        self.enabled = True
        self.frame_skip = 2  # Process every nth frame for performance
        self.frame_counter = 0
        self.last_emotions = {}  # Store last detected emotions by face ID
        self.logging_enabled = False
        self.log_file = "emotion_logs.csv"
        self.emoji_map = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'fear': '😨',
            'surprise': '😲',
            'neutral': '😐',
            'disgust': '🤢'
        }
        
        # Performance tracking
        self.processing_times = []
        self.max_times_history = 30
        
        # Face detection improvement parameters
        self.face_detection_params = {
            'scale_factor': 1.1,            # Scale factor for cascade classifier
            'min_neighbors': 5,             # Min neighbors for cascade classifier
            'min_face_size': (60, 60),      # Minimum face size to detect
            'brightness_correction': True,  # Apply brightness correction
            'contrast_enhancement': True,   # Apply contrast enhancement
            'blur_reduction': True,         # Apply blur reduction
            'detect_method': 'opencv',      # Face detection method
            'confidence_threshold': 0.5,    # Confidence threshold
        }
        
        # Load face cascade for backup detection
        self.face_cascade = None
        try:
            # Standard OpenCV Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("✅ Face cascade loaded for backup detection")
        except Exception as e:
            print(f"⚠️ Could not load face cascade: {e}")
        
        print("✅ Emotion detector initialized")
        
        # Create log file if logging enabled
        if self.logging_enabled and not os.path.exists(self.log_file):
            self._create_log_file()
    
    def _create_log_file(self):
        """Create emotion log file with header"""
        try:
            with open(self.log_file, 'w', newline='') as f:
                f.write("timestamp,face_id,emotion,confidence\n")
            print(f"✅ Created emotion log file: {self.log_file}")
        except Exception as e:
            print(f"❌ Error creating log file: {e}")
            self.logging_enabled = False
    
    def toggle_emotion_detection(self):
        """Toggle emotion detection on/off"""
        self.enabled = not self.enabled
        status = "ENABLED" if self.enabled else "DISABLED"
        print(f"🎭 Emotion detection: {status}")
        return self.enabled
    
    def toggle_logging(self):
        """Toggle emotion logging to CSV file"""
        self.logging_enabled = not self.logging_enabled
        
        if self.logging_enabled and not os.path.exists(self.log_file):
            self._create_log_file()
        
        status = "ENABLED" if self.logging_enabled else "DISABLED"
        print(f"📝 Emotion logging: {status}")
        return self.logging_enabled
    
    # We're replacing this method with the new enhance_frame_for_detection method
    # and detect_faces_cascade methods that were added earlier
    
    def detect_emotions(self, frame, faces=None):
        """
        Detect emotions in the given frame.
        
        Args:
            frame: The video frame (BGR format)
            faces: Optional list of face bounding boxes [x, y, w, h]
                  If None, DeepFace will detect faces internally
        
        Returns:
            Processed frame with emotion labels
            Dictionary of detected emotions
        """
        if not self.enabled:
            return frame, {}
        
        # Skip frames for performance
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            # Return last emotions but don't process new ones
            return frame, self.last_emotions
        
        # Make a copy to avoid modifying the original
        result_frame = frame.copy()
        detected_emotions = {}
        
        try:
            start_time = time.time()
            
            # Enhance frame for better face detection using the new method
            enhanced_frame = self.enhance_frame_for_detection(frame)
            
            # If no faces provided, try multi-method detection approach
            if faces is None:
                # First attempt: Try with DeepFace's internal detection
                try:
                    # Analyze the whole frame with multiple detector attempts if needed
                    for backend in ['opencv', 'ssd', 'mtcnn', 'retinaface']:
                        try:
                            analysis = DeepFace.analyze(
                                img_path=enhanced_frame,
                                actions=['emotion'],
                                enforce_detection=False,
                                detector_backend=backend
                            )
                            
                            # Handle both single face (dict) and multiple faces (list)
                            if isinstance(analysis, dict):
                                analysis = [analysis]
                                
                            # If faces detected successfully, break the loop
                            if len(analysis) > 0:
                                print(f"✅ Face detected using {backend} backend")
                                break
                        except Exception as e:
                            print(f"⚠️ {backend} detection failed: {str(e)[:50]}")
                            continue
                    
                    # If no faces detected with DeepFace backends, try cascade
                    if not 'analysis' in locals() or len(analysis) == 0:
                        faces = self.detect_faces_cascade(enhanced_frame)
                        if len(faces) > 0:
                            print(f"✅ Face detected using cascade (backup method)")
                            # Process these faces in the next section
                        else:
                            print("⚠️ No faces detected with any method")
                            analysis = []
                
                except Exception as e:
                    print(f"⚠️ Face detection error: {str(e)[:100]}")
                    # Try backup method with cascade
                    faces = self.detect_faces_cascade(enhanced_frame)
                    if len(faces) > 0:
                        print(f"✅ Face detected using cascade (backup method)")
                    else:
                        print("⚠️ No faces detected with any method")
                        analysis = []
                
                # Process each detected face
                for i, face_data in enumerate(analysis):
                    emotion = face_data['dominant_emotion']
                    emotion_scores = face_data['emotion']
                    confidence = emotion_scores[emotion]
                    
                    # Get face region if available
                    if 'region' in face_data:
                        x = face_data['region']['x']
                        y = face_data['region']['y']
                        w = face_data['region']['w']
                        h = face_data['region']['h']
                    else:
                        # If region not available, use full frame dimensions as fallback
                        x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]
                    
                    face_id = f"face_{i+1}"
                    detected_emotions[face_id] = {
                        'emotion': emotion,
                        'confidence': confidence,
                        'box': (x, y, w, h)
                    }
                    
                    # Draw on frame
                    self._draw_emotion_on_frame(result_frame, face_id, emotion, confidence, (x, y, w, h))
                    
                    # Log emotion if enabled
                    if self.logging_enabled:
                        self._log_emotion(face_id, emotion, confidence)
            
            # If faces are provided (e.g., from YOLO detection)
            else:
                for i, (x, y, w, h) in enumerate(faces):
                    # Ensure box is within frame boundaries
                    x, y, w, h = max(0, x), max(0, y), min(w, frame.shape[1]-x), min(h, frame.shape[0]-y)
                    
                    if w > 0 and h > 0:
                        # Extract face region with margin
                        margin = int(0.2 * max(w, h))  # 20% margin
                        face_x = max(0, x - margin)
                        face_y = max(0, y - margin)
                        face_w = min(frame.shape[1] - face_x, w + 2*margin)
                        face_h = min(frame.shape[0] - face_y, h + 2*margin)
                        
                        face_region = frame[face_y:face_y+face_h, face_x:face_x+face_w]
                        
                        if face_region.size > 0:
                            try:
                                analysis = DeepFace.analyze(
                                    img_path=face_region,
                                    actions=['emotion'],
                                    enforce_detection=False,
                                    detector_backend='skip'  # Skip detection as we already have the face
                                )
                                
                                emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
                                emotion_scores = analysis[0]['emotion'] if isinstance(analysis, list) else analysis['emotion']
                                confidence = emotion_scores[emotion]
                                
                                face_id = f"face_{i+1}"
                                detected_emotions[face_id] = {
                                    'emotion': emotion,
                                    'confidence': confidence,
                                    'box': (x, y, w, h)
                                }
                                
                                # Draw on frame
                                self._draw_emotion_on_frame(result_frame, face_id, emotion, confidence, (x, y, w, h))
                                
                                # Log emotion if enabled
                                if self.logging_enabled:
                                    self._log_emotion(face_id, emotion, confidence)
                                    
                            except Exception as e:
                                # If error analyzing this particular face, continue with others
                                print(f"⚠️ Error analyzing face {i+1}: {str(e)[:100]}")
            
            # Update processing time history
            process_time = time.time() - start_time
            self.processing_times.append(process_time)
            self.processing_times = self.processing_times[-self.max_times_history:]  # Keep only recent history
            
            # Update last emotions
            self.last_emotions = detected_emotions
            
            return result_frame, detected_emotions
            
        except Exception as e:
            print(f"❌ Emotion detection error: {str(e)[:100]}")
            return frame, self.last_emotions
    
    def _draw_emotion_on_frame(self, frame, face_id, emotion, confidence, bbox):
        """Draw emotion label and emoji on frame"""
        x, y, w, h = bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Prepare emotion text with emoji
        emoji = self.emoji_map.get(emotion, "")
        emotion_text = f"{emotion.upper()} {emoji} ({confidence:.0f}%)"
        
        # Draw background rectangle for text
        text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y-30), (x+text_size[0]+10, y), (0, 255, 0), -1)
        
        # Draw emotion text
        cv2.putText(
            frame, emotion_text, (x+5, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )
        
        return frame
    
    def _log_emotion(self, face_id, emotion, confidence):
        """Log detected emotion to CSV file"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, 'a', newline='') as f:
                f.write(f"{timestamp},{face_id},{emotion},{confidence}\n")
        except Exception as e:
            print(f"⚠️ Error logging emotion: {str(e)}")
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.processing_times:
            return {
                'avg_process_time': 0,
                'max_process_time': 0,
                'min_process_time': 0,
                'estimated_fps': 0
            }
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        max_time = max(self.processing_times)
        min_time = min(self.processing_times)
        
        # Estimated FPS accounting for frame skipping
        estimated_fps = 1.0 / (avg_time / self.frame_skip) if avg_time > 0 else 0
        
        return {
            'avg_process_time': avg_time,
            'max_process_time': max_time, 
            'min_process_time': min_time,
            'estimated_fps': estimated_fps,
            'frame_skip': self.frame_skip
        }
    
    def enhance_frame_for_detection(self, frame):
        """
        Enhance frame to improve face detection performance
        
        Args:
            frame: Input frame to enhance
            
        Returns:
            Enhanced frame
        """
        if frame is None:
            return None
            
        enhanced = frame.copy()
        
        # Apply brightness correction if enabled
        if self.face_detection_params['brightness_correction']:
            # Convert to LAB color space
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            # Split channels
            l, a, b = cv2.split(lab)
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            # Merge channels
            merged = cv2.merge((cl, a, b))
            # Convert back to BGR
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        # Apply contrast enhancement if enabled
        if self.face_detection_params['contrast_enhancement']:
            # Increase contrast
            alpha = 1.3  # Contrast control (1.0 means no change)
            beta = 0  # Brightness control (0 means no change)
            enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        
        # Apply blur reduction if enabled
        if self.face_detection_params['blur_reduction']:
            # Apply slight sharpening filter
            kernel = np.array([[-1, -1, -1], 
                              [-1, 9, -1], 
                              [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def detect_faces_cascade(self, frame):
        """
        Backup face detection using OpenCV cascade classifier
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected faces as (x, y, w, h)
        """
        if self.face_cascade is None or frame is None:
            return []
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.face_detection_params['scale_factor'],
            minNeighbors=self.face_detection_params['min_neighbors'],
            minSize=self.face_detection_params['min_face_size']
        )
        
        return faces
    
    def adjust_frame_skip(self, target_fps=15):
        """Automatically adjust frame skip to maintain target FPS"""
        stats = self.get_performance_stats()
        current_fps = stats['estimated_fps']
        
        if current_fps < target_fps * 0.8:  # Too slow
            self.frame_skip = min(5, self.frame_skip + 1)  # Increase skip, max 5
        elif current_fps > target_fps * 1.2 and self.frame_skip > 1:  # Too fast
            self.frame_skip = max(1, self.frame_skip - 1)  # Decrease skip, min 1
            
        return self.frame_skip

# Example usage
if __name__ == "__main__":
    print("🎭 Testing Emotion Detection System")
    
    # Initialize emotion detector
    emotion_detector = EmotionDetector()
    
    # Display face detection tips
    print("\n📋 FACE DETECTION TIPS:")
    print("  • Ensure your face is well-lit from the front")
    print("  • Position your face in the center of the frame")
    print("  • Maintain a reasonable distance (not too far, not too close)")
    print("  • Avoid extreme angles and partial face views")
    print("  • If detection fails, try adjusting room lighting")
    print("  • Minimize background distractions")
    print("  • Try pressing 'm' to cycle detection methods if face is not detected")
    print("  • Hold still for a moment when detection is difficult")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        exit()
    
    print("📷 Camera opened successfully")
    print("⌨️ Controls: q=quit | e=toggle detection | l=toggle logging | m=change detection method")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Can't receive frame")
                break
            
            frame_count += 1
            
            # Detect emotions
            processed_frame, emotions = emotion_detector.detect_emotions(frame)
            
            # Show FPS and status
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            status = "ON" if emotion_detector.enabled else "OFF"
            logging = "ON" if emotion_detector.logging_enabled else "OFF"
            
            # Get face detection method used
            backend = emotion_detector.face_detection_params['detect_method']
            
            cv2.putText(
                processed_frame, 
                f"FPS: {fps:.1f} | Detection: {status} | Logging: {logging}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            
            # Display number of faces detected and detection method
            cv2.putText(
                processed_frame,
                f"Faces: {len(emotions)} | Method: {backend}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            
            # Add face positioning guidance
            if len(emotions) == 0:
                cv2.putText(
                    processed_frame,
                    "No face detected! Position your face in center, ensure good lighting", 
                    (10, processed_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
                
                # Draw face positioning guide
                h, w = processed_frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                guide_size = min(w, h) // 3
                
                # Draw face positioning oval
                cv2.ellipse(processed_frame, 
                           (center_x, center_y), 
                           (guide_size, int(guide_size * 1.3)), 
                           0, 0, 360, (0, 255, 255), 2)
                
                # Draw face positioning markers
                cv2.circle(processed_frame, (center_x, center_y - guide_size//3), 5, (0, 255, 255), -1)  # Eyes
                cv2.circle(processed_frame, (center_x, center_y + guide_size//3), 5, (0, 255, 255), -1)  # Mouth
            
            # Display the resulting frame
            cv2.imshow('Emotion Detection', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("👋 Exiting...")
                break
            elif key & 0xFF == ord('e'):
                emotion_detector.toggle_emotion_detection()
            elif key & 0xFF == ord('l'):
                emotion_detector.toggle_logging()
            elif key & 0xFF == ord('m'):
                # Cycle through face detection methods
                methods = ['opencv', 'ssd', 'mtcnn', 'retinaface']
                current = emotion_detector.face_detection_params['detect_method']
                next_index = (methods.index(current) + 1) % len(methods) if current in methods else 0
                emotion_detector.face_detection_params['detect_method'] = methods[next_index]
                print(f"🔄 Switched to {methods[next_index]} detection method")
            
            # Adjust frame skipping every 30 frames
            if frame_count % 30 == 0:
                emotion_detector.adjust_frame_skip(target_fps=15)
    
    except Exception as e:
        print(f"❌ Error in main loop: {str(e)}")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print(f"📊 Processed {frame_count} frames in {elapsed:.1f}s (Avg FPS: {fps:.2f})")
        
        # Print performance stats
        stats = emotion_detector.get_performance_stats()
        print(f"⏱️ Avg processing time: {stats['avg_process_time']*1000:.1f}ms")
        print(f"⚡ Estimated emotion detection FPS: {stats['estimated_fps']:.1f}")
