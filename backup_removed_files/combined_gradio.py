#!/usr/bin/env python3
"""
Combined AI Vision - Gradio Interface
Standalone script for gesture recognition + object detection in web browser
"""

import cv2
import mediapipe as mp
import numpy as np
import gradio as gr
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Load models and initialize MediaPipe
def initialize_models():
    """Initialize both gesture and object detection models"""
    models = {'gesture': None, 'yolo': None}
    
    # Load gesture model
    try:
        if os.path.exists("gesture_model.pkl"):
            models['gesture'] = joblib.load("gesture_model.pkl")
            print("✅ Gesture model loaded")
        else:
            print("⚠️ gesture_model.pkl not found - gesture detection disabled")
    except Exception as e:
        print(f"⚠️ Gesture model error: {e}")
    
    # Load YOLO model
    try:
        from ultralytics import YOLO
        import torch
        
        # Fix for PyTorch compatibility
        old_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs.pop('weights_only', None)
            return old_load(*args, **kwargs, weights_only=False)
        torch.load = safe_load
        
        models['yolo'] = YOLO('yolov8n.pt')
        torch.load = old_load
        print("✅ YOLO model loaded")
    except Exception as e:
        print(f"⚠️ YOLO model error: {e}")
    
    return models

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_solution = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
    model_complexity=0
)

# Global variables
models = initialize_models()
frame_count = 0
last_time = time.time()

def detect_objects_yolo(frame, model, confidence=0.4):
    """Detect multiple objects using YOLO with enhanced detection"""
    if model is None:
        return frame, []
    
    try:
        # Run YOLO inference with multiple detection parameters
        results = model(
            frame, 
            conf=confidence,  # Lower confidence to catch more objects
            iou=0.5,         # Intersection over Union threshold
            max_det=50,      # Maximum detections per image
            verbose=False
        )
        
        detected = []
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
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    detected.append({
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'class_id': class_id
                    })
                    
                    # Use different colors for different object types
                    color = colors[class_id % len(colors)]
                    
                    # Draw thicker bounding box for better visibility
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    
                    # Enhanced label with background for better readability
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
                    
                    # Add object index number
                    cv2.putText(frame, f"#{i+1}", (int(x1), int(y1) + 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, detected
    except Exception as e:
        print(f"YOLO error: {e}")
        return frame, []

def process_frame(image):
    """Process frame for both gesture and object detection with enhanced multi-object support"""
    global frame_count, last_time
    
    # Performance control - process every 2nd frame instead of 3rd for better detection
    frame_count += 1
    current_time = time.time()
    
    # Reduced frame skipping for better object detection
    if frame_count % 2 != 0:
        if hasattr(process_frame, "last_result"):
            return process_frame.last_result
        return None, "Processing...", "Loading...", "Initializing..."
    
    if image is None:
        return None, "No camera input", "No camera", "Check camera permissions"
    
    try:
        # Convert and resize
        frame_rgb = np.array(image)
        height, width = frame_rgb.shape[:2]
        if width > 640:
            scale = 640 / width
            frame_rgb = cv2.resize(frame_rgb, (640, int(height * scale)))
        
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        display_frame = frame_bgr.copy()
        
        # Initialize results
        gesture_result = "No hand detected"
        object_result = "No objects detected"
        stats = {}
        
        # --- GESTURE DETECTION ---
        if models['gesture'] is not None:
            hand_results = hands_solution.process(frame_rgb)
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Draw landmarks with enhanced visibility
                    mp_drawing.draw_landmarks(
                        display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)
                    )
                    
                    # Extract features
                    features = []
                    for lm in hand_landmarks.landmark:
                        features.extend([lm.x, lm.y, lm.z])
                    
                    if len(features) == 63:
                        try:
                            features_np = np.array(features).reshape(1, -1)
                            pred = models['gesture'].predict(features_np)[0]
                            
                            if hasattr(models['gesture'], "predict_proba"):
                                conf = np.max(models['gesture'].predict_proba(features_np))
                                gesture_result = f"🤟 {pred} ({int(conf*100)}% confidence)"
                            else:
                                gesture_result = f"🤟 {pred}"
                        except Exception:
                            gesture_result = "🤟 Processing error"
                    else:
                        gesture_result = "🤟 Incomplete landmarks"
            else:
                gesture_result = "🤟 No hand detected"
        else:
            gesture_result = "🤟 Model not loaded"
        
        # --- ENHANCED OBJECT DETECTION ---
        detected_objects = []
        if models['yolo'] is not None:
            display_frame, detected_objects = detect_objects_yolo(display_frame, models['yolo'], confidence=0.4)
            
            if detected_objects:
                # Group objects by class for better display
                object_groups = {}
                for obj in detected_objects:
                    class_name = obj['class']
                    if class_name not in object_groups:
                        object_groups[class_name] = []
                    object_groups[class_name].append(obj)
                
                # Create detailed object result
                if len(detected_objects) == 1:
                    obj = detected_objects[0]
                    object_result = f"🎯 {obj['class']} ({obj['confidence']:.2f})"
                elif len(detected_objects) <= 5:
                    object_list = [f"{obj['class']} ({obj['confidence']:.1f})" for obj in detected_objects]
                    object_result = f"🎯 Found {len(detected_objects)}: {', '.join(object_list)}"
                else:
                    # Show summary for many objects
                    top_objects = sorted(detected_objects, key=lambda x: x['confidence'], reverse=True)[:3]
                    object_list = [f"{obj['class']} ({obj['confidence']:.1f})" for obj in top_objects]
                    object_result = f"🎯 {len(detected_objects)} objects: {', '.join(object_list)} +{len(detected_objects)-3} more"
                
                # Add object count by type
                if len(object_groups) > 1:
                    type_counts = [f"{cls}({len(objs)})" for cls, objs in object_groups.items()]
                    object_result += f"\n📊 Types: {', '.join(type_counts)}"
                    
            else:
                object_result = "🎯 No objects detected"
        else:
            object_result = "🎯 Model not loaded"
        
        # --- ENHANCED STATISTICS ---
        fps = 1.0 / (current_time - last_time) if current_time - last_time > 0 else 0
        last_time = current_time
        
        gesture_status = "✅" if models['gesture'] else "❌"
        object_status = "✅" if models['yolo'] else "❌"
        
        # Detailed object statistics
        unique_classes = len(set(obj['class'] for obj in detected_objects))
        avg_confidence = sum(obj['confidence'] for obj in detected_objects) / len(detected_objects) if detected_objects else 0
        
        stats_result = f"""📊 System Status:
• Gesture Model: {gesture_status}
• Object Model: {object_status}
• Total Objects: {len(detected_objects)}
• Unique Types: {unique_classes}
• Avg Confidence: {avg_confidence:.2f}
• Processing FPS: {fps:.1f}
• Frame Size: {frame_rgb.shape[1]}x{frame_rgb.shape[0]}
• Detection Mode: Multi-Object Enhanced"""
        
        # Add enhanced overlay info
        cv2.putText(display_frame, "Multi-Object AI Vision", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display_frame, f"Objects: {len(detected_objects)} | Types: {unique_classes}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Convert back to RGB
        result_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Cache result
        process_frame.last_result = (result_frame, gesture_result, object_result, stats_result)
        
        return result_frame, gesture_result, object_result, stats_result
        
    except Exception as e:
        error_msg = f"Error: {str(e)[:50]}"
        error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Processing Error", (200, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return error_frame, error_msg, error_msg, error_msg

# Create Gradio interface
def create_interface():
    """Create the Gradio interface for combined AI vision with multi-object detection"""
    
    with gr.Blocks(title="Multi-Object AI Vision", theme=gr.themes.Soft()) as iface:
        gr.Markdown("# 🤖 Multi-Object AI Vision: Gestures + Multiple Objects")
        gr.Markdown("### Real-time hand gesture recognition and simultaneous multi-object detection")
        
        with gr.Row():
            with gr.Column(scale=3):
                webcam = gr.Image(
                    label="📹 Live Multi-Object AI Vision",
                    streaming=True,
                    height=480,
                    width=640,
                    mirror_webcam=False
                )
            
            with gr.Column(scale=2):
                gesture_output = gr.Textbox(
                    label="🤟 Gesture Recognition",
                    interactive=False,
                    lines=2,
                    placeholder="Detecting gestures..."
                )
                
                object_output = gr.Textbox(
                    label="🎯 Multi-Object Detection", 
                    interactive=False,
                    lines=4,
                    placeholder="Detecting multiple objects..."
                )
                
                stats_output = gr.Textbox(
                    label="📊 Detection Statistics",
                    interactive=False,
                    lines=8,
                    placeholder="Loading AI models..."
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ## 🚀 Enhanced Features:
                - **Multi-Object Detection**: Detects up to 50 objects simultaneously
                - **Color-Coded Boxes**: Different colors for different object types  
                - **Object Indexing**: Numbers on each detected object
                - **Real-time Counting**: Live count of objects and types
                - **Smart Grouping**: Groups similar objects together
                
                ## 🎯 Advanced Object Detection:
                **Detects 80+ object classes including:**
                - 👥 People, animals, vehicles
                - 📱 Electronics (phone, laptop, TV, etc.)
                - 🍽️ Food & kitchen items  
                - ⚽ Sports equipment
                - 🪑 Furniture & household items
                - And many more!
                """)
            
            with gr.Column():
                gr.Markdown("""
                ## 💡 Optimization Tips:
                - **Multiple Objects**: Point camera at scenes with many items
                - **Good Lighting**: Better lighting = more accurate detection
                - **Varied Angles**: Try different camera angles for better coverage
                - **Object Separation**: Spread objects for clearer detection
                - **Distance**: Optimal distance is 1-3 feet from camera
                
                ## 📊 Real-time Information:
                - **Object Count**: Total number detected
                - **Unique Types**: Different object categories  
                - **Confidence Scores**: Accuracy of each detection
                - **Processing Speed**: Live FPS monitoring
                - **Detection Mode**: Multi-object enhanced processing
                """)
        
        # Connect streaming
        webcam.stream(
            fn=process_frame,
            inputs=webcam,
            outputs=[webcam, gesture_output, object_output, stats_output],
            show_progress=False
        )
    
    return iface

def main():
    """Main function to run the combined interface"""
    print("🚀 Combined AI Vision - Gradio Interface")
    print("=" * 50)
    
    # Check models
    if models['gesture']:
        print("✅ Gesture recognition ready")
    else:
        print("⚠️ Gesture recognition disabled (no model)")
    
    if models['yolo']:
        print("✅ Object detection ready")
    else:
        print("⚠️ Object detection disabled (no YOLO)")
    
    if not models['gesture'] and not models['yolo']:
        print("❌ No models loaded! Please check your setup.")
        return
    
    print("\n🌐 Starting web interface...")
    print("📱 Opening at: http://localhost:7860")
    
    # Create and launch interface
    iface = create_interface()
    iface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    main()
