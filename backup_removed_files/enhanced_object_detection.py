#!/usr/bin/env python3
"""
Enhanced Object Detection System
Advanced multi-object detection with custom training and improved models
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import requests
import os
import zipfile
from pathlib import Path
import time
import json

class EnhancedObjectDetector:
    """Advanced object detection with multiple models and custom training capabilities"""
    
    def __init__(self):
        self.models = {}
        self.confidence_threshold = 0.3  # Lower for better detection
        self.iou_threshold = 0.4
        self.max_detections = 100
        self.colors = self._generate_colors()
        self.detection_history = []
        
    def _generate_colors(self):
        """Generate vibrant colors for different object classes"""
        colors = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green  
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 165, 0),   # Orange
            (128, 0, 128),   # Purple
            (255, 192, 203), # Pink
            (0, 128, 0),     # Dark Green
            (128, 128, 0),   # Olive
            (0, 0, 128),     # Navy
            (128, 0, 0),     # Maroon
            (0, 128, 128),   # Teal
            (192, 192, 192), # Silver
            (255, 215, 0),   # Gold
            (255, 69, 0),    # Red Orange
            (50, 205, 50),   # Lime Green
            (30, 144, 255),  # Dodger Blue
            (220, 20, 60)    # Crimson
        ]
        return colors
    
    def fix_torch_compatibility(self):
        """Fix PyTorch loading issues"""
        try:
            # Add safe globals for ultralytics
            torch.serialization.add_safe_globals([
                'ultralytics.nn.tasks.DetectionModel',
                'ultralytics.nn.modules.head.Detect',
                'ultralytics.nn.modules.conv.Conv',
                'ultralytics.nn.modules.block.C2f',
                'ultralytics.nn.modules.block.SPPF',
                'collections.OrderedDict'
            ])
            
            # Alternative: patch torch.load
            original_load = torch.load
            def safe_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = safe_load
            
            return True
        except Exception as e:
            print(f"⚠️ Torch compatibility fix warning: {e}")
            return False
    
    def download_better_models(self):
        """Download and setup better YOLO models"""
        models_to_download = [
            ('yolov8s.pt', 'YOLOv8 Small - Better accuracy'),
            ('yolov8m.pt', 'YOLOv8 Medium - High accuracy'),  
            ('yolov8l.pt', 'YOLOv8 Large - Highest accuracy'),
            ('yolov8x.pt', 'YOLOv8 Extra Large - Maximum accuracy')
        ]
        
        print("🚀 Downloading enhanced YOLO models...")
        
        for model_name, description in models_to_download:
            if not os.path.exists(model_name):
                print(f"📥 Downloading {model_name} - {description}")
                try:
                    # YOLO will auto-download if not present
                    model = YOLO(model_name)
                    print(f"✅ {model_name} downloaded successfully")
                except Exception as e:
                    print(f"❌ Failed to download {model_name}: {e}")
            else:
                print(f"✅ {model_name} already exists")
    
    def load_models(self):
        """Load multiple YOLO models for ensemble detection"""
        self.fix_torch_compatibility()
        
        model_configs = [
            ('yolov8n.pt', 'nano', 'Fast detection'),
            ('yolov8s.pt', 'small', 'Balanced performance'),
            ('yolov8m.pt', 'medium', 'High accuracy'),
        ]
        
        print("🔄 Loading YOLO models...")
        
        for model_path, model_type, description in model_configs:
            try:
                if os.path.exists(model_path):
                    model = YOLO(model_path)
                    self.models[model_type] = {
                        'model': model,
                        'path': model_path,
                        'description': description,
                        'classes': model.names
                    }
                    print(f"✅ Loaded {model_type} model: {description}")
                    print(f"   Classes available: {len(model.names)}")
                else:
                    print(f"⚠️ {model_path} not found, attempting download...")
                    model = YOLO(model_path)  # Auto-download
                    self.models[model_type] = {
                        'model': model,
                        'path': model_path, 
                        'description': description,
                        'classes': model.names
                    }
                    print(f"✅ Downloaded and loaded {model_type} model")
                    
            except Exception as e:
                print(f"❌ Failed to load {model_path}: {e}")
        
        if not self.models:
            print("❌ No models loaded successfully!")
            return False
        
        # Set primary model (best available)
        if 'medium' in self.models:
            self.primary_model = 'medium'
        elif 'small' in self.models:
            self.primary_model = 'small'
        else:
            self.primary_model = list(self.models.keys())[0]
        
        print(f"🎯 Primary model set to: {self.primary_model}")
        print(f"📊 Total models loaded: {len(self.models)}")
        
        return True
    
    def detect_objects_enhanced(self, frame):
        """Enhanced object detection with multiple optimizations"""
        if not self.models:
            return frame, []
        
        try:
            # Use primary model for detection
            model_info = self.models[self.primary_model]
            model = model_info['model']
            
            # Enhanced detection parameters
            results = model(
                frame,
                conf=self.confidence_threshold,  # Lower threshold for more detections
                iou=self.iou_threshold,         # Lower IOU for overlapping objects  
                max_det=self.max_detections,    # More detections allowed
                verbose=False,
                save=False,
                show=False,
                device='0' if torch.cuda.is_available() else 'cpu'
            )
            
            detected_objects = []
            object_counts = {}
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        # Extract box information
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        
                        # Count objects by class
                        object_counts[class_name] = object_counts.get(class_name, 0) + 1
                        
                        detected_objects.append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'class_id': class_id,
                            'area': (x2 - x1) * (y2 - y1)
                        })
                        
                        # Enhanced visualization
                        color = self.colors[class_id % len(self.colors)]
                        
                        # Draw thick bounding box with gradient effect
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                        cv2.rectangle(frame, (int(x1)-1, int(y1)-1), (int(x2)+1, int(y2)+1), (255,255,255), 1)
                        
                        # Enhanced label with better styling
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        
                        # Background rectangle with shadow effect
                        cv2.rectangle(frame, 
                                    (int(x1), int(y1) - label_size[1] - 15),
                                    (int(x1) + label_size[0] + 10, int(y1) - 5),
                                    (0, 0, 0), -1)  # Black background
                        cv2.rectangle(frame, 
                                    (int(x1), int(y1) - label_size[1] - 12),
                                    (int(x1) + label_size[0] + 8, int(y1) - 2),
                                    color, -1)  # Colored background
                        
                        # White text for better contrast
                        cv2.putText(frame, label, (int(x1) + 4, int(y1) - 8),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Add object number
                        cv2.putText(frame, f"#{i+1}", (int(x1), int(y2) + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add detection statistics to frame
            self._add_statistics_overlay(frame, detected_objects, object_counts)
            
            # Store detection history for analysis
            self.detection_history.append({
                'timestamp': time.time(),
                'total_objects': len(detected_objects),
                'object_counts': object_counts,
                'model_used': self.primary_model
            })
            
            # Keep only last 100 detections
            if len(self.detection_history) > 100:
                self.detection_history.pop(0)
            
            return frame, detected_objects
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            cv2.putText(frame, f"Detection Error: {str(e)[:30]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, []
    
    def _add_statistics_overlay(self, frame, detected_objects, object_counts):
        """Add enhanced statistics overlay to the frame"""
        height, width = frame.shape[:2]
        
        # Statistics panel background
        panel_width = 300
        panel_height = min(400, height - 20)
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "🎯 DETECTION STATS", (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_offset = panel_y + 60
        
        # Total objects
        cv2.putText(frame, f"Total Objects: {len(detected_objects)}", 
                   (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        # Unique classes
        cv2.putText(frame, f"Unique Types: {len(object_counts)}", 
                   (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        # Model info
        cv2.putText(frame, f"Model: {self.primary_model.upper()}", 
                   (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_offset += 25
        
        # Separator line
        cv2.line(frame, (panel_x + 10, y_offset), (panel_x + panel_width - 10, y_offset),
                (255, 255, 255), 1)
        y_offset += 20
        
        # Object counts
        cv2.putText(frame, "DETECTED OBJECTS:", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        y_offset += 25
        
        for i, (class_name, count) in enumerate(sorted(object_counts.items())):
            if y_offset < panel_y + panel_height - 30:
                color = self.colors[i % len(self.colors)]
                cv2.putText(frame, f"• {class_name}: {count}", 
                           (panel_x + 15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
        
        # Performance info at bottom
        if self.detection_history:
            avg_objects = np.mean([h['total_objects'] for h in self.detection_history[-10:]])
            cv2.putText(frame, f"Avg Objects: {avg_objects:.1f}", 
                       (panel_x + 10, panel_y + panel_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    def ensemble_detection(self, frame):
        """Use multiple models for better detection accuracy"""
        if len(self.models) < 2:
            return self.detect_objects_enhanced(frame)
        
        all_detections = []
        
        # Run detection with multiple models
        for model_name, model_info in list(self.models.items())[:2]:  # Use top 2 models
            try:
                model = model_info['model']
                results = model(frame, conf=self.confidence_threshold, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = model.names[class_id]
                            
                            all_detections.append({
                                'class': class_name,
                                'confidence': conf,
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'class_id': class_id,
                                'model': model_name
                            })
            except Exception as e:
                print(f"Error in ensemble detection with {model_name}: {e}")
        
        # Apply Non-Maximum Suppression to remove duplicates
        filtered_detections = self._apply_nms(all_detections)
        
        # Visualize filtered detections
        frame = self._visualize_detections(frame, filtered_detections)
        
        return frame, filtered_detections
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if not detections:
            return []
        
        # Group by class
        class_groups = {}
        for det in detections:
            class_name = det['class']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(det)
        
        filtered = []
        
        for class_name, group in class_groups.items():
            # Sort by confidence
            group.sort(key=lambda x: x['confidence'], reverse=True)
            
            keep = []
            while group:
                # Keep highest confidence detection
                best = group.pop(0)
                keep.append(best)
                
                # Remove overlapping detections
                group = [det for det in group 
                        if self._calculate_iou(best['bbox'], det['bbox']) < iou_threshold]
            
            filtered.extend(keep)
        
        return filtered
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        # Calculate intersection
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _visualize_detections(self, frame, detections):
        """Enhanced visualization of detections"""
        object_counts = {}
        
        for i, det in enumerate(detections):
            class_name = det['class']
            conf = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            # Enhanced visualization
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box with thickness based on confidence
            thickness = max(2, int(conf * 6))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Enhanced label
            label = f"{class_name}: {conf:.2f}"
            if 'model' in det:
                label += f" ({det['model'][0].upper()})"
            
            # Label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0] + 5, y1), color, -1)
            
            # Label text
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add statistics
        self._add_statistics_overlay(frame, detections, object_counts)
        
        return frame
    
    def run_enhanced_detection(self):
        """Run enhanced object detection with real-time processing"""
        print("🚀 Starting Enhanced Object Detection System")
        print("=" * 60)
        
        # Initialize models
        if not self.load_models():
            print("❌ Failed to load models. Exiting.")
            return
        
        # Camera setup
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"\n🎯 Enhanced Detection Features:")
        print(f"• Multiple YOLO models: {', '.join(self.models.keys())}")
        print(f"• Confidence threshold: {self.confidence_threshold}")
        print(f"• Maximum detections: {self.max_detections}")
        print(f"• Enhanced visualization with statistics")
        print(f"• Real-time performance monitoring")
        print(f"\n🎮 Controls:")
        print(f"• 'q' - Quit")
        print(f"• 's' - Switch model")
        print(f"• '+/-' - Adjust confidence")
        print(f"• 'e' - Toggle ensemble mode")
        print(f"\n🚀 Starting detection...")
        
        ensemble_mode = False
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            fps_counter += 1
            
            # Run detection
            if ensemble_mode and len(self.models) > 1:
                annotated_frame, detections = self.ensemble_detection(frame)
            else:
                annotated_frame, detections = self.detect_objects_enhanced(frame)
            
            # Add FPS counter
            if fps_counter % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - start_time)
                start_time = current_time
                
            # Add control info
            cv2.putText(annotated_frame, "Enhanced Object Detection", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            mode_text = "ENSEMBLE" if ensemble_mode else self.primary_model.upper()
            cv2.putText(annotated_frame, f"Mode: {mode_text}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(annotated_frame, f"Confidence: {self.confidence_threshold:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow("Enhanced Object Detection", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Switch model
                model_names = list(self.models.keys())
                current_idx = model_names.index(self.primary_model)
                self.primary_model = model_names[(current_idx + 1) % len(model_names)]
                print(f"🔄 Switched to {self.primary_model} model")
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = min(0.9, self.confidence_threshold + 0.05)
                print(f"📈 Confidence: {self.confidence_threshold:.2f}")
            elif key == ord('-'):
                self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                print(f"📉 Confidence: {self.confidence_threshold:.2f}")
            elif key == ord('e'):
                ensemble_mode = not ensemble_mode
                mode_text = "ENSEMBLE" if ensemble_mode else "SINGLE"
                print(f"🔄 Detection mode: {mode_text}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print detection summary
        if self.detection_history:
            avg_objects = np.mean([h['total_objects'] for h in self.detection_history])
            max_objects = max([h['total_objects'] for h in self.detection_history])
            
            print(f"\n📊 Detection Summary:")
            print(f"• Total frames processed: {len(self.detection_history)}")
            print(f"• Average objects per frame: {avg_objects:.1f}")
            print(f"• Maximum objects detected: {max_objects}")
            print(f"• Models used: {', '.join(self.models.keys())}")

def main():
    """Main function to run enhanced object detection"""
    detector = EnhancedObjectDetector()
    detector.run_enhanced_detection()

if __name__ == "__main__":
    main()
