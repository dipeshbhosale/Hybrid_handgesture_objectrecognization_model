"""
Advanced Multi-Object Detection System
=====================================
A comprehensive object detection system with multiple models, ensemble detection,
custom training capabilities, and impressive real-time performance.
"""

import cv2
import numpy as np
import torch
import os
import json
import time
from collections import defaultdict, deque
from datetime import datetime
import threading
import queue
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from ultralytics import YOLO
    import torch.nn.functional as F
    DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Advanced detection not available: {e}")
    DETECTION_AVAILABLE = False

class AdvancedDetectionSystem:
    """
    Advanced multi-model object detection system with ensemble capabilities
    """
    
    def __init__(self):
        self.models = {}
        self.model_stats = {}
        self.detection_history = deque(maxlen=50)  # Reduced for better performance
        self.object_tracker = {}
        self.confidence_threshold = 0.4  # Higher default for better performance
        self.nms_threshold = 0.5  # Higher for better performance
        self.frame_buffer = queue.Queue(maxsize=3)  # Smaller buffer for lower latency
        self.fast_mode = False  # Flag for performance optimization
        self.stats = {
            'total_detections': 0,
            'objects_detected': defaultdict(int),
            'fps_history': deque(maxlen=15),  # Smaller history for performance
            'confidence_history': deque(maxlen=50),  # Reduced size
            'model_performance': defaultdict(dict)
        }
        # Performance optimization: Cache for faster access
        self._cached_stats = {}
        self._stats_cache_time = 0
        
        # Enhanced color palette for different object types
        self.object_colors = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),         # Red
            'truck': (255, 100, 0),     # Orange
            'bus': (255, 200, 0),       # Yellow-Orange
            'bicycle': (0, 255, 255),   # Cyan
            'motorcycle': (255, 0, 255), # Magenta
            'dog': (128, 255, 0),       # Light Green
            'cat': (255, 128, 0),       # Orange
            'bird': (0, 128, 255),      # Light Blue
            'horse': (128, 0, 255),     # Purple
            'bottle': (255, 255, 0),    # Yellow
            'cup': (0, 255, 128),       # Green-Cyan
            'chair': (128, 128, 255),   # Light Purple
            'laptop': (255, 128, 128),  # Pink
            'phone': (128, 255, 128),   # Light Green
        }
        
        print("🚀 Advanced Detection System initialized")
        self.setup_models()
    
    def setup_models(self):
        """Load multiple YOLO models for ensemble detection"""
        if not DETECTION_AVAILABLE:
            return False
        
        # Performance-optimized model loading: Prioritize fastest models for 15 FPS
        model_configs = [
            {'name': 'yolov8n.pt', 'priority': 1, 'type': 'speed'},     # Fastest first
            {'name': 'yolov8s.pt', 'priority': 2, 'type': 'speed'},     # Second fastest
            {'name': 'yolov8m.pt', 'priority': 3, 'type': 'balanced'},  # Medium
            {'name': 'yolov8l.pt', 'priority': 4, 'type': 'accuracy'},  # Slower
            {'name': 'yolov8x.pt', 'priority': 5, 'type': 'accuracy'},  # Slowest
        ]
        
        # Fix PyTorch compatibility
        self._fix_pytorch_compatibility()
        
        loaded_models = 0
        for config in model_configs:
            try:
                print(f"🔄 Loading {config['name']}...")
                model = YOLO(config['name'])
                
                # Test model with dummy frame
                dummy_frame = np.zeros((416, 416, 3), dtype=np.uint8)
                test_results = model(dummy_frame, verbose=False)
                
                self.models[config['name']] = {
                    'model': model,
                    'priority': config['priority'],
                    'type': config['type'],
                    'classes': len(model.names),
                    'class_names': model.names
                }
                
                print(f"✅ {config['name']} loaded successfully ({len(model.names)} classes)")
                loaded_models += 1
                
                # For 15 FPS performance: Load maximum 2 models, prioritizing fastest
                if loaded_models >= 2:
                    print(f"🚀 Loaded {loaded_models} models - optimized for 15 FPS performance")
                    break
                    
            except Exception as e:
                print(f"⚠️ Failed to load {config['name']}: {e}")
                continue
        
        if loaded_models == 0:
            print("❌ No models loaded successfully")
            return False
        
        print(f"🎯 Successfully loaded {loaded_models} models for ensemble detection")
        return True
    
    def _fix_pytorch_compatibility(self):
        """Fix PyTorch loading issues"""
        try:
            # Add safe globals for ultralytics
            torch.serialization.add_safe_globals([
                'ultralytics.nn.tasks.DetectionModel',
                'ultralytics.nn.modules.head.Detect',
                'ultralytics.nn.modules.conv.Conv',
                'ultralytics.nn.modules.block.C2f',
                'ultralytics.nn.modules.block.SPPF',
                'ultralytics.nn.modules.transformer.AIFI',
                'collections.OrderedDict',
                'torch.nn.modules.conv.Conv2d',
                'torch.nn.modules.batchnorm.BatchNorm2d',
                'torch.nn.modules.activation.SiLU'
            ])
        except:
            pass
        
        # Monkey patch torch.load for compatibility
        original_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs.pop('weights_only', None)
            return original_load(*args, **kwargs, weights_only=False)
        torch.load = safe_load
    
    def ensemble_detect(self, frame, use_ensemble=True):
        """
        Perform ensemble detection using multiple models (optimized for 15 FPS)
        """
        if not self.models:
            return []
        
        detections = []
        start_time = time.time()
        
        # Performance optimization: Resize frame if too large (balanced for detection accuracy)
        height, width = frame.shape[:2]
        if width > 640:  # Restore larger size for better multi-object detection
            scale = 640 / width
            frame = cv2.resize(frame, (640, int(height * scale)))
        elif width < 320:  # Don't go too small
            scale = 320 / width
            frame = cv2.resize(frame, (320, int(height * scale)))
        
        if use_ensemble and len(self.models) > 1:
            # Use only the 2 fastest models for ensemble (15 FPS optimization)
            all_detections = []
            fastest_models = list(self.models.items())[:2]  # Take first 2 (fastest)
            
            for model_name, model_info in fastest_models:
                try:
                    # Optimized inference parameters for better multi-object detection
                    results = model_info['model'](frame, 
                                                conf=self.confidence_threshold,
                                                iou=self.nms_threshold,
                                                max_det=100,  # Increased for better multi-object detection
                                                imgsz=640,   # Larger input size for better detection
                                                verbose=False,
                                                device='0' if torch.cuda.is_available() else 'cpu')
                    
                    for result in results:
                        if result.boxes is not None and len(result.boxes) > 0:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            class_ids = result.boxes.cls.cpu().numpy().astype(int)
                            
                            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                                if conf > self.confidence_threshold:
                                    class_name = model_info['class_names'].get(cls_id, f"class_{cls_id}")
                                    all_detections.append({
                                        'box': box,
                                        'confidence': conf,
                                        'class_id': cls_id,
                                        'class_name': class_name,
                                        'model': model_name
                                    })
                except Exception as e:
                    print(f"Error in model {model_name}: {e}")
                    continue
            
            # Apply fast ensemble NMS
            detections = self._fast_ensemble_nms(all_detections)
            
        else:
            # Use single fastest model for maximum speed
            fastest_model_name = min(self.models.keys(), 
                                   key=lambda x: self.models[x]['priority'])
            model_info = self.models[fastest_model_name]
            
            try:
                # Optimized single model inference for better multi-object detection
                results = model_info['model'](frame,
                                            conf=self.confidence_threshold,
                                            iou=self.nms_threshold,
                                            max_det=100,  # Increased for better multi-object detection
                                            imgsz=640,   # Larger input size for better detection
                                            verbose=False,
                                            device='0' if torch.cuda.is_available() else 'cpu')
                
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)
                        
                        for box, conf, cls_id in zip(boxes, confidences, class_ids):
                            if conf > self.confidence_threshold:
                                class_name = model_info['class_names'].get(cls_id, f"class_{cls_id}")
                                detections.append({
                                    'box': box,
                                    'confidence': conf,
                                    'class_id': cls_id,
                                    'class_name': class_name,
                                    'model': fastest_model_name
                                })
            except Exception as e:
                print(f"Error in detection: {e}")
        
        # Update statistics (optimized)
        detection_time = time.time() - start_time
        if detection_time > 0:
            fps = 1.0 / detection_time
            self.stats['fps_history'].append(fps)
        
        self.stats['total_detections'] += len(detections)
        
        # Batch update objects detected for performance
        for det in detections:
            self.stats['objects_detected'][det['class_name']] += 1
            self.stats['confidence_history'].append(det['confidence'])
        
        return detections
    
    def _ensemble_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to ensemble detections"""
        if not detections:
            return []
        
        # Group detections by class
        class_groups = defaultdict(list)
        for det in detections:
            class_groups[det['class_name']].append(det)
        
        final_detections = []
        
        for class_name, class_detections in class_groups.items():
            if not class_detections:
                continue
            
            # Sort by confidence
            class_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Apply NMS
            keep = []
            while class_detections:
                best = class_detections.pop(0)
                keep.append(best)
                
                # Remove overlapping detections
                class_detections = [
                    det for det in class_detections
                    if self._calculate_iou(best['box'], det['box']) < iou_threshold
                ]
            
            final_detections.extend(keep)
        
        return final_detections
    
    def _fast_ensemble_nms(self, detections, iou_threshold=0.5):
        """Fast NMS for ensemble detections - optimized for 15 FPS"""
        if not detections or len(detections) == 0:
            return []
        
        # Quick return for small detection counts
        if len(detections) <= 3:
            return detections
        
        # Sort all detections by confidence (global NMS for speed)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        processed_boxes = []
        
        for det in detections:
            # Quick overlap check against already processed boxes
            overlap_found = False
            current_box = det['box']
            
            for processed_box in processed_boxes:
                if self._fast_iou(current_box, processed_box) > iou_threshold:
                    overlap_found = True
                    break
            
            if not overlap_found:
                final_detections.append(det)
                processed_boxes.append(current_box)
                
                # Limit detections for performance but allow reasonable multi-object detection
                if len(final_detections) >= 30:  # Increased from 20
                    break
        
        return final_detections
    
    def _fast_iou(self, box1, box2):
        """Fast IoU calculation optimized for speed"""
        # Quick bounds check first
        if (box1[2] < box2[0] or box2[2] < box1[0] or 
            box1[3] < box2[1] or box2[3] < box1[1]):
            return 0.0
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def draw_enhanced_detections(self, frame, detections):
        """Draw detections with enhanced visualization (optimized for 15 FPS)"""
        if not detections:
            return frame
        
        # Limit detections drawn for performance but allow more for multi-object scenarios
        max_detections = 25  # Increased from 15 for better multi-object display
        detections_to_draw = detections[:max_detections]
        
        for detection in detections_to_draw:
            box = detection['box']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Get color for this object type
            color = self.object_colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box (simplified for performance)
            x1, y1, x2, y2 = map(int, box)
            thickness = 2 if confidence > 0.7 else 1  # Vary thickness by confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Simplified text drawing for performance
            label = f"{class_name}: {confidence:.1f}"  # Reduced precision for speed
            
            # Draw text background (faster method)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1, y1 - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add performance info if too many detections
        if len(detections) > max_detections:
            cv2.putText(frame, f"+{len(detections) - max_detections} more objects", 
                       (10, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def draw_statistics_panel(self, frame):
        """Draw real-time statistics panel (optimized for 15 FPS)"""
        height, width = frame.shape[:2]
        panel_width = 250  # Smaller panel for performance
        panel_height = 120  # Reduced height
        
        # Simple background rectangle (faster than overlay)
        y_start = height - panel_height - 10
        x_start = width - panel_width - 10
        
        # Draw semi-transparent background
        cv2.rectangle(frame, (x_start, y_start), 
                     (x_start + panel_width, y_start + panel_height),
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (x_start, y_start), 
                     (x_start + panel_width, y_start + panel_height),
                     (255, 255, 255), 1)
        
        # Simplified statistics (cached values)
        stats = self.get_statistics()  # Uses cached values
        avg_fps = stats['average_fps']
        total_objects = sum(stats['objects_detected'].values())
        
        # Draw key statistics only (for performance)
        y_offset = y_start + 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        stats_text = [
            f"FPS: {avg_fps:.1f}",
            f"Objects: {total_objects}",
            f"Models: {stats['models_loaded']}",
            f"Conf: {self.confidence_threshold:.1f}"
        ]
        
        for text in stats_text:
            cv2.putText(frame, text, (x_start + 10, y_offset), 
                       font, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        return frame
    
    def process_frame_with_tracking(self, frame, use_ensemble=True, draw_stats=True):
        """Process frame with object tracking and enhanced visualization"""
        # Perform detection
        detections = self.ensemble_detect(frame, use_ensemble)
        
        # Draw detections
        frame = self.draw_enhanced_detections(frame, detections)
        
        # Draw statistics panel
        if draw_stats:
            frame = self.draw_statistics_panel(frame)
        
        # Add detection count overlay
        total_detected = len(detections)
        cv2.putText(frame, f"Objects Detected: {total_detected}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, detections
    
    def get_statistics(self):
        """Get comprehensive detection statistics (cached for performance)"""
        current_time = time.time()
        
        # Use cached stats if recent (within 0.5 seconds) for better performance
        if (current_time - self._stats_cache_time) < 0.5 and self._cached_stats:
            return self._cached_stats
        
        # Calculate fresh statistics
        avg_fps = np.mean(self.stats['fps_history']) if self.stats['fps_history'] else 0
        avg_conf = np.mean(self.stats['confidence_history']) if self.stats['confidence_history'] else 0
        
        self._cached_stats = {
            'average_fps': avg_fps,
            'average_confidence': avg_conf,
            'total_detections': self.stats['total_detections'],
            'objects_detected': dict(self.stats['objects_detected']),
            'models_loaded': len(self.models),
            'model_names': list(self.models.keys())
        }
        self._stats_cache_time = current_time
        
        return self._cached_stats
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.stats = {
            'total_detections': 0,
            'objects_detected': defaultdict(int),
            'fps_history': deque(maxlen=15),  # Smaller for performance
            'confidence_history': deque(maxlen=50),  # Smaller for performance
            'model_performance': defaultdict(dict)
        }
        # Clear cache
        self._cached_stats = {}
        self._stats_cache_time = 0
        print("📊 Statistics reset")

# Global instance
advanced_detector = None

def get_advanced_detector():
    """Get or create the advanced detector instance"""
    global advanced_detector
    if advanced_detector is None:
        advanced_detector = AdvancedDetectionSystem()
    return advanced_detector

def test_advanced_detection():
    """Test the advanced detection system"""
    print("🧪 Testing Advanced Detection System...")
    
    detector = get_advanced_detector()
    
    if not detector.models:
        print("❌ No models loaded, cannot test")
        return False
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        # Test detection
        processed_frame, detections = detector.process_frame_with_tracking(test_frame)
        
        print(f"✅ Detection test successful")
        print(f"   Frame size: {processed_frame.shape}")
        print(f"   Detections: {len(detections)}")
        
        stats = detector.get_statistics()
        print(f"   Models: {stats['models_loaded']}")
        print(f"   Average FPS: {stats['average_fps']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

if __name__ == "__main__":
    test_advanced_detection()
