#!/usr/bin/env python3
"""
Performance optimization script for the advanced detection system
This will modify the model loading to prioritize speed over accuracy for 15 FPS
"""

import os
import sys
sys.path.append('.')

def create_performance_optimized_detection():
    """Create a performance-optimized version of the advanced detection system"""
    
    print("🔧 Creating Performance-Optimized Advanced Detection System")
    print("="*60)
    
    # Create an optimized version of advanced_detection_system.py
    optimized_code = '''"""
Performance-Optimized Advanced Multi-Object Detection System
==========================================================
Optimized for 15 FPS real-time performance while maintaining accuracy
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

class PerformanceOptimizedDetectionSystem:
    """
    Performance-optimized multi-model object detection system for 15 FPS
    """
    
    def __init__(self):
        self.models = {}
        self.model_stats = {}
        self.detection_history = deque(maxlen=50)  # Reduced for performance
        self.object_tracker = {}
        self.confidence_threshold = 0.4  # Higher for better performance
        self.nms_threshold = 0.5
        self.frame_buffer = queue.Queue(maxsize=3)  # Smaller buffer
        self.stats = {
            'total_detections': 0,
            'objects_detected': defaultdict(int),
            'fps_history': deque(maxlen=20),  # Smaller for performance
            'confidence_history': deque(maxlen=50),  # Smaller for performance
            'model_performance': defaultdict(dict)
        }
        
        # Performance-optimized color palette
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
            'bottle': (255, 255, 0),    # Yellow
            'cup': (0, 255, 128),       # Green-Cyan
            'chair': (128, 128, 255),   # Light Purple
            'laptop': (255, 128, 128),  # Pink
            'phone': (128, 255, 128),   # Light Green
        }
        
        print("🚀 Performance-Optimized Detection System initialized")
        self.setup_models()
    
    def setup_models(self):
        """Load models prioritizing speed for 15 FPS performance"""
        if not DETECTION_AVAILABLE:
            return False
        
        # Performance-optimized model priority (fastest first)
        model_configs = [
            {'name': 'yolov8n.pt', 'priority': 1, 'type': 'speed'},      # Fastest
            {'name': 'yolov8s.pt', 'priority': 2, 'type': 'speed'},      # Fast
            {'name': 'yolov8m.pt', 'priority': 3, 'type': 'balanced'},   # Moderate
            {'name': 'yolov8l.pt', 'priority': 4, 'type': 'accuracy'},   # Slower
            {'name': 'yolov8x.pt', 'priority': 5, 'type': 'accuracy'},   # Slowest
        ]
        
        # Fix PyTorch compatibility
        self._fix_pytorch_compatibility()
        
        loaded_models = 0
        for config in model_configs:
            try:
                print(f"🔄 Loading {config['name']} (Priority: {config['type']})...")
                model = YOLO(config['name'])
                
                # Test model with smaller dummy frame for speed
                dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
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
                
                # For 15 FPS, prioritize loading fast models first
                if loaded_models >= 1 and config['type'] == 'speed':
                    print(f"🚀 Fast model loaded - ready for 15 FPS performance")
                    break  # Use only the fastest model for optimal performance
                    
            except Exception as e:
                print(f"⚠️ Failed to load {config['name']}: {e}")
                continue
        
        if loaded_models == 0:
            print("❌ No models loaded successfully")
            return False
        
        print(f"🎯 Successfully loaded {loaded_models} model(s) optimized for 15 FPS")
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
    
    def fast_detect(self, frame, use_ensemble=False):
        """
        Fast detection optimized for 15 FPS
        """
        if not self.models:
            return []
        
        detections = []
        start_time = time.time()
        
        # Use single fastest model for 15 FPS (ignore ensemble for performance)
        fastest_model_name = min(self.models.keys(), 
                                key=lambda x: self.models[x]['priority'])
        model_info = self.models[fastest_model_name]
        
        try:
            # Optimized inference parameters for speed
            results = model_info['model'](frame,
                                        conf=self.confidence_threshold,
                                        iou=self.nms_threshold,
                                        max_det=50,  # Limit detections for speed
                                        verbose=False,
                                        imgsz=416,  # Smaller image size for speed
                                        device='0' if torch.cuda.is_available() else 'cpu')
            
            for result in results:
                if result.boxes is not None:
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
            print(f"Error in fast detection: {e}")
        
        # Update statistics (simplified for performance)
        detection_time = time.time() - start_time
        fps = 1.0 / detection_time if detection_time > 0 else 0
        self.stats['fps_history'].append(fps)
        self.stats['total_detections'] += len(detections)
        
        for det in detections:
            self.stats['objects_detected'][det['class_name']] += 1
            self.stats['confidence_history'].append(det['confidence'])
        
        return detections
    
    def draw_fast_detections(self, frame, detections):
        """Draw detections with performance-optimized visualization"""
        for detection in detections:
            box = detection['box']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Get color for this object type
            color = self.object_colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box (simplified for speed)
            x1, y1, x2, y2 = map(int, box)
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label (simplified for speed)
            label = f"{class_name}: {confidence:.1f}"
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def draw_fast_statistics(self, frame):
        """Draw simplified statistics panel for performance"""
        height, width = frame.shape[:2]
        
        # Simplified statistics
        avg_fps = np.mean(self.stats['fps_history']) if self.stats['fps_history'] else 0
        total_objects = sum(self.stats['objects_detected'].values())
        
        # Draw simple stats
        stats_text = [
            f"FPS: {avg_fps:.1f}",
            f"Objects: {total_objects}",
            f"Models: {len(self.models)}"
        ]
        
        y_offset = 20
        for text in stats_text:
            cv2.putText(frame, text, (width - 150, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
        
        return frame
    
    def process_frame_with_tracking(self, frame, use_ensemble=False, draw_stats=True):
        """Process frame with optimized tracking for 15 FPS"""
        # Perform fast detection
        detections = self.fast_detect(frame, use_ensemble=False)  # Force single model
        
        # Draw detections
        frame = self.draw_fast_detections(frame, detections)
        
        # Draw statistics
        if draw_stats:
            frame = self.draw_fast_statistics(frame)
        
        # Add detection count overlay
        total_detected = len(detections)
        cv2.putText(frame, f"Detected: {total_detected}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, detections
    
    def get_statistics(self):
        """Get simplified detection statistics"""
        avg_fps = np.mean(self.stats['fps_history']) if self.stats['fps_history'] else 0
        avg_conf = np.mean(self.stats['confidence_history']) if self.stats['confidence_history'] else 0
        
        return {
            'average_fps': avg_fps,
            'average_confidence': avg_conf,
            'total_detections': self.stats['total_detections'],
            'objects_detected': dict(self.stats['objects_detected']),
            'models_loaded': len(self.models),
            'model_names': list(self.models.keys())
        }
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.stats = {
            'total_detections': 0,
            'objects_detected': defaultdict(int),
            'fps_history': deque(maxlen=20),
            'confidence_history': deque(maxlen=50),
            'model_performance': defaultdict(dict)
        }
        print("📊 Statistics reset")

# Global instance for performance
performance_detector = None

def get_performance_detector():
    """Get or create the performance-optimized detector instance"""
    global performance_detector
    if performance_detector is None:
        performance_detector = PerformanceOptimizedDetectionSystem()
    return performance_detector

def test_performance_detection():
    """Test the performance-optimized detection system"""
    print("🧪 Testing Performance-Optimized Detection System...")
    
    detector = get_performance_detector()
    
    if not detector.models:
        print("❌ No models loaded, cannot test")
        return False
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        # Test detection
        start_time = time.time()
        processed_frame, detections = detector.process_frame_with_tracking(test_frame)
        process_time = time.time() - start_time
        
        fps = 1.0 / process_time if process_time > 0 else 0
        
        print(f"✅ Performance test successful")
        print(f"   Frame size: {processed_frame.shape}")
        print(f"   Processing time: {process_time:.3f}s")
        print(f"   FPS: {fps:.1f}")
        print(f"   Detections: {len(detections)}")
        
        stats = detector.get_statistics()
        print(f"   Models: {stats['models_loaded']}")
        
        if fps >= 15:
            print(f"   🎉 TARGET 15 FPS ACHIEVED!")
        elif fps >= 10:
            print(f"   ⚡ Good performance (>10 FPS)")
        else:
            print(f"   💫 Moderate performance")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

if __name__ == "__main__":
    test_performance_detection()
'''
    
    # Write the optimized version
    with open('performance_advanced_detection_system.py', 'w') as f:
        f.write(optimized_code)
    
    print("✅ Created performance_advanced_detection_system.py")
    print("💡 This version prioritizes the fastest YOLO models for 15 FPS")
    
    return True

if __name__ == "__main__":
    create_performance_optimized_detection()
    
    # Test the new performance system
    print("\n🧪 Testing the new performance-optimized system...")
    try:
        exec(open('performance_advanced_detection_system.py').read())
    except Exception as e:
        print(f"❌ Error testing: {e}")
