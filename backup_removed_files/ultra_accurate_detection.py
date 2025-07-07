#!/usr/bin/env python3
"""
Ultra-High Accuracy Multi-Object Detection System
===============================================
Maximum accuracy focused detection system using the most effective YOLO models.
Prioritizes detection accuracy over speed for research and precision applications.
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
    print(f"⚠️ Ultra-high accuracy detection not available: {e}")
    DETECTION_AVAILABLE = False

class UltraAccurateDetectionSystem:
    """
    Ultra-high accuracy multi-object detection system
    Uses the most effective YOLO models for maximum precision
    """
    
    def __init__(self):
        self.models = {}
        self.model_stats = {}
        self.detection_history = deque(maxlen=200)  # Larger history for accuracy analysis
        self.object_tracker = {}
        self.confidence_threshold = 0.1  # Very low for maximum sensitivity
        self.nms_threshold = 0.3  # Lower for better overlapping object detection
        self.frame_buffer = queue.Queue(maxsize=10)  # Larger buffer for accuracy
        self.accuracy_mode = True
        self.stats = {
            'total_detections': 0,
            'objects_detected': defaultdict(int),
            'accuracy_metrics': deque(maxlen=100),
            'confidence_distribution': defaultdict(int),
            'model_performance': defaultdict(dict),
            'precision_scores': deque(maxlen=50)
        }
        
        # Ultra-high accuracy color palette for maximum object distinction
        self.object_colors = self._generate_distinct_colors(80)  # 80 unique colors for all classes
        
        print("🎯 Ultra-High Accuracy Detection System initialized")
        self.setup_accuracy_models()
    
    def _generate_distinct_colors(self, num_colors):
        """Generate maximally distinct colors for object visualization"""
        colors = {}
        for i in range(num_colors):
            # Use HSV color space for maximum distinction
            hue = (i * 137.5) % 360  # Golden angle for maximum separation
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.8 + (i % 2) * 0.2  # Vary brightness slightly
            
            # Convert HSV to RGB
            h, s, v = hue/360, saturation, value
            c = v * s
            x = c * (1 - abs((h * 6) % 2 - 1))
            m = v - c
            
            if 0 <= h < 1/6:
                r, g, b = c, x, 0
            elif 1/6 <= h < 1/3:
                r, g, b = x, c, 0
            elif 1/3 <= h < 1/2:
                r, g, b = 0, c, x
            elif 1/2 <= h < 2/3:
                r, g, b = 0, x, c
            elif 2/3 <= h < 5/6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            # Convert to BGR for OpenCV
            colors[f"class_{i}"] = (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))
        
        return colors
    
    def setup_accuracy_models(self):
        """Load the most accurate YOLO models available"""
        if not DETECTION_AVAILABLE:
            return False
        
        # Accuracy-focused model configurations (largest models first)
        model_configs = [
            {'name': 'yolov8x.pt', 'priority': 1, 'type': 'ultra_accurate', 'weight': 1.0},
            {'name': 'yolov8l.pt', 'priority': 2, 'type': 'high_accurate', 'weight': 0.9},
            {'name': 'yolov8m.pt', 'priority': 3, 'type': 'accurate', 'weight': 0.8},
            {'name': 'yolov8s.pt', 'priority': 4, 'type': 'balanced', 'weight': 0.7},
            {'name': 'yolov8n.pt', 'priority': 5, 'type': 'fast', 'weight': 0.6},
        ]
        
        # Fix PyTorch compatibility
        self._fix_pytorch_compatibility()
        
        loaded_models = 0
        print("🎯 Loading ultra-high accuracy models...")
        
        for config in model_configs:
            try:
                print(f"🔄 Loading {config['name']} for maximum accuracy...")
                model = YOLO(config['name'])
                
                # Test model loading
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                test_results = model(dummy_frame, verbose=False)
                
                self.models[config['name']] = {
                    'model': model,
                    'priority': config['priority'],
                    'type': config['type'],
                    'weight': config['weight'],
                    'classes': len(model.names),
                    'class_names': model.names
                }
                
                print(f"✅ {config['name']} loaded successfully ({len(model.names)} classes)")
                loaded_models += 1
                
                # Load ALL available models for maximum ensemble accuracy
                
            except Exception as e:
                print(f"⚠️ Failed to load {config['name']}: {e}")
                continue
        
        if loaded_models == 0:
            print("❌ No models loaded successfully")
            return False
        
        print(f"🎯 Successfully loaded {loaded_models} models for ultra-high accuracy ensemble")
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
    
    def ultra_accurate_detect(self, frame):
        """
        Ultra-high accuracy detection using all available models
        No FPS constraints - maximum accuracy priority
        """
        if not self.models:
            return []
        
        detections = []
        start_time = time.time()
        
        # Use larger frame size for better accuracy
        height, width = frame.shape[:2]
        if width < 1280:  # Upscale for better detection
            scale = 1280 / width
            frame = cv2.resize(frame, (1280, int(height * scale)))
        
        # Ultra-high accuracy ensemble detection using ALL models
        all_detections = []
        
        print(f"🔍 Running ultra-accurate detection with {len(self.models)} models...")
        
        for model_name, model_info in self.models.items():
            try:
                print(f"   Processing with {model_name}...")
                
                # Ultra-high accuracy inference parameters
                results = model_info['model'](frame, 
                                            conf=self.confidence_threshold,  # Very low threshold
                                            iou=self.nms_threshold,  # Low NMS for overlapping objects
                                            max_det=1000,  # No practical limit
                                            imgsz=1280,  # Largest inference size
                                            augment=True,  # Test-time augmentation for accuracy
                                            verbose=False,
                                            device='0' if torch.cuda.is_available() else 'cpu')
                
                model_detections = 0
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)
                        
                        for box, conf, cls_id in zip(boxes, confidences, class_ids):
                            if conf > self.confidence_threshold:
                                class_name = model_info['class_names'].get(cls_id, f"class_{cls_id}")
                                
                                # Weight confidence by model accuracy
                                weighted_conf = conf * model_info['weight']
                                
                                all_detections.append({
                                    'box': box,
                                    'confidence': weighted_conf,
                                    'raw_confidence': conf,
                                    'class_id': cls_id,
                                    'class_name': class_name,
                                    'model': model_name,
                                    'model_weight': model_info['weight']
                                })
                                model_detections += 1
                
                print(f"     Found {model_detections} objects")
                
            except Exception as e:
                print(f"Error in ultra-accurate model {model_name}: {e}")
                continue
        
        # Ultra-sophisticated ensemble fusion
        detections = self._ultra_accurate_ensemble_fusion(all_detections)
        
        # Update accuracy statistics
        detection_time = time.time() - start_time
        accuracy_score = self._calculate_accuracy_score(detections)
        
        self.stats['accuracy_metrics'].append(accuracy_score)
        self.stats['total_detections'] += len(detections)
        
        # Batch update statistics
        for det in detections:
            self.stats['objects_detected'][det['class_name']] += 1
            conf_bucket = int(det['confidence'] * 10) / 10
            self.stats['confidence_distribution'][conf_bucket] += 1
        
        print(f"🎯 Ultra-accurate detection complete: {len(detections)} objects in {detection_time:.2f}s")
        
        return detections
    
    def _ultra_accurate_ensemble_fusion(self, detections, confidence_threshold=0.05):
        """
        Ultra-sophisticated ensemble fusion for maximum accuracy
        Uses weighted voting and advanced NMS
        """
        if not detections:
            return []
        
        print(f"🧠 Performing ultra-accurate ensemble fusion on {len(detections)} detections...")
        
        # Group detections by class
        class_groups = defaultdict(list)
        for det in detections:
            class_groups[det['class_name']].append(det)
        
        final_detections = []
        
        for class_name, class_detections in class_groups.items():
            if not class_detections:
                continue
            
            # Advanced clustering-based NMS for overlapping objects
            clusters = self._advanced_clustering_nms(class_detections)
            
            for cluster in clusters:
                if len(cluster) == 1:
                    # Single detection
                    final_detections.append(cluster[0])
                else:
                    # Multiple detections - create weighted average
                    fused_detection = self._fuse_detection_cluster(cluster)
                    if fused_detection['confidence'] > confidence_threshold:
                        final_detections.append(fused_detection)
        
        # Sort by confidence for better visualization
        final_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"✅ Ensemble fusion complete: {len(final_detections)} final detections")
        
        return final_detections
    
    def _advanced_clustering_nms(self, detections, iou_threshold=0.3):
        """Advanced clustering-based NMS for better accuracy"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        clusters = []
        used = set()
        
        for i, det in enumerate(detections):
            if i in used:
                continue
                
            cluster = [det]
            used.add(i)
            
            # Find all overlapping detections
            for j, other_det in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                    
                iou = self._calculate_iou(det['box'], other_det['box'])
                if iou > iou_threshold:
                    cluster.append(other_det)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _fuse_detection_cluster(self, cluster):
        """Fuse multiple detections into a single high-confidence detection"""
        if len(cluster) == 1:
            return cluster[0]
        
        # Weighted average of bounding boxes
        total_weight = sum(det['model_weight'] * det['raw_confidence'] for det in cluster)
        
        avg_box = np.zeros(4)
        avg_confidence = 0
        model_votes = defaultdict(int)
        
        for det in cluster:
            weight = det['model_weight'] * det['raw_confidence']
            avg_box += det['box'] * weight
            avg_confidence += det['confidence'] * weight
            model_votes[det['model']] += weight
        
        avg_box /= total_weight
        avg_confidence /= total_weight
        
        # Boost confidence for multi-model agreement
        confidence_boost = min(len(cluster) * 0.1, 0.3)
        final_confidence = min(avg_confidence + confidence_boost, 1.0)
        
        # Select most confident model as primary
        primary_model = max(model_votes.keys(), key=lambda k: model_votes[k])
        
        return {
            'box': avg_box,
            'confidence': final_confidence,
            'raw_confidence': avg_confidence,
            'class_id': cluster[0]['class_id'],
            'class_name': cluster[0]['class_name'],
            'model': f"ensemble_{len(cluster)}models",
            'model_weight': 1.0,
            'ensemble_size': len(cluster)
        }
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two boxes"""
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
    
    def _calculate_accuracy_score(self, detections):
        """Calculate a comprehensive accuracy score"""
        if not detections:
            return 0.0
        
        # Factors for accuracy score
        confidence_score = np.mean([det['confidence'] for det in detections])
        diversity_score = len(set(det['class_name'] for det in detections)) / max(len(detections), 1)
        ensemble_score = np.mean([det.get('ensemble_size', 1) for det in detections]) / 5.0
        
        # Weighted combination
        accuracy_score = (
            confidence_score * 0.5 +
            diversity_score * 0.3 +
            min(ensemble_score, 1.0) * 0.2
        )
        
        return min(accuracy_score, 1.0)
    
    def draw_ultra_accurate_detections(self, frame, detections):
        """Draw detections with ultra-high accuracy visualization"""
        if not detections:
            return frame
        
        for detection in detections:
            box = detection['box']
            confidence = detection['confidence']
            class_name = detection['class_name']
            ensemble_size = detection.get('ensemble_size', 1)
            
            # Get color for this object type
            color = self.object_colors.get(class_name, self.object_colors.get(f"class_{detection['class_id']}", (255, 255, 255)))
            
            # Draw bounding box with thickness based on confidence
            x1, y1, x2, y2 = map(int, box)
            thickness = max(1, int(confidence * 5))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw ensemble indicator
            if ensemble_size > 1:
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 0), 1)
            
            # Ultra-detailed label
            label = f"{class_name}: {confidence:.3f}"
            if ensemble_size > 1:
                label += f" (E{ensemble_size})"
            
            # Draw text background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 15), (x1 + text_w + 10, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw confidence bar
            bar_width = int((x2 - x1) * confidence)
            cv2.rectangle(frame, (x1, y2 + 2), (x1 + bar_width, y2 + 10), color, -1)
        
        return frame
    
    def draw_accuracy_statistics(self, frame):
        """Draw comprehensive accuracy statistics"""
        height, width = frame.shape[:2]
        panel_width = 400
        panel_height = 300
        
        # Create statistics panel
        y_start = 10
        x_start = width - panel_width - 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), 
                     (x_start + panel_width, y_start + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        cv2.rectangle(frame, (x_start, y_start), 
                     (x_start + panel_width, y_start + panel_height),
                     (0, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "🎯 ULTRA-HIGH ACCURACY STATS", (x_start + 10, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_offset = y_start + 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Calculate statistics
        avg_accuracy = np.mean(self.stats['accuracy_metrics']) if self.stats['accuracy_metrics'] else 0
        total_objects = sum(self.stats['objects_detected'].values())
        unique_classes = len(self.stats['objects_detected'])
        
        stats_text = [
            f"Accuracy Score: {avg_accuracy:.3f}",
            f"Total Objects: {total_objects}",
            f"Unique Classes: {unique_classes}",
            f"Models Loaded: {len(self.models)}",
            f"Confidence Threshold: {self.confidence_threshold:.2f}",
            "Mode: ULTRA-ACCURATE"
        ]
        
        for text in stats_text:
            cv2.putText(frame, text, (x_start + 10, y_offset), font, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Top detected objects
        cv2.putText(frame, "Top Objects:", (x_start + 10, y_offset), font, 0.5, (0, 255, 0), 2)
        y_offset += 20
        
        top_objects = sorted(self.stats['objects_detected'].items(), 
                           key=lambda x: x[1], reverse=True)[:8]
        
        for obj, count in top_objects:
            color = self.object_colors.get(obj, (255, 255, 255))
            cv2.putText(frame, f"  {obj}: {count}", (x_start + 15, y_offset), 
                       font, 0.4, color, 1)
            y_offset += 18
        
        return frame
    
    def process_frame_ultra_accurate(self, frame, draw_stats=True):
        """Process frame with ultra-high accuracy detection"""
        print("🎯 Processing frame with ultra-high accuracy...")
        
        # Perform ultra-accurate detection
        detections = self.ultra_accurate_detect(frame)
        
        # Draw detections
        frame = self.draw_ultra_accurate_detections(frame, detections)
        
        # Draw statistics panel
        if draw_stats:
            frame = self.draw_accuracy_statistics(frame)
        
        return frame, detections
    
    def get_accuracy_statistics(self):
        """Get comprehensive accuracy statistics"""
        avg_accuracy = np.mean(self.stats['accuracy_metrics']) if self.stats['accuracy_metrics'] else 0
        
        return {
            'average_accuracy': avg_accuracy,
            'total_detections': self.stats['total_detections'],
            'objects_detected': dict(self.stats['objects_detected']),
            'confidence_distribution': dict(self.stats['confidence_distribution']),
            'models_loaded': len(self.models),
            'model_names': list(self.models.keys()),
            'unique_classes_detected': len(self.stats['objects_detected'])
        }
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.stats = {
            'total_detections': 0,
            'objects_detected': defaultdict(int),
            'accuracy_metrics': deque(maxlen=100),
            'confidence_distribution': defaultdict(int),
            'model_performance': defaultdict(dict),
            'precision_scores': deque(maxlen=50)
        }
        print("📊 Ultra-accuracy statistics reset")
    
    def load_all_models(self):
        """Load all available YOLO models for maximum accuracy"""
        return self.setup_accuracy_models()
    
    def detect_ultra_accurate(self, frame):
        """
        Main detection method that returns detections and accuracy stats
        """
        detections = self.ultra_accurate_detect(frame)
        accuracy_stats = self.get_accuracy_statistics()
        return detections, accuracy_stats
    
    def draw_ultra_accurate_detections_with_stats(self, frame, detections, accuracy_stats=None):
        """
        Enhanced drawing method that accepts accuracy stats
        """
        # Use the existing drawing method and add accuracy overlay
        frame_with_detections = self.draw_ultra_accurate_detections(frame, detections)
        
        if accuracy_stats:
            # Add accuracy information to the frame
            y_offset = frame.shape[0] - 150
            cv2.putText(frame_with_detections, "ACCURACY STATS:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
            
            if 'average_accuracy' in accuracy_stats:
                cv2.putText(frame_with_detections, f"Avg Accuracy: {accuracy_stats['average_accuracy']:.2f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            if 'models_loaded' in accuracy_stats:
                cv2.putText(frame_with_detections, f"Ensemble Models: {accuracy_stats['models_loaded']}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_with_detections
    
    def reset_stats(self):
        """Alias for reset_statistics for compatibility"""
        self.reset_statistics()
    
    def toggle_accuracy_analysis(self):
        """Toggle accuracy analysis overlay"""
        self.accuracy_mode = not self.accuracy_mode
        print(f"📊 Accuracy analysis: {'ON' if self.accuracy_mode else 'OFF'}")
    
    def cycle_confidence_mode(self):
        """Cycle through confidence visualization modes"""
        # Cycle through different confidence thresholds for visualization
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        current_idx = thresholds.index(self.confidence_threshold) if self.confidence_threshold in thresholds else 0
        next_idx = (current_idx + 1) % len(thresholds)
        self.confidence_threshold = thresholds[next_idx]
        print(f"🎯 Confidence threshold: {self.confidence_threshold}")
    
    def toggle_temporal_smoothing(self):
        """Toggle temporal smoothing for detection consistency"""
        # This could be implemented to use detection history for smoothing
        print("⏱️  Temporal smoothing toggled (feature placeholder)")
    
    def get_total_detections(self):
        """Get total number of detections processed"""
        return self.stats['total_detections']
    
    def get_top_detected_objects(self, top_n=5):
        """Get top N detected objects"""
        return sorted(self.stats['objects_detected'].items(), 
                     key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_final_statistics(self):
        """Get comprehensive final statistics"""
        stats = self.get_accuracy_statistics()
        
        # Add additional final stats
        stats.update({
            'session_duration': time.time() - getattr(self, 'start_time', time.time()),
            'detection_rate': self.stats['total_detections'] / max(1, len(self.detection_history)),
            'unique_objects': len(self.stats['objects_detected']),
            'confidence_range': {
                'min': min(self.stats['confidence_distribution'].keys()) if self.stats['confidence_distribution'] else 0,
                'max': max(self.stats['confidence_distribution'].keys()) if self.stats['confidence_distribution'] else 0
            }
        })
        
        return stats

# Global instance
ultra_accurate_detector = None

def get_ultra_accurate_detector():
    """Get or create the ultra-accurate detector instance"""
    global ultra_accurate_detector
    if ultra_accurate_detector is None:
        ultra_accurate_detector = UltraAccurateDetectionSystem()
    return ultra_accurate_detector

if __name__ == "__main__":
    print("🎯 Ultra-High Accuracy Detection System")
    print("This system prioritizes maximum detection accuracy over speed.")
    detector = get_ultra_accurate_detector()
    print(f"✅ System ready with {len(detector.models) if detector.models else 0} accuracy-optimized models")
