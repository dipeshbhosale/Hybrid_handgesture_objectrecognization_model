#!/usr/bin/env python3
"""
HYBRID ULTIMATE DETECTION SYSTEM
===============================
Combines ultra-accurate detection with performance optimization for the ultimate AI vision experience.
Features adaptive performance modes, auto-configuration, and seamless multi-modal detection.

Author: AI Vision System
Date: July 3, 2025
Version: 1.0 - Production Ready
"""

import cv2
import numpy as np
import torch
import os
import json
import time
import threading
import queue
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import gesture recognition
import mediapipe as mp
import joblib

# Auto-detection and optimization imports
import psutil
import platform
import GPUtil

try:
    from ultralytics import YOLO
    import torch.nn.functional as F
    DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Detection dependencies not available: {e}")
    DETECTION_AVAILABLE = False

class HybridUltimateDetectionSystem:
    """
    Ultimate detection system that combines:
    - Ultra-accurate object detection (YOLOv8 ensemble)
    - Real-time performance optimization (15+ FPS)
    - Hand gesture recognition (MediaPipe + ML)
    - Auto-configuration and device optimization
    - Production-ready stability and error handling
    """
    
    def __init__(self, auto_configure=True):
        print("🚀 Initializing Hybrid Ultimate Detection System...")
        
        # System specifications
        self.system_specs = self._analyze_system_capabilities()
        self.performance_mode = "auto"  # auto, fast, balanced, ultra_accurate
        
        # Model storage
        self.yolo_models = {}
        self.gesture_model = None
        self.mp_hands = None
        self.mp_drawing = None
        
        # Performance tracking
        self.stats = {
            'total_detections': 0,
            'gesture_detections': 0,
            'objects_detected': defaultdict(int),
            'gestures_detected': defaultdict(int),
            'fps_history': deque(maxlen=100),
            'accuracy_scores': deque(maxlen=50),
            'model_performance': {},
            'session_start': time.time()
        }
        
        # Detection parameters (auto-optimized)
        self.confidence_threshold = 0.25  # Will be auto-adjusted
        self.nms_threshold = 0.45
        self.max_detections = 100  # Will be auto-adjusted
        
        # Performance optimization
        self.detection_history = deque(maxlen=200)
        self.frame_buffer = queue.Queue(maxsize=5)
        self.result_cache = {}
        self.cache_timeout = 0.1  # 100ms cache
        
        # Object colors for visualization
        self.colors = self._generate_color_palette(80)
        
        # Auto-configuration
        if auto_configure:
            self._auto_configure_system()
        
        print(f"✅ System initialized in '{self.performance_mode}' mode")
        print(f"🖥️  Hardware: {self.system_specs['device']} | RAM: {self.system_specs['ram_gb']:.1f}GB")
        
    def _analyze_system_capabilities(self) -> Dict[str, Any]:
        """Analyze system hardware and recommend optimal settings"""
        specs = {
            'cpu_cores': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else 2000,
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'platform': platform.system(),
            'device': 'cpu',
            'gpu_available': False,
            'gpu_memory': 0
        }
        
        # Check for GPU
        try:
            if torch.cuda.is_available():
                specs['device'] = 'cuda'
                specs['gpu_available'] = True
                specs['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)} ({specs['gpu_memory']:.1f}GB)")
            else:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        specs['gpu_available'] = True
                        specs['gpu_memory'] = gpus[0].memoryTotal / 1024
                        print(f"🎮 GPU detected: {gpus[0].name} ({specs['gpu_memory']:.1f}GB)")
                except:
                    pass
        except:
            pass
        
        return specs
    
    def _auto_configure_system(self):
        """Automatically configure the system based on hardware capabilities"""
        ram_gb = self.system_specs['ram_gb']
        cpu_cores = self.system_specs['cpu_cores']
        gpu_available = self.system_specs['gpu_available']
        gpu_memory = self.system_specs['gpu_memory']
        
        print("🔧 Auto-configuring system based on hardware...")
        
        # Determine optimal performance mode
        if gpu_available and gpu_memory >= 4.0 and ram_gb >= 8:
            self.performance_mode = "ultra_accurate"
            self.confidence_threshold = 0.15
            self.max_detections = 300
            print("🏆 Mode: ULTRA_ACCURATE (High-end hardware detected)")
            
        elif gpu_available and gpu_memory >= 2.0 and ram_gb >= 6:
            self.performance_mode = "balanced"
            self.confidence_threshold = 0.25
            self.max_detections = 150
            print("⚖️ Mode: BALANCED (Mid-range hardware detected)")
            
        elif cpu_cores >= 4 and ram_gb >= 4:
            self.performance_mode = "fast"
            self.confidence_threshold = 0.4
            self.max_detections = 100
            print("⚡ Mode: FAST (CPU optimized)")
            
        else:
            self.performance_mode = "efficient"
            self.confidence_threshold = 0.5
            self.max_detections = 50
            print("💡 Mode: EFFICIENT (Limited hardware detected)")
    
    def _generate_color_palette(self, num_colors: int) -> Dict[int, Tuple[int, int, int]]:
        """Generate distinct colors for object visualization"""
        colors = {}
        for i in range(num_colors):
            hue = (i * 137.5) % 360  # Golden angle
            saturation = 0.7 + (i % 3) * 0.1
            value = 0.8 + (i % 2) * 0.2
            
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
            colors[i] = (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))
        
        return colors
    
    def _load_yolo_models(self) -> bool:
        """Load YOLO models based on performance mode"""
        if not DETECTION_AVAILABLE:
            return False
        
        print("📥 Loading YOLO models...")
        
        # Model configurations based on performance mode
        if self.performance_mode == "ultra_accurate":
            model_configs = [
                {'name': 'yolov8x.pt', 'priority': 1, 'weight': 1.0},
                {'name': 'yolov8l.pt', 'priority': 2, 'weight': 0.9},
                {'name': 'yolov8m.pt', 'priority': 3, 'weight': 0.8}
            ]
        elif self.performance_mode == "balanced":
            model_configs = [
                {'name': 'yolov8l.pt', 'priority': 1, 'weight': 1.0},
                {'name': 'yolov8m.pt', 'priority': 2, 'weight': 0.8}
            ]
        elif self.performance_mode == "fast":
            model_configs = [
                {'name': 'yolov8s.pt', 'priority': 1, 'weight': 1.0},
                {'name': 'yolov8n.pt', 'priority': 2, 'weight': 0.7}
            ]
        else:  # efficient
            model_configs = [
                {'name': 'yolov8n.pt', 'priority': 1, 'weight': 1.0}
            ]
        
        # PyTorch compatibility fix
        self._fix_pytorch_compatibility()
        
        loaded_count = 0
        for config in model_configs:
            try:
                print(f"🔄 Loading {config['name']}...")
                model = YOLO(config['name'])
                
                # Test model
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                test_results = model(dummy_frame, verbose=False)
                
                self.yolo_models[config['name']] = {
                    'model': model,
                    'priority': config['priority'],
                    'weight': config['weight'],
                    'classes': len(model.names),
                    'class_names': model.names
                }
                
                print(f"✅ {config['name']} loaded successfully")
                loaded_count += 1
                
            except Exception as e:
                print(f"⚠️ Failed to load {config['name']}: {e}")
                continue
        
        if loaded_count == 0:
            print("❌ No YOLO models loaded successfully")
            return False
        
        print(f"🎯 Successfully loaded {loaded_count} YOLO model(s)")
        return True
    
    def _load_gesture_model(self) -> bool:
        """Load gesture recognition model"""
        print("🤟 Loading gesture recognition...")
        
        try:
            # Initialize MediaPipe
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands_solution = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
                model_complexity=0 if self.performance_mode in ["fast", "efficient"] else 1
            )
            
            # Load trained gesture classifier if available
            if os.path.exists("gesture_model.pkl"):
                self.gesture_model = joblib.load("gesture_model.pkl")
                print("✅ Gesture classifier loaded")
            else:
                print("⚠️ No trained gesture model found - using basic gesture detection")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load gesture recognition: {e}")
            return False
    
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
                'collections.OrderedDict'
            ])
        except:
            pass
        
        # Monkey patch torch.load for compatibility
        original_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs.pop('weights_only', None)
            return original_load(*args, **kwargs, weights_only=False)
        torch.load = safe_load
    
    def initialize_models(self) -> bool:
        """Initialize all detection models"""
        print("🔧 Initializing all detection models...")
        
        yolo_success = self._load_yolo_models()
        gesture_success = self._load_gesture_model()
        
        if not yolo_success and not gesture_success:
            print("❌ Failed to load any detection models")
            return False
        
        print("✅ Model initialization complete")
        return True
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using YOLO models with ensemble or single model"""
        if not self.yolo_models:
            return []
        
        # Cache key for performance
        frame_hash = hash(frame.tobytes())
        cache_key = f"objects_{frame_hash}_{self.confidence_threshold}"
        
        # Check cache
        current_time = time.time()
        if cache_key in self.result_cache:
            cached_result, cache_time = self.result_cache[cache_key]
            if current_time - cache_time < self.cache_timeout:
                return cached_result
        
        detections = []
        
        try:
            if len(self.yolo_models) == 1 or self.performance_mode in ["fast", "efficient"]:
                # Single model detection for performance
                model_name = list(self.yolo_models.keys())[0]
                model_info = self.yolo_models[model_name]
                model = model_info['model']
                
                results = model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.nms_threshold,
                    max_det=self.max_detections,
                    verbose=False,
                    device=self.system_specs['device']
                )
                
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)
                        
                        for box, conf, cls_id in zip(boxes, confidences, class_ids):
                            if conf >= self.confidence_threshold:
                                detections.append({
                                    'box': box,
                                    'confidence': float(conf),
                                    'class_id': int(cls_id),
                                    'class_name': model.names[cls_id],
                                    'source_model': model_name
                                })
            
            else:
                # Ensemble detection for ultra accuracy
                all_detections = []
                
                for model_name, model_info in self.yolo_models.items():
                    model = model_info['model']
                    weight = model_info['weight']
                    
                    results = model(
                        frame,
                        conf=self.confidence_threshold * 0.8,  # Lower threshold for ensemble
                        iou=self.nms_threshold,
                        max_det=self.max_detections,
                        verbose=False,
                        device=self.system_specs['device']
                    )
                    
                    for result in results:
                        if result.boxes is not None and len(result.boxes) > 0:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            class_ids = result.boxes.cls.cpu().numpy().astype(int)
                            
                            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                                all_detections.append({
                                    'box': box,
                                    'confidence': float(conf) * weight,
                                    'class_id': int(cls_id),
                                    'class_name': model.names[cls_id],
                                    'source_model': model_name,
                                    'weight': weight
                                })
                
                # Ensemble fusion using weighted NMS
                detections = self._ensemble_fusion(all_detections)
            
            # Cache result
            self.result_cache[cache_key] = (detections, current_time)
            
            # Clean old cache entries
            if len(self.result_cache) > 50:
                oldest_keys = sorted(self.result_cache.keys(), 
                                   key=lambda k: self.result_cache[k][1])[:25]
                for key in oldest_keys:
                    del self.result_cache[key]
            
            return detections
            
        except Exception as e:
            print(f"⚠️ Object detection error: {e}")
            return []
    
    def _ensemble_fusion(self, detections: List[Dict]) -> List[Dict]:
        """Fuse detections from multiple models using advanced NMS"""
        if not detections:
            return []
        
        # Group detections by class
        class_detections = defaultdict(list)
        for det in detections:
            class_detections[det['class_id']].append(det)
        
        fused_detections = []
        
        for class_id, dets in class_detections.items():
            if not dets:
                continue
            
            # Sort by confidence
            dets.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Apply weighted NMS
            keep = []
            while dets:
                # Take highest confidence detection
                best = dets.pop(0)
                keep.append(best)
                
                # Remove overlapping detections
                remaining = []
                for det in dets:
                    iou = self._calculate_iou(best['box'], det['box'])
                    if iou < 0.5:  # Keep if IoU is low
                        remaining.append(det)
                    else:
                        # Merge overlapping detections with weighted averaging
                        if det['confidence'] > best['confidence'] * 0.8:
                            # High confidence overlap - merge
                            total_weight = best['weight'] + det['weight']
                            best['confidence'] = (best['confidence'] * best['weight'] + 
                                                det['confidence'] * det['weight']) / total_weight
                            best['weight'] = total_weight
                
                dets = remaining
            
            fused_detections.extend(keep)
        
        # Sort final results by confidence
        fused_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return fused_detections[:self.max_detections]
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _detect_gestures(self, frame: np.ndarray) -> List[Dict]:
        """Detect hand gestures using MediaPipe and trained classifier"""
        if not self.mp_hands:
            return []
        
        gestures = []
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands_solution.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Extract hand features
                    features = []
                    for lm in hand_landmarks.landmark:
                        features.extend([lm.x, lm.y, lm.z])
                    
                    gesture_name = "unknown"
                    confidence = 0.5
                    
                    # Classify gesture if model is available
                    if self.gesture_model and len(features) == 63:
                        try:
                            prediction = self.gesture_model.predict([features])[0]
                            probabilities = self.gesture_model.predict_proba([features])[0]
                            confidence = float(np.max(probabilities))
                            gesture_name = prediction
                        except:
                            # Simple rule-based gesture detection as fallback
                            gesture_name = self._rule_based_gesture_detection(hand_landmarks)
                    else:
                        # Rule-based detection
                        gesture_name = self._rule_based_gesture_detection(hand_landmarks)
                    
                    # Calculate bounding box for hand
                    landmarks_array = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                    x_min, y_min = np.min(landmarks_array, axis=0)
                    x_max, y_max = np.max(landmarks_array, axis=0)
                    
                    # Convert to pixel coordinates
                    h, w = frame.shape[:2]
                    box = [x_min * w, y_min * h, x_max * w, y_max * h]
                    
                    gestures.append({
                        'box': box,
                        'confidence': confidence,
                        'gesture': gesture_name,
                        'hand_landmarks': hand_landmarks,
                        'hand_index': hand_idx
                    })
            
            return gestures
            
        except Exception as e:
            print(f"⚠️ Gesture detection error: {e}")
            return []
    
    def _rule_based_gesture_detection(self, hand_landmarks) -> str:
        """Simple rule-based gesture detection as fallback"""
        try:
            # Extract key landmark positions
            landmarks = hand_landmarks.landmark
            
            # Fingertip and joint positions
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            ring_tip = landmarks[16]
            ring_pip = landmarks[14]
            pinky_tip = landmarks[20]
            pinky_pip = landmarks[18]
            
            # Count extended fingers
            extended_fingers = 0
            
            # Thumb (different logic due to orientation)
            if thumb_tip.x > thumb_mcp.x:  # Assuming right hand
                extended_fingers += 1
            
            # Other fingers
            if index_tip.y < index_pip.y:
                extended_fingers += 1
            if middle_tip.y < middle_pip.y:
                extended_fingers += 1
            if ring_tip.y < ring_pip.y:
                extended_fingers += 1
            if pinky_tip.y < pinky_pip.y:
                extended_fingers += 1
            
            # Classify gesture based on extended fingers
            if extended_fingers == 0:
                return "fist"
            elif extended_fingers == 1:
                if index_tip.y < index_pip.y:
                    return "pointing"
                else:
                    return "thumbs_up"
            elif extended_fingers == 2:
                if index_tip.y < index_pip.y and middle_tip.y < middle_pip.y:
                    return "peace"
                else:
                    return "two_fingers"
            elif extended_fingers == 5:
                return "open_palm"
            else:
                return "other"
                
        except:
            return "unknown"
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process a single frame with all detection modes"""
        start_time = time.time()
        
        # Detect objects and gestures in parallel if possible
        if self.performance_mode == "ultra_accurate" and self.system_specs['cpu_cores'] >= 4:
            # Parallel processing for high-end systems
            object_thread = threading.Thread(target=lambda: setattr(self, '_temp_objects', self._detect_objects(frame)))
            gesture_thread = threading.Thread(target=lambda: setattr(self, '_temp_gestures', self._detect_gestures(frame)))
            
            object_thread.start()
            gesture_thread.start()
            
            object_thread.join()
            gesture_thread.join()
            
            object_detections = getattr(self, '_temp_objects', [])
            gesture_detections = getattr(self, '_temp_gestures', [])
        else:
            # Sequential processing
            object_detections = self._detect_objects(frame)
            gesture_detections = self._detect_gestures(frame)
        
        # Draw detections on frame
        annotated_frame = self._draw_detections(frame, object_detections, gesture_detections)
        
        # Calculate processing time and FPS
        process_time = time.time() - start_time
        fps = 1.0 / process_time if process_time > 0 else 0
        self.stats['fps_history'].append(fps)
        
        # Update statistics
        self.stats['total_detections'] += len(object_detections)
        self.stats['gesture_detections'] += len(gesture_detections)
        
        for obj in object_detections:
            self.stats['objects_detected'][obj['class_name']] += 1
        
        for gesture in gesture_detections:
            self.stats['gestures_detected'][gesture['gesture']] += 1
        
        # Create result summary
        result_summary = {
            'objects': object_detections,
            'gestures': gesture_detections,
            'fps': fps,
            'processing_time': process_time,
            'performance_mode': self.performance_mode,
            'total_detections': len(object_detections) + len(gesture_detections)
        }
        
        return annotated_frame, result_summary
    
    def _draw_detections(self, frame: np.ndarray, objects: List[Dict], gestures: List[Dict]) -> np.ndarray:
        """Draw all detections on the frame with enhanced visualization"""
        annotated_frame = frame.copy()
        
        # Draw object detections
        for obj in objects:
            box = obj['box']
            confidence = obj['confidence']
            class_name = obj['class_name']
            class_id = obj['class_id']
            
            # Get color for this class
            color = self.colors.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            thickness = max(1, int(confidence * 3))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            if 'source_model' in obj and len(self.yolo_models) > 1:
                model_short = obj['source_model'].replace('yolov8', '').replace('.pt', '')
                label += f" ({model_short})"
            
            # Label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
            
            # Label text
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw gesture detections
        for gesture in gestures:
            box = gesture['box']
            confidence = gesture['confidence']
            gesture_name = gesture['gesture']
            
            # Draw hand landmarks if available
            if 'hand_landmarks' in gesture:
                self.mp_drawing.draw_landmarks(
                    annotated_frame, 
                    gesture['hand_landmarks'], 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            
            # Draw gesture bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Draw gesture label
            gesture_label = f"👋 {gesture_name}: {confidence:.2f}"
            cv2.putText(annotated_frame, gesture_label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw performance statistics
        self._draw_performance_overlay(annotated_frame)
        
        return annotated_frame
    
    def _draw_performance_overlay(self, frame: np.ndarray):
        """Draw performance statistics on the frame"""
        # Removed overlay information box as requested for clean production display
        # Statistics are still tracked internally and available via get_statistics()
        pass
    
    def get_statistics(self) -> Dict:
        """Get comprehensive system statistics"""
        session_time = time.time() - self.stats['session_start']
        avg_fps = np.mean(self.stats['fps_history']) if self.stats['fps_history'] else 0
        
        return {
            'session_duration': session_time,
            'average_fps': avg_fps,
            'total_object_detections': self.stats['total_detections'],
            'total_gesture_detections': self.stats['gesture_detections'],
            'performance_mode': self.performance_mode,
            'yolo_models_loaded': len(self.yolo_models),
            'gesture_model_loaded': self.gesture_model is not None,
            'hardware_device': self.system_specs['device'],
            'top_detected_objects': dict(sorted(self.stats['objects_detected'].items(), 
                                              key=lambda x: x[1], reverse=True)[:5]),
            'top_detected_gestures': dict(sorted(self.stats['gestures_detected'].items(), 
                                                key=lambda x: x[1], reverse=True)[:5])
        }
    
    def adjust_performance_mode(self, new_mode: str):
        """Dynamically adjust performance mode"""
        if new_mode in ["efficient", "fast", "balanced", "ultra_accurate"]:
            self.performance_mode = new_mode
            self._auto_configure_system()
            print(f"🔄 Switched to {new_mode} mode")
        else:
            print(f"❌ Invalid mode: {new_mode}")
    
    def save_detection_results(self, filename: str, results: Dict):
        """Save detection results to JSON file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"💾 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
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

def run_hybrid_ultimate_detection_system():
    """
    Main function to run the Hybrid Ultimate Detection System
    """
    print("🚀 HYBRID ULTIMATE DETECTION SYSTEM")
    print("=" * 80)
    print("🎯 Features:")
    print("  • Auto-optimized performance based on hardware")
    print("  • Ultra-accurate object detection with YOLO ensemble")
    print("  • Real-time hand gesture recognition")
    print("  • Adaptive performance modes (efficient/fast/balanced/ultra_accurate)")
    print("  • Production-ready stability and error handling")
    print("  • Multi-threaded processing for maximum performance")
    print("\n📋 Controls:")
    print("  • 'q' - Quit system")
    print("  • 's' - Save current frame and results")
    print("  • 'r' - Reset statistics")
    print("  • '1-4' - Switch performance mode (1=efficient, 2=fast, 3=balanced, 4=ultra_accurate)")
    print("  • 'c' - Cycle confidence threshold")
    print("=" * 80)
    
    # Initialize the system
    detector = HybridUltimateDetectionSystem(auto_configure=True)
    
    if not detector.initialize_models():
        print("❌ Failed to initialize detection models")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Set optimal camera settings based on performance mode
    if detector.performance_mode == "ultra_accurate":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
    elif detector.performance_mode == "balanced":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:  # fast or efficient
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 25)
    
    # Set up fullscreen window
    try:
        cv2.namedWindow("Hybrid Ultimate Detection System", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Hybrid Ultimate Detection System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception as e:
        print(f"⚠️ Fullscreen mode not supported, using normal window: {e}")
        cv2.namedWindow("Hybrid Ultimate Detection System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hybrid Ultimate Detection System", 1920, 1080)
    
    print(f"✅ Camera initialized in {detector.performance_mode} mode")
    print("🎬 Starting detection in fullscreen mode... Press 'q' to quit")
    
    try:
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame with hybrid detection
            annotated_frame, results = detector.process_frame(frame)
            
            # Display the result (now in fullscreen without overlay)
            cv2.imshow("Hybrid Ultimate Detection System", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Exiting Hybrid Ultimate Detection System...")
                break
            elif key == ord('s'):
                # Save current frame and results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                frame_filename = f"hybrid_detection_frame_{timestamp}.jpg"
                results_filename = f"hybrid_detection_results_{timestamp}.json"
                
                cv2.imwrite(frame_filename, annotated_frame)
                detector.save_detection_results(results_filename, results)
                
                print(f"💾 Saved: {frame_filename} and {results_filename}")
            
            elif key == ord('r'):
                # Reset statistics
                detector.stats = {
                    'total_detections': 0,
                    'gesture_detections': 0,
                    'objects_detected': defaultdict(int),
                    'gestures_detected': defaultdict(int),
                    'fps_history': deque(maxlen=100),
                    'accuracy_scores': deque(maxlen=50),
                    'model_performance': {},
                    'session_start': time.time()
                }
                print("🔄 Statistics reset")
            
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                # Switch performance mode
                modes = ['efficient', 'fast', 'balanced', 'ultra_accurate']
                mode_index = int(chr(key)) - 1
                if 0 <= mode_index < len(modes):
                    detector.adjust_performance_mode(modes[mode_index])
            
            elif key == ord('c'):
                # Cycle confidence threshold
                thresholds = [0.1, 0.25, 0.4, 0.5, 0.7]
                current_idx = thresholds.index(detector.confidence_threshold) if detector.confidence_threshold in thresholds else 1
                next_idx = (current_idx + 1) % len(thresholds)
                detector.confidence_threshold = thresholds[next_idx]
                print(f"🎚️ Confidence threshold: {detector.confidence_threshold}")
            
            # Print periodic statistics
            if frame_count % 100 == 0:
                stats = detector.get_statistics()
                print(f"\n📊 Frame {frame_count} - Stats:")
                print(f"   FPS: {stats['average_fps']:.1f}")
                print(f"   Objects: {stats['total_object_detections']}")
                print(f"   Gestures: {stats['total_gesture_detections']}")
                print(f"   Mode: {stats['performance_mode']}")
    
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        final_stats = detector.get_statistics()
        print("\n" + "=" * 60)
        print("📊 FINAL SESSION STATISTICS")
        print("=" * 60)
        print(f"🕐 Session Duration: {final_stats['session_duration']:.1f} seconds")
        print(f"📈 Average FPS: {final_stats['average_fps']:.2f}")
        print(f"🎯 Total Object Detections: {final_stats['total_object_detections']}")
        print(f"🤟 Total Gesture Detections: {final_stats['total_gesture_detections']}")
        print(f"⚡ Performance Mode: {final_stats['performance_mode']}")
        print(f"🤖 YOLO Models Used: {final_stats['yolo_models_loaded']}")
        print(f"👋 Gesture Model: {'✅' if final_stats['gesture_model_loaded'] else '❌'}")
        print(f"🖥️  Hardware: {final_stats['hardware_device'].upper()}")
        
        if final_stats['top_detected_objects']:
            print(f"\n🏆 Top Detected Objects:")
            for obj, count in final_stats['top_detected_objects'].items():
                print(f"   • {obj}: {count}")
        
        if final_stats['top_detected_gestures']:
            print(f"\n👋 Top Detected Gestures:")
            for gesture, count in final_stats['top_detected_gestures'].items():
                print(f"   • {gesture}: {count}")
        
        print("=" * 60)
        print("✅ Hybrid Ultimate Detection System session completed!")

if __name__ == "__main__":
    run_hybrid_ultimate_detection_system()
