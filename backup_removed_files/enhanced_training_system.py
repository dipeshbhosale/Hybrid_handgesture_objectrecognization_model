#!/usr/bin/env python3
"""
Enhanced Training System for Ultra-Accurate Detection
====================================================
This script extends the ultra-accurate detection system with additional training
capabilities using external datasets and GitHub repositories.
"""

import os
import sys
import requests
import zipfile
import json
import cv2
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import urllib.request
import hashlib

class EnhancedTrainingSystem:
    """Enhanced training system for maximum accuracy"""
    
    def __init__(self):
        self.data_dir = Path("training_data")
        self.models_dir = Path("enhanced_models")
        self.cache_dir = Path("dataset_cache")
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.cache_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Popular datasets for object detection training
        self.datasets = {
            "coco_sample": {
                "name": "COCO Sample Dataset",
                "description": "Sample from COCO dataset for enhanced training",
                "url": "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",
                "size": "6.8MB",
                "images": 128,
                "classes": 80
            },
            "open_images_sample": {
                "name": "Open Images Sample",
                "description": "Sample from Google Open Images dataset",
                "url": "https://storage.googleapis.com/openimages/web/download.html",
                "note": "Custom download required"
            },
            "roboflow_datasets": {
                "name": "Roboflow Public Datasets",
                "description": "Community-contributed datasets from Roboflow",
                "url": "https://public.roboflow.com/",
                "note": "API access required"
            }
        }
    
    def download_coco_sample(self):
        """Download and prepare COCO sample dataset"""
        print("📥 Downloading COCO sample dataset...")
        
        coco_url = self.datasets["coco_sample"]["url"]
        cache_file = self.cache_dir / "coco128.zip"
        extract_dir = self.data_dir / "coco128"
        
        try:
            if not cache_file.exists():
                print(f"🔄 Downloading from {coco_url}")
                urllib.request.urlretrieve(coco_url, cache_file)
                print("✅ Download completed")
            else:
                print("✅ Using cached dataset")
            
            # Extract if not already extracted
            if not extract_dir.exists():
                print("📂 Extracting dataset...")
                with zipfile.ZipFile(cache_file, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                print("✅ Dataset extracted")
            
            # Verify dataset structure
            if (extract_dir / "images").exists() and (extract_dir / "labels").exists():
                image_count = len(list((extract_dir / "images").glob("*.jpg")))
                label_count = len(list((extract_dir / "labels").glob("*.txt")))
                
                print(f"📊 Dataset verified: {image_count} images, {label_count} labels")
                return str(extract_dir)
            else:
                print("❌ Dataset structure invalid")
                return None
                
        except Exception as e:
            print(f"❌ Failed to download COCO sample: {e}")
            return None
    
    def create_custom_training_config(self, dataset_path, model_name="yolov8n"):
        """Create custom training configuration for enhanced accuracy"""
        print(f"⚙️ Creating training config for {model_name}...")
        
        config = {
            "model": f"{model_name}.pt",
            "data": {
                "path": str(dataset_path),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": 80,  # Number of classes
                "names": [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
                    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                    'toothbrush'
                ]
            },
            "training": {
                "epochs": 100,
                "imgsz": 640,
                "batch": 16,
                "lr0": 0.01,
                "lrf": 0.01,
                "momentum": 0.937,
                "weight_decay": 0.0005,
                "warmup_epochs": 3,
                "warmup_momentum": 0.8,
                "warmup_bias_lr": 0.1,
                "box": 0.05,
                "cls": 0.5,
                "cls_pw": 1.0,
                "obj": 1.0,
                "obj_pw": 1.0,
                "iou_t": 0.20,
                "anchor_t": 4.0,
                "fl_gamma": 0.0,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic": 1.0,
                "mixup": 0.0,
                "copy_paste": 0.0
            }
        }
        
        config_file = self.models_dir / f"{model_name}_training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Training config saved: {config_file}")
        return config_file
    
    def train_enhanced_model(self, model_name="yolov8n", dataset_path=None, epochs=50):
        """Train an enhanced model with custom dataset"""
        print(f"🚀 Starting enhanced training for {model_name}")
        print("=" * 50)
        
        try:
            from ultralytics import YOLO
            
            # Load base model
            print(f"📥 Loading base model: {model_name}.pt")
            model = YOLO(f"{model_name}.pt")
            
            if dataset_path is None:
                # Use downloaded COCO sample
                dataset_path = self.download_coco_sample()
                if dataset_path is None:
                    print("❌ No dataset available for training")
                    return None
            
            # Create training configuration
            config_file = self.create_custom_training_config(dataset_path, model_name)
            
            # Enhanced training parameters for maximum accuracy
            training_args = {
                'data': str(Path(dataset_path) / 'data.yaml'),
                'epochs': epochs,
                'imgsz': 640,
                'batch': 8,  # Smaller batch for better accuracy
                'lr0': 0.005,  # Lower learning rate for fine-tuning
                'patience': 20,  # Early stopping patience
                'save': True,
                'save_period': 10,
                'cache': True,
                'device': 'auto',
                'workers': 4,
                'project': str(self.models_dir),
                'name': f"{model_name}_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'AdamW',  # Better optimizer for accuracy
                'verbose': True,
                'seed': 42,
                'deterministic': True,
                'single_cls': False,
                'rect': True,  # Rectangular training for better accuracy
                'cos_lr': True,  # Cosine learning rate schedule
                'close_mosaic': 10,  # Close mosaic augmentation in last epochs
                'resume': False,
                'amp': True,  # Automatic Mixed Precision
                'fraction': 1.0,  # Use full dataset
                'profile': False,
                'freeze': None,
                'multi_scale': True,  # Multi-scale training for robustness
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'split': 'val',
                'save_json': True,
                'save_hybrid': False,
                'conf': None,
                'iou': 0.7,
                'max_det': 300,
                'half': False,
                'dnn': False,
                'plots': True,
                'source': None,
                'show': False,
                'save_txt': False,
                'save_conf': False,
                'save_crop': False,
                'show_labels': True,
                'show_conf': True,
                'vid_stride': 1,
                'stream_buffer': False,
                'line_width': None,
                'visualize': False,
                'augment': False,
                'agnostic_nms': False,
                'classes': None,
                'retina_masks': False,
                'boxes': True,
                'format': 'torchscript',
                'keras': False,
                'optimize': False,
                'int8': False,
                'dynamic': False,
                'simplify': False,
                'opset': None,
                'workspace': 4,
                'nms': False,
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 1.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'bgr': 0.0,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0,
                'auto_augment': 'randaugment',
                'erasing': 0.4,
                'crop_fraction': 1.0,
                'cfg': None,
                'tracker': 'botsort.yaml',
                'save_frames': False,
                'save_txt': False,
                'save_conf': False,
                'save_crop': False,
                'show_labels': True,
                'show_conf': True,
                'vid_stride': 1,
                'line_width': None,
                'visualize': False,
                'augment': False,
                'agnostic_nms': False,
                'classes': None,
                'retina_masks': False,
                'embed': None,
                'show': False,
                'save': True,
                'plots': True,
                'source': None
            }
            
            print("🔥 Starting enhanced training with maximum accuracy settings...")
            print(f"📊 Training parameters:")
            print(f"   • Epochs: {epochs}")
            print(f"   • Batch size: 8 (optimized for accuracy)")
            print(f"   • Learning rate: 0.005 (fine-tuning)")
            print(f"   • Optimizer: AdamW")
            print(f"   • Multi-scale training: Enabled")
            print(f"   • Cosine LR schedule: Enabled")
            
            # Start training
            start_time = time.time()
            results = model.train(**training_args)
            training_time = time.time() - start_time
            
            print(f"✅ Training completed in {training_time:.1f} seconds")
            print(f"📊 Results saved in: {results.save_dir}")
            
            # Save enhanced model
            enhanced_model_path = self.models_dir / f"{model_name}_enhanced.pt"
            model.save(enhanced_model_path)
            
            print(f"💾 Enhanced model saved: {enhanced_model_path}")
            
            return enhanced_model_path
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def validate_enhanced_model(self, model_path, test_images_dir=None):
        """Validate the enhanced model against test images"""
        print(f"🧪 Validating enhanced model: {model_path}")
        
        try:
            from ultralytics import YOLO
            
            # Load enhanced model
            model = YOLO(str(model_path))
            
            if test_images_dir is None:
                # Use sample images from dataset
                test_images_dir = self.data_dir / "coco128" / "images"
            
            if not Path(test_images_dir).exists():
                print("❌ No test images found")
                return False
            
            # Test on sample images
            test_images = list(Path(test_images_dir).glob("*.jpg"))[:10]
            
            results = []
            for img_path in test_images:
                result = model(str(img_path))
                results.append(result)
            
            print(f"✅ Validated on {len(test_images)} test images")
            print(f"📊 Average confidence: {np.mean([r.boxes.conf.mean() if r.boxes else 0 for r in results]):.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False
    
    def create_dataset_yaml(self, dataset_path):
        """Create dataset YAML file for training"""
        yaml_content = f"""
path: {dataset_path}
train: images
val: images
test: images

nc: 80
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        """
        
        yaml_path = Path(dataset_path) / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content.strip())
        
        print(f"📄 Dataset YAML created: {yaml_path}")
        return yaml_path

def main():
    """Main training function"""
    print("🎯 Enhanced Training System for Ultra-Accurate Detection")
    print("=" * 60)
    
    trainer = EnhancedTrainingSystem()
    
    print("🔽 Available training options:")
    print("1. Download and prepare COCO sample dataset")
    print("2. Train enhanced YOLOv8n model")
    print("3. Train enhanced YOLOv8s model")
    print("4. Train enhanced YOLOv8m model")
    print("5. Train all models for maximum ensemble accuracy")
    print("6. Validate existing enhanced models")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == "1":
        print("📥 Downloading and preparing dataset...")
        dataset_path = trainer.download_coco_sample()
        if dataset_path:
            # Create dataset YAML
            trainer.create_dataset_yaml(dataset_path)
            print("✅ Dataset preparation completed")
        else:
            print("❌ Dataset preparation failed")
    
    elif choice in ["2", "3", "4"]:
        model_map = {"2": "yolov8n", "3": "yolov8s", "4": "yolov8m"}
        model_name = model_map[choice]
        
        print(f"🚀 Training enhanced {model_name} model...")
        
        # Prepare dataset
        dataset_path = trainer.download_coco_sample()
        if dataset_path:
            trainer.create_dataset_yaml(dataset_path)
            
            # Train model
            enhanced_model = trainer.train_enhanced_model(model_name, dataset_path, epochs=50)
            
            if enhanced_model:
                # Validate model
                trainer.validate_enhanced_model(enhanced_model)
                print(f"✅ Enhanced {model_name} model training completed!")
            else:
                print(f"❌ Enhanced {model_name} model training failed")
        else:
            print("❌ Cannot proceed without dataset")
    
    elif choice == "5":
        print("🎯 Training all models for maximum ensemble accuracy...")
        models = ["yolov8n", "yolov8s", "yolov8m"]
        
        # Prepare dataset once
        dataset_path = trainer.download_coco_sample()
        if dataset_path:
            trainer.create_dataset_yaml(dataset_path)
            
            enhanced_models = []
            for model_name in models:
                print(f"\n🔥 Training {model_name}...")
                enhanced_model = trainer.train_enhanced_model(model_name, dataset_path, epochs=30)
                if enhanced_model:
                    enhanced_models.append(enhanced_model)
                    print(f"✅ {model_name} completed")
                else:
                    print(f"❌ {model_name} failed")
            
            print(f"\n🎉 Enhanced training completed!")
            print(f"📊 Successfully trained {len(enhanced_models)}/{len(models)} models")
            print("💡 These models will provide maximum accuracy in ensemble mode")
        else:
            print("❌ Cannot proceed without dataset")
    
    elif choice == "6":
        print("🧪 Validating existing enhanced models...")
        enhanced_models = list(trainer.models_dir.glob("*enhanced*.pt"))
        
        if enhanced_models:
            for model_path in enhanced_models:
                print(f"\n🔍 Validating {model_path.name}...")
                trainer.validate_enhanced_model(model_path)
        else:
            print("❌ No enhanced models found")
            print("💡 Run training first with options 2-5")
    
    else:
        print("❌ Invalid option selected")

if __name__ == "__main__":
    main()
