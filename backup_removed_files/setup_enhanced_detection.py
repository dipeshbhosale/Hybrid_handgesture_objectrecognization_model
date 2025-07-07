#!/usr/bin/env python3
"""
Enhanced Object Detection Setup
Downloads and configures enhanced models for better object detection
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
import torch
from ultralytics import YOLO
import time

def install_requirements():
    """Install or upgrade required packages"""
    print("🔧 Installing/upgrading required packages...")
    
    packages = [
        "ultralytics>=8.0.0",
        "torch>=2.0.0", 
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0"
    ]
    
    for package in packages:
        try:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Failed to install {package}: {e}")

def fix_torch_compatibility():
    """Fix PyTorch compatibility issues"""
    print("🔧 Fixing PyTorch compatibility...")
    
    try:
        # Add safe globals
        torch.serialization.add_safe_globals([
            'ultralytics.nn.tasks.DetectionModel',
            'ultralytics.nn.modules.head.Detect', 
            'ultralytics.nn.modules.conv.Conv',
            'ultralytics.nn.modules.block.C2f',
            'ultralytics.nn.modules.block.SPPF',
            'collections.OrderedDict'
        ])
        print("✅ Torch compatibility fixed")
        return True
    except Exception as e:
        print(f"⚠️ Torch compatibility warning: {e}")
        return False

def download_enhanced_models():
    """Download multiple YOLO models for better object detection"""
    print("🚀 Downloading enhanced YOLO models...")
    
    models = [
        ("yolov8n.pt", "Nano - Fastest"),
        ("yolov8s.pt", "Small - Balanced"), 
        ("yolov8m.pt", "Medium - Better accuracy"),
        ("yolov8l.pt", "Large - High accuracy"),
        ("yolov8x.pt", "Extra Large - Highest accuracy")
    ]
    
    downloaded_models = []
    
    for model_name, description in models:
        try:
            if os.path.exists(model_name):
                print(f"✅ {model_name} already exists")
                downloaded_models.append(model_name)
                continue
                
            print(f"📥 Downloading {model_name} - {description}")
            
            # Use YOLO's auto-download feature
            model = YOLO(model_name)
            
            # Test the model
            test_frame = torch.zeros((3, 640, 640))
            model(test_frame, verbose=False)
            
            downloaded_models.append(model_name)
            print(f"✅ {model_name} downloaded and verified")
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
    
    return downloaded_models

def test_models(model_list):
    """Test downloaded models for functionality"""
    print("🧪 Testing model functionality...")
    
    working_models = []
    
    for model_name in model_list:
        try:
            print(f"🔍 Testing {model_name}...")
            
            model = YOLO(model_name)
            
            # Test with dummy image
            import numpy as np
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            start_time = time.time()
            results = model(test_image, verbose=False)
            inference_time = time.time() - start_time
            
            print(f"   ⏱️ Inference time: {inference_time:.3f}s")
            print(f"   📊 Classes: {len(model.names)}")
            print(f"   🎯 Sample classes: {list(model.names.values())[:5]}")
            
            working_models.append(model_name)
            print(f"✅ {model_name} working correctly")
            
        except Exception as e:
            print(f"❌ {model_name} test failed: {e}")
    
    return working_models

def create_enhanced_config():
    """Create enhanced configuration for better detection"""
    print("⚙️ Creating enhanced detection configuration...")
    
    config = {
        "detection_settings": {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.4,
            "max_detections": 100,
            "image_size": 640
        },
        "visualization": {
            "color_palette": "enhanced",
            "show_confidence": True,
            "show_statistics": True,
            "thick_boxes": True
        },
        "performance": {
            "use_gpu": torch.cuda.is_available(),
            "batch_size": 1,
            "half_precision": False
        }
    }
    
    import json
    with open("detection_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration saved to detection_config.json")
    return config

def run_quick_test():
    """Run a quick test of the enhanced detection system"""
    print("🎯 Running quick detection test...")
    
    try:
        from enhanced_object_detection import EnhancedObjectDetector
        
        detector = EnhancedObjectDetector()
        
        if detector.load_models():
            print("✅ Enhanced detection system ready!")
            print(f"   📊 Models loaded: {len(detector.models)}")
            print(f"   🎯 Primary model: {detector.primary_model}")
            return True
        else:
            print("❌ Enhanced detection system failed to initialize")
            return False
            
    except ImportError:
        print("⚠️ Enhanced detection module not found, using standard detection")
        try:
            model = YOLO("yolov8s.pt")
            print("✅ Standard YOLO detection available")
            return True
        except:
            print("❌ No detection system available")
            return False

def main():
    """Main setup function"""
    print("🚀 Enhanced Object Detection Setup")
    print("=" * 50)
    
    # Step 1: Install requirements
    install_requirements()
    
    # Step 2: Fix compatibility
    fix_torch_compatibility()
    
    # Step 3: Download models
    downloaded_models = download_enhanced_models()
    
    if not downloaded_models:
        print("❌ No models downloaded successfully!")
        return False
    
    # Step 4: Test models
    working_models = test_models(downloaded_models)
    
    if not working_models:
        print("❌ No working models found!")
        return False
    
    # Step 5: Create configuration
    config = create_enhanced_config()
    
    # Step 6: Quick test
    test_result = run_quick_test()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETE!")
    print("=" * 50)
    print(f"✅ Models available: {', '.join(working_models)}")
    print(f"✅ GPU available: {torch.cuda.is_available()}")
    print(f"✅ Configuration created: detection_config.json")
    print(f"✅ System test: {'PASSED' if test_result else 'FAILED'}")
    
    print("\n🚀 Ready to run enhanced object detection!")
    print("📋 Run these commands:")
    print("   python main.py  (choose option 7 or 8)")
    print("   python enhanced_object_detection.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed! Please check error messages above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")
