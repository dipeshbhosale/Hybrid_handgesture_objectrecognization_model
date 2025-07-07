import sys
import os
import traceback

def verify_files():
    """Check if all required files exist"""
    required_files = [
        "hybrid_ultimate_detection.py",
        "main.py",
        "hybrid_web_interface.py",
        "launch_hybrid.py",
        "gesture_model.pkl",
        "gesture_data.csv",
        "requirements.txt"
    ]
    
    yolo_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    
    print("\n=== Verifying Files ===")
    all_files_exist = True
    
    for file in required_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"{status} {file}")
        if not exists:
            all_files_exist = False
    
    print("\n--- YOLO Models ---")
    yolo_exists = False
    for model in yolo_models:
        exists = os.path.exists(model)
        status = "✅" if exists else "❌"
        print(f"{status} {model}")
        if exists:
            yolo_exists = True
    
    if not yolo_exists:
        print("⚠️ No YOLO models found. At least one is required.")
        all_files_exist = False
    
    return all_files_exist

def verify_imports():
    """Check if all required packages can be imported"""
    print("\n=== Verifying Imports ===")
    
    imports = [
        ("cv2", "opencv-python"), 
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("joblib", "joblib"),
        ("gradio", "gradio>=3.50.0"),
        ("keyboard", "keyboard"),
        ("torch", "torch"),
        ("ultralytics", "ultralytics")
    ]
    
    all_imports_ok = True
    
    for module, package in imports:
        try:
            __import__(module)
            print(f"✅ {module} ({package})")
        except ImportError as e:
            print(f"❌ {module} ({package}) - {e}")
            all_imports_ok = False
    
    return all_imports_ok

def test_module_import():
    """Test importing the main modules"""
    print("\n=== Testing Module Imports ===")
    all_modules_ok = True
    
    modules = [
        "hybrid_ultimate_detection",
        "main",
        "hybrid_web_interface",
        "launch_hybrid"
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}.py")
        except Exception as e:
            print(f"❌ {module}.py - {str(e)}")
            all_modules_ok = False
    
    return all_modules_ok

def test_system_init():
    """Test initializing the hybrid detection system"""
    print("\n=== Testing System Initialization ===")
    
    try:
        from hybrid_ultimate_detection import HybridUltimateDetectionSystem
        print("✅ Imported HybridUltimateDetectionSystem class")
        
        try:
            system = HybridUltimateDetectionSystem(auto_configure=True)
            print("✅ System initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Error initializing system: {str(e)}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Error importing system class: {str(e)}")
        traceback.print_exc()
        return False

def verify_web_interface():
    """Verify the web interface configuration"""
    print("\n=== Verifying Web Interface ===")
    
    try:
        from hybrid_web_interface import HybridWebInterface
        # Check if the class is properly defined
        if hasattr(HybridWebInterface, 'initialize_detector') and \
           hasattr(HybridWebInterface, 'process_video_frame'):
            print("✅ Web interface class is properly defined")
            return True
        else:
            print("❌ Web interface class is missing required methods")
            return False
    except Exception as e:
        print(f"❌ Error importing web interface: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n==== HYBRID ULTIMATE DETECTION SYSTEM - VERIFICATION ====")
    
    files_ok = verify_files()
    imports_ok = verify_imports()
    modules_ok = test_module_import()
    system_ok = test_system_init()
    web_ok = verify_web_interface()
    
    print("\n=== Verification Summary ===")
    print(f"Files: {'✅' if files_ok else '❌'}")
    print(f"Imports: {'✅' if imports_ok else '❌'}")
    print(f"Modules: {'✅' if modules_ok else '❌'}")
    print(f"System: {'✅' if system_ok else '❌'}")
    print(f"Web Interface: {'✅' if web_ok else '❌'}")
    
    print("\n📌 Web interface should be accessible at: http://127.0.0.1:7860 when launched")
    
    if all([files_ok, imports_ok, modules_ok, system_ok, web_ok]):
        print("\n✅ VERIFICATION PASSED - System is ready to use!")
    else:
        print("\n❌ VERIFICATION FAILED - Please check the issues above")
