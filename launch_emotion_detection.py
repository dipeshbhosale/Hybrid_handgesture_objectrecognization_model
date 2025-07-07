#!/usr/bin/env python3
"""
Emotion Detection Launcher Script
Integrates with the existing Hybrid Ultimate Detection System
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import cv2
        print("✅ OpenCV is installed")
    except ImportError:
        print("❌ OpenCV is not installed")
        print("💡 Install with: pip install opencv-python")
        return False

    try:
        from deepface import DeepFace
        print("✅ DeepFace is installed")
    except ImportError:
        print("❌ DeepFace is not installed")
        print("💡 Install with: pip install deepface")
        return False

    # Check for optional dependencies
    try:
        import pandas as pd
        print("✅ Pandas is installed (for emotion logging)")
    except ImportError:
        print("⚠️ Pandas is not installed (required for emotion logging)")
        print("💡 Install with: pip install pandas")

    # Check for main.py
    if os.path.exists('main.py'):
        print("✅ main.py found")
    else:
        print("⚠️ main.py not found (required for full integration)")

    # Check for hybrid_ultimate_detection.py
    if os.path.exists('hybrid_ultimate_detection.py'):
        print("✅ hybrid_ultimate_detection.py found")
    else:
        print("⚠️ hybrid_ultimate_detection.py not found (required for hybrid integration)")

    return True

def display_menu():
    """Display the emotion detection menu"""
    print("\n" + "="*60)
    print("🎭 EMOTION DETECTION SYSTEM - MENU")
    print("="*60)
    print("1. Run Standalone Emotion Detection")
    print("2. Run Emotion Detection with Hybrid Integration")
    print("3. Toggle Emotion Logging")
    print("4. View Emotion Log Analysis")
    print("5. Update Dependencies")
    print("6. Go Back to Main Menu")
    print("7. Exit")
    print("="*60)

def update_requirements_file():
    """Add required dependencies to requirements.txt"""
    requirements = []
    
    # Read existing requirements if file exists
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as file:
            requirements = [line.strip() for line in file.readlines()]
    
    # Add required packages if not already in the list
    required_packages = ["deepface", "opencv-python", "numpy", "pandas"]
    for package in required_packages:
        if not any(p.startswith(package) for p in requirements):
            requirements.append(package)
    
    # Write updated requirements
    with open('requirements.txt', 'w') as file:
        for package in requirements:
            file.write(f"{package}\n")
    
    print("✅ Updated requirements.txt with emotion detection dependencies")

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Update requirements.txt first
    update_requirements_file()
    
    # Install required packages
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Successfully installed dependencies")
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        print("💡 Try running: pip install -r requirements.txt")

def view_emotion_logs():
    """View analysis of emotion logs if available"""
    log_file = "emotion_logs.csv"
    if not os.path.exists(log_file):
        print("❌ No emotion logs found")
        print("💡 Enable logging first by running emotion detection and pressing 'l'")
        return
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Load the logs
        print("📊 Loading emotion logs...")
        logs = pd.read_csv(log_file)
        
        # Convert timestamp to datetime
        logs['timestamp'] = pd.to_datetime(logs['timestamp'])
        
        # Basic statistics
        print(f"\n📈 EMOTION LOG ANALYSIS:")
        print(f"  • Total entries: {len(logs)}")
        print(f"  • Time period: {logs['timestamp'].min()} to {logs['timestamp'].max()}")
        print(f"  • Unique faces: {logs['face_id'].nunique()}")
        
        # Emotion distribution
        emotion_counts = logs['emotion'].value_counts()
        print(f"\n🎭 EMOTION DISTRIBUTION:")
        for emotion, count in emotion_counts.items():
            percentage = (count / len(logs)) * 100
            print(f"  • {emotion}: {count} ({percentage:.1f}%)")
        
        print("\n💡 For advanced analysis, you can open the CSV file in Excel or another tool")
        
    except ImportError:
        print("❌ Pandas or matplotlib not installed")
        print("💡 Install with: pip install pandas matplotlib")
    except Exception as e:
        print(f"❌ Error analyzing logs: {e}")

def run_standalone_emotion():
    """Run the standalone emotion detection system"""
    try:
        print("\n🚀 Starting standalone emotion detection...")
        from emotion_detection import EmotionDetector
        
        # Create and run the detector
        detector = EmotionDetector()
        
        # Use the main function from the module
        if hasattr(detector, "main"):
            detector.main()
        else:
            # Fallback to running the module directly
            import emotion_detection
            if hasattr(emotion_detection, "main"):
                emotion_detection.main()
            else:
                # If no main function, import the module which should run automatically
                subprocess.run([sys.executable, "emotion_detection.py"])
    
    except ImportError:
        print("❌ Could not import emotion detection module")
        print("💡 Make sure emotion_detection.py is in the current directory")
    except Exception as e:
        print(f"❌ Error running emotion detection: {e}")

def run_hybrid_integration():
    """Run the emotion detection with hybrid integration"""
    try:
        print("\n🚀 Starting emotion detection with hybrid integration...")
        # Try to import and run
        import emotion_hybrid_integration
        emotion_hybrid_integration.run_emotion_detection_system()
    except ImportError:
        print("❌ Could not import emotion hybrid integration module")
        print("💡 Make sure emotion_hybrid_integration.py is in the current directory")
    except Exception as e:
        print(f"❌ Error running hybrid integration: {e}")

def toggle_emotion_logging():
    """Toggle emotion logging settings"""
    try:
        from emotion_detection import EmotionDetector
        detector = EmotionDetector()
        logging_enabled = detector.toggle_logging()
        status = "enabled" if logging_enabled else "disabled"
        print(f"✅ Emotion logging {status}")
        print(f"📁 Log file: emotion_logs.csv")
    except Exception as e:
        print(f"❌ Error toggling emotion logging: {e}")

def main():
    """Main function for the emotion detection launcher"""
    print("\n🎭 EMOTION DETECTION LAUNCHER")
    print("="*40)
    
    # First, check if emotion_detection.py exists
    if not os.path.exists('emotion_detection.py'):
        print("❌ emotion_detection.py not found!")
        print("💡 Make sure the file is in the current directory")
        return
    
    # Check dependencies
    dependencies_ok = check_dependencies()
    if not dependencies_ok:
        print("\n⚠️ Some dependencies are missing")
        install = input("Do you want to install required dependencies? (y/N): ")
        if install.lower() == 'y':
            install_dependencies()
        else:
            print("⚠️ Dependencies not installed. Some features may not work.")
    
    # Update requirements.txt in any case
    update_requirements_file()
    
    # Main menu loop
    while True:
        display_menu()
        choice = input("Enter your choice (1-7): ")
        
        if choice == '1':
            run_standalone_emotion()
        elif choice == '2':
            run_hybrid_integration()
        elif choice == '3':
            toggle_emotion_logging()
        elif choice == '4':
            view_emotion_logs()
        elif choice == '5':
            install_dependencies()
        elif choice == '6':
            # Try to run the main.py script
            print("\n🔄 Returning to main menu...")
            try:
                subprocess.run([sys.executable, "main.py"])
                # After main.py exits, we'll continue from here
                print("\n🎭 Back to Emotion Detection Launcher")
            except Exception as e:
                print(f"❌ Error running main.py: {e}")
        elif choice == '7':
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
