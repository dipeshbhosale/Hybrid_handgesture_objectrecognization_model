#!/usr/bin/env python3
"""
HYBRID ULTIMATE DETECTION - LAUNCHER
===================================
Unified launcher for Hybrid Ultimate Detection System
Choose between different interfaces and modes

Author: AI Vision System
Date: July 3, 2025
Version: 1.0
"""

import os
import sys
import argparse
import time

def print_banner():
    """Display system banner"""
    print("\033[1;36m") # Bright cyan color
    print("=" * 80)
    print("""
    🚀 HYBRID ULTIMATE DETECTION SYSTEM - LAUNCHER
    ---------------------------------------------
    The ultimate AI vision experience with auto-configuration,
    multi-modal detection, and enterprise-grade features.
    """)
    print("=" * 80)
    print("\033[0m") # Reset color

def validate_requirements():
    """Check if all requirements are installed"""
    try:
        import cv2
        import numpy
        import torch
        import mediapipe
        import gradio
        from ultralytics import YOLO
        import GPUtil
        import psutil
        
        print("✅ All core dependencies verified")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("ℹ️ Run 'pip install -r requirements.txt' to install dependencies")
        return False

def main():
    """Main launcher function"""
    print_banner()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Launch Hybrid Ultimate Detection System')
    
    parser.add_argument('--mode', type=str, choices=['cli', 'web', 'fullscreen', 'auto'],
                       default='auto', help='Interface mode (cli, web, fullscreen, auto)')
    
    parser.add_argument('--performance', type=str, 
                       choices=['efficient', 'fast', 'balanced', 'ultra_accurate', 'auto'],
                       default='auto', help='Performance mode')
    
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source (0 for webcam, file path, or URL)')
    
    parser.add_argument('--record', action='store_true', 
                       help='Record detection output to video file')
    
    parser.add_argument('--no-check', action='store_true',
                       help='Skip dependency checks')
    
    args = parser.parse_args()
    
    # Validate dependencies unless --no-check is specified
    if not args.no_check:
        if not validate_requirements():
            user_choice = input("Continue anyway? (y/n): ").lower()
            if user_choice != 'y':
                sys.exit(1)
    
    # Launch the selected interface
    try:
        if args.mode == 'web' or (args.mode == 'auto' and has_display()):
            # Web interface (Gradio)
            print("🌐 Launching web interface...")
            print("📌 Access the interface at: http://127.0.0.1:7860")
            time.sleep(1)
            from hybrid_web_interface import create_web_interface
            create_web_interface()
            
        elif args.mode == 'fullscreen':
            # Fullscreen mode (OpenCV)
            print("📺 Launching fullscreen interface...")
            time.sleep(1)
            from hybrid_ultimate_detection import run_hybrid_ultimate_detection_system
            run_hybrid_ultimate_detection_system()
            
        else:
            # CLI/Default mode
            print("💻 Launching standard interface...")
            time.sleep(1)
            from hybrid_ultimate_detection import run_hybrid_ultimate_detection_system
            run_hybrid_ultimate_detection_system()
    
    except KeyboardInterrupt:
        print("\n👋 Exiting Hybrid Ultimate Detection System...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("⚠️ Try running with --no-check if there are dependency issues")

def has_display():
    """Check if system has display capability for GUI"""
    # Simple check for display capability
    try:
        if sys.platform.startswith('win'):
            return True  # Assume Windows has a display
        else:
            return os.environ.get('DISPLAY') is not None
    except:
        return True  # Default to True if we can't determine

if __name__ == "__main__":
    main()
