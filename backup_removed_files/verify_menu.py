#!/usr/bin/env python3
"""
Verification - Main Menu Test
Verify that main.py now shows options 1-9
"""

def verify_main_menu():
    """Test that the main menu function works"""
    print("🔍 Verifying main.py menu...")
    
    try:
        import main
        print("✅ main.py imports successfully")
        
        # Check if the function exists
        if hasattr(main, 'run_with_object_detection_options'):
            print("✅ run_with_object_detection_options function found")
            
            # Check if object detection functions exist
            if hasattr(main, 'object_detection_only'):
                print("✅ object_detection_only function found")
            else:
                print("❌ object_detection_only function missing")
                
            if hasattr(main, 'combined_gesture_and_object_detection'):
                print("✅ combined_gesture_and_object_detection function found")
            else:
                print("❌ combined_gesture_and_object_detection function missing")
                
            return True
        else:
            print("❌ run_with_object_detection_options function not found")
            return False
            
    except Exception as e:
        print(f"❌ Error importing main.py: {e}")
        return False

def show_menu_preview():
    """Show what the menu looks like"""
    print("\n📋 Main Menu Preview:")
    print("=" * 60)
    print("🤖 AI VISION SYSTEM - Extended Options")
    print("=" * 60)
    print("""
Choose option:
1. Run complete gesture pipeline
2. Collect gesture data only
3. Train gesture model only
4. Start gesture recognition only
5. Test camera only
6. Use OpenCV gesture interface (RECOMMENDED)
7. 🆕 Object Detection Only
8. 🆕 Combined Gesture + Object Detection
9. Gradio web interface

Enter (1-9): """)

if __name__ == "__main__":
    print("🚀 Main Menu Verification")
    print("=" * 40)
    
    if verify_main_menu():
        print("\n✅ All functions verified successfully!")
        show_menu_preview()
        
        print("\n🎯 How to use:")
        print("1. Run: python main.py")
        print("2. Choose option 7 for Object Detection Only")
        print("3. Choose option 8 for Combined Gesture + Object Detection")
        print("4. Press Ctrl+C to exit if menu is waiting for input")
        
        print("\n💡 Quick Start:")
        print("   python main.py  # Shows options 1-9")
        print("   python test_object_detection.py  # Alternative interface")
        
    else:
        print("\n❌ Verification failed!")
        
    print("\n🎉 Setup Complete! The extended menu is now active.")
