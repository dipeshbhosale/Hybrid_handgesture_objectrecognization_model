#!/usr/bin/env python3
"""
Test script for Hybrid Ultimate Detection System web interface
"""

import sys
import os
import time

def test_web_interface():
    """Test launching the web interface"""
    print("\n=== Testing Web Interface ===")
    
    try:
        # Import the function
        from hybrid_web_interface import create_web_interface
        print("✅ Successfully imported web interface module")
        
        print("📌 Web interface should be accessible at: http://127.0.0.1:7860")
        print("ℹ️ Press Ctrl+C to stop the web interface after testing")
        
        # Launch the interface
        create_web_interface()
        
        return True
    except Exception as e:
        print(f"❌ Error launching web interface: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n==== HYBRID WEB INTERFACE - TEST ====")
    test_web_interface()
