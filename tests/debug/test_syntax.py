#!/usr/bin/env python3
"""Test syntax of who.py file"""

import sys
sys.path.append('.')

try:
    print("Testing import of who.py...")
    from src.analyzers.who import enhanced_who_detection_v2
    print("✅ Import successful")
    
    print("Testing import of analyzer.py...")
    from src.core.analyzer import EnhancedControlAnalyzer
    print("✅ Analyzer import successful")
    
    print("Testing analyzer initialization...")
    analyzer = EnhancedControlAnalyzer()
    print("✅ Analyzer initialization successful")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()