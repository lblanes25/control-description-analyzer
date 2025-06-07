#!/usr/bin/env python3
"""Debug the analyzer's element initialization"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Initialize analyzer exactly like the GUI does
script_dir = os.path.dirname(os.path.abspath('src/gui/main_window.py'))
project_root = os.path.dirname(os.path.dirname(script_dir))
config_path = os.path.join(project_root, 'config', 'control_analyzer_updated.yaml')

try:
    analyzer = EnhancedControlAnalyzer(config_path)
    print("✅ Analyzer created successfully")
    
    # Debug element initialization
    print(f"\n🔍 Element Analysis:")
    for name, element in analyzer.elements.items():
        print(f"{name}:")
        print(f"  Weight: {element.weight}")
        print(f"  Keywords count: {len(element.keywords)}")
        print(f"  Keywords sample: {element.keywords[:3] if element.keywords else 'None'}")
    
    # Test WHO detection directly with keywords
    from src.analyzers.who import enhanced_who_detection_v2
    import spacy
    
    nlp = spacy.load('en_core_web_md')
    test_text = "The Finance Manager reviews monthly statements"
    
    print(f"\n🧪 Testing WHO detection with analyzer keywords:")
    who_element = analyzer.elements["WHO"]
    result = enhanced_who_detection_v2(
        test_text, nlp, None, None, who_element.keywords, analyzer.config_adapter
    )
    print(f"  Result: {result.get('primary', {}).get('text', 'None')}")
    print(f"  Confidence: {result.get('confidence', 0)}")
    
    print(f"\n🧪 Testing WHO detection without keywords:")
    result2 = enhanced_who_detection_v2(
        test_text, nlp, None, None, [], analyzer.config_adapter
    )
    print(f"  Result: {result2.get('primary', {}).get('text', 'None')}")
    print(f"  Confidence: {result2.get('confidence', 0)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()