#!/usr/bin/env python3
"""Test exactly what the GUI does"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Initialize analyzer exactly like the GUI does
script_dir = os.path.dirname(os.path.abspath('src/gui/main_window.py'))
project_root = os.path.dirname(os.path.dirname(script_dir))
config_path = os.path.join(project_root, 'config', 'control_analyzer_updated.yaml')

print(f"Config path: {config_path}")
print(f"Config exists: {os.path.exists(config_path)}")

try:
    analyzer = EnhancedControlAnalyzer(config_path)
    print("✅ Analyzer created successfully")
    
    # Test a control like the GUI would
    test_control_id = "TEST001"
    test_description = "The Finance Manager reviews and reconciles monthly bank statements within 5 business days."
    
    print(f"\n🧪 Testing control analysis:")
    print(f"ID: {test_control_id}")
    print(f"Description: {test_description}")
    
    # This is what the GUI calls
    result = analyzer.analyze_control(
        control_id=test_control_id,
        description=test_description
    )
    
    print(f"\n📊 Results:")
    print(f"WHO: {result.get('WHO', 'No WHO result')}")
    print(f"WHEN: {result.get('WHEN', 'No WHEN result')}")
    print(f"WHAT: {result.get('WHAT', 'No WHAT result')}")
    print(f"WHY: {result.get('WHY', 'No WHY result')}")
    print(f"Total Score: {result.get('total_score', 'No score')}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()