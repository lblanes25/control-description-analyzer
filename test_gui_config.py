#!/usr/bin/env python3
"""Test GUI config loading"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Test the config path that GUI will use
script_dir = os.path.dirname(os.path.abspath('src/gui/main_window.py'))
project_root = os.path.dirname(os.path.dirname(script_dir))
config_path = os.path.join(project_root, 'config', 'control_analyzer_updated.yaml')

print(f'Config path: {config_path}')
print(f'Exists: {os.path.exists(config_path)}')

if os.path.exists(config_path):
    try:
        analyzer = EnhancedControlAnalyzer(config_path)
        print('✅ Analyzer created successfully')
        print(f'WHO keywords count: {len(analyzer.elements["WHO"].keywords)}')
        print(f'WHO keywords sample: {analyzer.elements["WHO"].keywords[:3]}')
        
        # Test a simple WHO detection
        test_text = "The Finance Manager reviews monthly statements"
        result = analyzer.analyze_single_control("TEST001", test_text)
        print(f'WHO detection result: {result.get("WHO", {}).get("primary", "No result")}')
        
    except Exception as e:
        print(f'❌ Error creating analyzer: {e}')
        import traceback
        traceback.print_exc()
else:
    print("❌ Config file not found")