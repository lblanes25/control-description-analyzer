#!/usr/bin/env python3
"""Debug GUI result format"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Initialize analyzer
script_dir = os.path.dirname(os.path.abspath('src/gui/main_window.py'))
project_root = os.path.dirname(os.path.dirname(script_dir))
config_path = os.path.join(project_root, 'config', 'control_analyzer.yaml')

analyzer = EnhancedControlAnalyzer(config_path)
test_text = "The Finance Manager reviews monthly statements"

result = analyzer.analyze_control("TEST001", test_text)

print("🔍 Full result structure:")
for key, value in result.items():
    print(f"  {key}: {value}")

print(f"\n🔍 Specific WHO-related fields:")
print(f"  matched_keywords['WHO']: {result.get('matched_keywords', {}).get('WHO', 'NOT FOUND')}")
print(f"  enhancement_feedback['WHO']: {result.get('enhancement_feedback', {}).get('WHO', 'NOT FOUND')}")
print(f"  weighted_scores['WHO']: {result.get('weighted_scores', {}).get('WHO', 'NOT FOUND')}")

# Check the WHO element directly
who_element = analyzer.elements["WHO"]
print(f"\n🔍 WHO element after analysis:")
print(f"  enhanced_results: {who_element.enhanced_results}")
print(f"  Primary WHO: {who_element.enhanced_results.get('primary', {}).get('text', 'NOT FOUND')}")