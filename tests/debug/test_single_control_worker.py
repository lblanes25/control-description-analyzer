#!/usr/bin/env python3
"""Test single control worker thread logic"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Initialize analyzer exactly like the GUI does
script_dir = os.path.dirname(os.path.abspath('src/gui/main_window.py'))
project_root = os.path.dirname(os.path.dirname(script_dir))
config_path = os.path.join(project_root, 'config', 'control_analyzer_updated.yaml')

analyzer = EnhancedControlAnalyzer(config_path)

# Simulate exactly what the worker thread does for single control
control_data = {
    'id': 'TEST001',
    'description': 'The Finance Manager reviews monthly statements',
    'frequency': None,
    'type': None,
    'risk': None
}

print("🧪 Simulating single control worker thread:")
print(f"Control data: {control_data}")

# This is exactly what the worker thread does
control_id = control_data.get('id', 'CONTROL-1')
description = control_data.get('description', '')
frequency = control_data.get('frequency')
control_type = control_data.get('type')
risk = control_data.get('risk')

print(f"\nExtracted values:")
print(f"  control_id: '{control_id}'")
print(f"  description: '{description}'")
print(f"  frequency: {frequency}")
print(f"  control_type: {control_type}")
print(f"  risk: {risk}")

try:
    result = analyzer.analyze_control(
        control_id, description, frequency, control_type, risk
    )
    results = [result]
    
    print(f"\n✅ Analysis successful!")
    print(f"Results count: {len(results)}")
    print(f"Result structure: {list(result.keys())}")
    print(f"WHO: {result.get('matched_keywords', {}).get('WHO', 'NOT FOUND')}")
    print(f"Total score: {result.get('total_score', 'NOT FOUND')}")
    
    # Test the filter logic on this result
    print(f"\n🔍 Testing filters:")
    category_filter = "All Categories"
    min_score = 0
    max_score = 100
    leader_filter = "All Leaders"
    
    # Apply same filter logic as GUI
    category_passes = category_filter == "All Categories" or result.get("category", "") == category_filter
    score = result.get("total_score", 0)
    score_passes = min_score <= score <= max_score
    
    if leader_filter != "All Leaders":
        result_leader = result.get("Audit Leader", result.get("metadata", {}).get("Audit Leader", "Unknown"))
        leader_passes = result_leader == leader_filter
    else:
        leader_passes = True
    
    print(f"  Category passes: {category_passes}")
    print(f"  Score passes: {score_passes}")
    print(f"  Leader passes: {leader_passes}")
    print(f"  All pass: {category_passes and score_passes and leader_passes}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()