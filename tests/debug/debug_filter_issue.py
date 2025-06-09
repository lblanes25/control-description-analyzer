#!/usr/bin/env python3
"""Debug the filter issue"""

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

print("🔍 Testing GUI filter logic:")
print(f"Result category: '{result.get('category', '')}'")
print(f"Result total_score: {result.get('total_score', 0)}")
print(f"Result Audit Leader: '{result.get('Audit Leader', 'NOT FOUND')}'")

# Simulate filter checks
category_filter = "All Categories"
min_score = 0
max_score = 100
leader_filter = "All Leaders"

print(f"\n🧪 Filter simulation:")
print(f"Category filter: '{category_filter}'")
print(f"Min/Max score: {min_score}-{max_score}")
print(f"Leader filter: '{leader_filter}'")

# Check category filter
category_passes = category_filter == "All Categories" or result.get("category", "") == category_filter
print(f"Category passes: {category_passes}")

# Check score filter
score = result.get("total_score", 0)
score_passes = min_score <= score <= max_score
print(f"Score passes: {score_passes} (score: {score})")

# Check audit leader filter
if leader_filter != "All Leaders":
    result_leader = result.get("Audit Leader", result.get("metadata", {}).get("Audit Leader", "Unknown"))
    leader_passes = result_leader == leader_filter
    print(f"Leader passes: {leader_passes} (result_leader: '{result_leader}', filter: '{leader_filter}')")
else:
    leader_passes = True
    print(f"Leader passes: {leader_passes} (filter is 'All Leaders')")

all_pass = category_passes and score_passes and leader_passes
print(f"\n✅ All filters pass: {all_pass}")

# Test what happens if leader filter is not "All Leaders"
print(f"\n🧪 Testing with leader filter != 'All Leaders':")
leader_filter = "Some Leader"
result_leader = result.get("Audit Leader", result.get("metadata", {}).get("Audit Leader", "Unknown"))
leader_passes = result_leader == leader_filter
print(f"Result leader: '{result_leader}'")
print(f"Filter leader: '{leader_filter}'")
print(f"Leader passes: {leader_passes}")