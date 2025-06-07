#!/usr/bin/env python3
"""Debug the ControlElement analysis"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Initialize analyzer
script_dir = os.path.dirname(os.path.abspath('src/gui/main_window.py'))
project_root = os.path.dirname(os.path.dirname(script_dir))
config_path = os.path.join(project_root, 'config', 'control_analyzer_updated.yaml')

analyzer = EnhancedControlAnalyzer(config_path)
test_text = "The Finance Manager reviews monthly statements"

print(f"🧪 Testing WHO element analysis:")
print(f"Text: {test_text}")

# Get the WHO element
who_element = analyzer.elements["WHO"]
print(f"\nBefore analysis:")
print(f"  Score: {who_element.score}")
print(f"  Normalized score: {who_element.normalized_score}")
print(f"  Enhanced results: {who_element.enhanced_results}")
print(f"  Matched keywords: {who_element.matched_keywords}")

# Run the element analysis
context = {
    "control_type": None,
    "frequency": None,
    "risk_description": None,
    "analyzer_config": analyzer.config,
    "config_adapter": analyzer.config_adapter
}

who_element.analyze(test_text, analyzer.nlp, True, **context)

print(f"\nAfter analysis:")
print(f"  Score: {who_element.score}")
print(f"  Normalized score: {who_element.normalized_score}")
print(f"  Enhanced results: {who_element.enhanced_results}")
print(f"  Matched keywords: {who_element.matched_keywords}")

# Test what the full analyzer returns
print(f"\n🧪 Full analyzer test:")
result = analyzer.analyze_control("TEST001", test_text)
print(f"WHO in result: {result.get('WHO', 'NOT FOUND')}")
print(f"Total score: {result.get('total_score')}")
print(f"Weighted scores: {result.get('weighted_scores', {})}")