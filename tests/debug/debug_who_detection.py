#!/usr/bin/env python3
"""Debug WHO detection specifically"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Initialize analyzer
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config', 'control_analyzer.yaml')
analyzer = EnhancedControlAnalyzer(config_path)

# Test the specific case that's failing
test_description = "The Compliance Officer tests the user entitlements periodically to ensure compliance with internal policies."

print("🔍 Debugging WHO Detection")
print(f"Description: {test_description}")
print()

# Get the WHO element analyzer
who_element = analyzer.elements.get("WHO")
if who_element:
    print(f"WHO Element type: {type(who_element)}")
    
    # Run WHO analysis directly with nlp parameter
    who_result = who_element.analyze(test_description, analyzer.nlp)
    print(f"WHO Analysis Result: {who_result}")
    print()
    
    # Check confidence scoring
    if hasattr(who_element, 'confidence_scores'):
        print("WHO Confidence Scores:")
        for key, value in who_element.confidence_scores.items():
            print(f"  {key}: {value}")
    
    # Check enhanced results if available
    if hasattr(who_element, 'enhanced_results'):
        print("WHO Enhanced Results:")
        for key, value in who_element.enhanced_results.items():
            print(f"  {key}: {value}")
    
    print()

# Test a few more problematic cases
test_cases = [
    "The Compliance Officer tests the user entitlements periodically to ensure compliance with internal policies.",
    "The Finance Manager reviews monthly reports.",
    "Management reviews financial statements.",
    "Staff performs reconciliations."
]

print("Testing multiple WHO detection cases:")
for i, desc in enumerate(test_cases, 1):
    result = analyzer.analyze_control(f"TEST{i}", desc)
    who_keywords = result.get('matched_keywords', {}).get('WHO', [])
    missing = result.get('missing_elements', [])
    who_missing = 'WHO' in missing
    
    print(f"Test {i}: {desc[:50]}...")
    print(f"  WHO Keywords: {who_keywords}")
    print(f"  WHO Missing: {who_missing}")
    print(f"  Elements Found: {result.get('elements_found_count')}")
    print()