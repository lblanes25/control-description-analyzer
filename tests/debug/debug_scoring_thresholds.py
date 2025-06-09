#!/usr/bin/env python3
"""Debug scoring thresholds and weighted scores"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Initialize analyzer
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config', 'control_analyzer.yaml')
analyzer = EnhancedControlAnalyzer(config_path)

print("🔍 Debugging Scoring Thresholds")
print()

# Check the thresholds
element_thresholds = analyzer.config.get("scoring", {}).get("element_thresholds", {})
print("Element Thresholds from config:")
for element, threshold in element_thresholds.items():
    print(f"  {element}: {threshold}")

print()

# Test the problematic cases
test_cases = [
    {
        "id": "TEST1",
        "description": "Exceptions are routed to the appropriate team as needed. Access is restricted to authorized users. The system automatically ages receivables monthly."
    },
    {
        "id": "TEST4", 
        "description": "The Compliance Officer tests the user entitlements periodically to ensure compliance with internal policies."
    }
]

for test in test_cases:
    print(f"=== {test['id']} ===")
    print(f"Description: {test['description'][:60]}...")
    print()
    
    result = analyzer.analyze_control(test['id'], test['description'])
    
    # Get the actual weighted scores
    weighted_scores = result.get('weighted_scores', {})
    matched_keywords = result.get('matched_keywords', {})
    
    print("Detailed Scoring Analysis:")
    elements_above_threshold = 0
    
    for element in ['WHO', 'WHEN', 'WHAT', 'WHY', 'ESCALATION']:
        score = weighted_scores.get(element, 0)
        threshold = element_thresholds.get(element, 5.0)
        keywords = matched_keywords.get(element, [])
        
        above_threshold = score >= threshold
        if above_threshold:
            elements_above_threshold += 1
            
        print(f"  {element}:")
        print(f"    Score: {score:.2f} | Threshold: {threshold} | Above: {above_threshold}")
        print(f"    Keywords: {keywords}")
        print()
    
    print(f"Elements above threshold: {elements_above_threshold}")
    print(f"Reported elements found: {result.get('elements_found_count')}")
    print(f"Simple category: {result.get('simple_category')}")
    print(f"Missing elements: {result.get('missing_elements')}")
    print()
    print("-" * 80)
    print()