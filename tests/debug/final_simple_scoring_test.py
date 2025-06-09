#!/usr/bin/env python3
"""Final test of simple scoring implementation"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Initialize analyzer
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config', 'control_analyzer.yaml')
analyzer = EnhancedControlAnalyzer(config_path)

print("🎯 Final Simple Scoring Test")
print()

# Test cases with expected results
test_cases = [
    {
        "id": "EXCELLENT", 
        "description": "The Finance Manager reviews and reconciles monthly bank statements within 5 business days to ensure accuracy. Discrepancies exceeding $10,000 are escalated to the CFO.",
        "expected": "Should have 5/5 elements - WHO, WHEN, WHAT, WHY, ESCALATION all present with good specificity"
    },
    {
        "id": "GOOD",
        "description": "The Operations Team monitors the monthly reconciliations within defined timelines to ensure compliance with internal policies.",
        "expected": "Should have 4/5 elements - missing ESCALATION"
    },
    {
        "id": "NEEDS_IMPROVEMENT",
        "description": "Management reviews financial statements periodically and addresses issues as appropriate.",
        "expected": "Should have ~2-3/5 elements - vague timing (periodically) and vague escalation (as appropriate)"
    }
]

for test in test_cases:
    print(f"=== {test['id']} ===")
    print(f"Description: {test['description']}")
    print(f"Expected: {test['expected']}")
    print()
    
    result = analyzer.analyze_control(test['id'], test['description'])
    
    print("Results:")
    print(f"  Elements Found: {result.get('elements_found_count')}")
    print(f"  Simple Category: {result.get('simple_category')}")
    print(f"  Missing Elements: {result.get('missing_elements')}")
    print(f"  Total Score: {result.get('total_score'):.1f}")
    print(f"  Traditional Category: {result.get('category')}")
    
    # Show which elements scored above threshold
    weighted_scores = result.get('weighted_scores', {})
    print(f"  Elements above threshold (≥5.0):")
    for element in ['WHO', 'WHEN', 'WHAT', 'WHY', 'ESCALATION']:
        score = weighted_scores.get(element, 0)
        above = "✓" if score >= 5.0 else "✗"
        print(f"    {above} {element}: {score:.1f}")
    
    print()
    print("-" * 80)
    print()

print("✅ Simple scoring implementation test complete!")
print()
print("Summary:")
print("- Elements Found count = number of elements with weighted score ≥ 5.0")
print("- Missing Elements list = elements with weighted score < 5.0") 
print("- Simple Category based on thresholds: 4+ = Meets Expectations, 3+ = Requires Attention, <3 = Needs Improvement")
print("- Keywords detected but penalized for vagueness still count as 'detected' but may not meet scoring threshold")