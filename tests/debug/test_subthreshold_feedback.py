#!/usr/bin/env python3
"""Test the enhanced subthreshold feedback system"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os
import json

# Initialize analyzer
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config', 'control_analyzer_updated.yaml')
analyzer = EnhancedControlAnalyzer(config_path)

print("🔍 Testing Enhanced Subthreshold Feedback System")
print()

# Test cases that should trigger subthreshold feedback
test_cases = [
    {
        "id": "VAGUE_TIMING",
        "description": "The Compliance Officer tests the user entitlements periodically to ensure compliance with internal policies.",
        "expected_subthreshold": ["WHEN"],
        "note": "WHEN should have 'periodically' detected but penalized for vagueness"
    },
    {
        "id": "VAGUE_MULTIPLE", 
        "description": "Staff reviews reports periodically and addresses issues as appropriate.",
        "expected_subthreshold": ["WHO", "WHEN", "ESCALATION"],
        "note": "Multiple elements should have content but score below threshold"
    },
    {
        "id": "CLEAR_CONTROL",
        "description": "The Finance Manager reconciles monthly bank statements within 5 business days to ensure accuracy. Discrepancies exceeding $10,000 are escalated to the CFO.",
        "expected_subthreshold": [],
        "note": "All elements should score above threshold - no subthreshold feedback expected"
    }
]

for test in test_cases:
    print(f"=== {test['id']} ===")
    print(f"Description: {test['description']}")
    print(f"Expected subthreshold elements: {test['expected_subthreshold']}")
    print(f"Note: {test['note']}")
    print()
    
    result = analyzer.analyze_control(test['id'], test['description'])
    
    # Display basic results
    print("Results:")
    print(f"  Elements Found: {result.get('elements_found_count')}")
    print(f"  Simple Category: {result.get('simple_category')}")
    print(f"  Missing Elements: {result.get('missing_elements')}")
    print()
    
    # Check for subthreshold feedback
    enhancement_feedback = result.get('enhancement_feedback', {})
    print("Enhancement Feedback:")
    
    subthreshold_found = []
    for element in ['WHO', 'WHEN', 'WHAT', 'WHY', 'ESCALATION']:
        feedback = enhancement_feedback.get(element, [])
        
        # Look for subthreshold messages (check for both "below threshold" and "too vague to meet scoring threshold")
        subthreshold_messages = []
        if isinstance(feedback, list):
            subthreshold_messages = [msg for msg in feedback if ("below threshold" in str(msg) or "too vague to meet scoring threshold" in str(msg))]
        elif isinstance(feedback, str) and ("below threshold" in feedback or "too vague to meet scoring threshold" in feedback):
            subthreshold_messages = [feedback]
        
        if subthreshold_messages:
            subthreshold_found.append(element)
            print(f"  {element}:")
            for msg in subthreshold_messages:
                print(f"    - {msg}")
        elif feedback:
            print(f"  {element}: {feedback}")
    
    if not any(enhancement_feedback.values()):
        print("  (No feedback generated)")
    
    print()
    
    # Verify expectations
    if set(subthreshold_found) == set(test['expected_subthreshold']):
        print("✅ Subthreshold feedback matches expectations")
    else:
        print(f"❌ Mismatch - Expected: {test['expected_subthreshold']}, Got: {subthreshold_found}")
    
    print()
    print("-" * 80)
    print()

print("✅ Subthreshold feedback system test complete!")