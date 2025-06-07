#!/usr/bin/env python3
"""Debug feedback handling to see why some are blank vs None"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Initialize analyzer
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config', 'control_analyzer_updated.yaml')
analyzer = EnhancedControlAnalyzer(config_path)

print("🔍 Debugging Feedback Handling")
print()

# Test the same control as in our Excel test
test_description = "The Finance Manager reviews and reconciles monthly bank statements within 5 business days to ensure accuracy. Discrepancies exceeding $10,000 are escalated to the CFO."

result = analyzer.analyze_control("EXCELLENT001", test_description)

enhancement_feedback = result.get('enhancement_feedback', {})

print("Raw Enhancement Feedback Structure:")
for element in ['WHO', 'WHEN', 'WHAT', 'WHY', 'ESCALATION']:
    feedback = enhancement_feedback.get(element)
    print(f"  {element}:")
    print(f"    Type: {type(feedback)}")
    print(f"    Value: {repr(feedback)}")
    print(f"    Is None: {feedback is None}")
    print(f"    Is Empty List: {feedback == []}")
    print()

print("=" * 60)
print()

# Now check how this gets processed in the DataFrame creation
print("DataFrame Processing Logic Test (FIXED):")

# Simulate the NEW feedback processing from _prepare_report_data
for element in ["WHO", "WHEN", "WHAT", "WHY", "ESCALATION"]:
    feedback = enhancement_feedback.get(element, None)
    
    # This is the NEW logic from the method
    if isinstance(feedback, list):
        if feedback:  # Non-empty list
            processed = "; ".join(str(item) for item in feedback)
        else:  # Empty list
            processed = "None"
    elif isinstance(feedback, str):
        if feedback.strip():  # Non-empty string
            processed = feedback
        else:  # Empty or whitespace-only string
            processed = "None"
    else:  # None or other types
        processed = "None"
    
    print(f"  {element}: {repr(feedback)} → {repr(processed)}")

print()
print("🎯 Issue Analysis:")
print("- If feedback is an empty list [], it should show 'None'")
print("- If feedback is None, it should show 'None'") 
print("- If feedback is a string, it should show the string")
print("- We need consistent 'None' handling across all cases")