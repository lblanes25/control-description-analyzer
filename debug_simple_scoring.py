#!/usr/bin/env python3
"""Debug script to test and fix simple scoring implementation"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Initialize analyzer
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config', 'control_analyzer_updated.yaml')
analyzer = EnhancedControlAnalyzer(config_path)

# Test cases that show issues in your results
test_cases = [
    {
        "id": "TEST1",
        "description": "Exceptions are routed to the appropriate team as needed. Access is restricted to authorized users. The system automatically ages receivables monthly.",
        "expected_elements": 5,
        "expected_category": "Meets expectations",
        "reported_elements": "5/5",
        "reported_category": "Meets expectations",
        "issue": "Missing WHEN but shows 5/5"
    },
    {
        "id": "TEST2",
        "description": "The Operations Team monitors the monthly reconciliations within defined timelines to ensure compliance with internal policies.",
        "expected_elements": 4,
        "expected_category": "Meets expectations",
        "reported_elements": "4/5",
        "reported_category": "Meets expectations", 
        "issue": "Missing ESCALATION correctly"
    },
    {
        "id": "TEST3",
        "description": "To ensure accuracy, management monitors the reports.",
        "expected_elements": 3,
        "expected_category": "Requires Attention",
        "reported_elements": "3/5",
        "reported_category": "Requires Attention",
        "issue": "Correct - WHO, WHAT, WHY present"
    },
    {
        "id": "TEST4",
        "description": "The Compliance Officer tests the user entitlements periodically to ensure compliance with internal policies.",
        "expected_elements": 4,  # Should have WHO, WHAT, WHEN (periodically), WHY
        "expected_category": "Meets expectations",
        "reported_elements": "4/5",
        "reported_category": "Meets expectations",
        "issue": "Shows correct count but missing elements list is wrong"
    }
]

print("🔍 Debugging Simple Scoring Implementation\n")
print(f"Config: {config_path}")
print(f"Simple scoring enabled: {analyzer.config.get('simple_scoring', {}).get('enabled', True)}")
print(f"Thresholds: excellent={analyzer.config.get('simple_scoring', {}).get('thresholds', {}).get('excellent', 4)}, "
      f"good={analyzer.config.get('simple_scoring', {}).get('thresholds', {}).get('good', 3)}")
print("\n" + "="*100 + "\n")

# Analyze each test case
for test in test_cases:
    print(f"Test Case: {test['id']}")
    print(f"Description: {test['description'][:80]}...")
    print(f"Issue: {test['issue']}")
    print()
    
    result = analyzer.analyze_control(
        control_id=test['id'],
        description=test['description']
    )
    
    # Get the matched keywords to understand element detection
    matched_keywords = result.get('matched_keywords', {})
    missing_elements = result.get('missing_elements', [])
    
    print("Element Detection Analysis:")
    elements_found = 0
    for element in ['WHO', 'WHEN', 'WHAT', 'WHY', 'ESCALATION']:
        keywords = matched_keywords.get(element, [])
        if keywords:
            elements_found += 1
            print(f"  ✓ {element}: {keywords}")
        else:
            print(f"  ✗ {element}: Not found")
    
    print(f"\nActual elements found: {elements_found}")
    print(f"Missing elements list: {missing_elements}")
    print(f"Elements found count: {result.get('elements_found_count', 'N/A')}")
    print(f"Simple category: {result.get('simple_category', 'N/A')}")
    
    # Check if the count matches
    elements_count_str = result.get('elements_found_count', '')
    if elements_count_str:
        reported_count = int(elements_count_str.split('/')[0])
        if reported_count != elements_found:
            print(f"❌ MISMATCH: Reported {reported_count} but actually found {elements_found}")
        else:
            print(f"✅ Count matches: {elements_found}")
    
    # Check category assignment
    simple_config = analyzer.config.get('simple_scoring', {})
    thresholds = simple_config.get('thresholds', {})
    excellent_threshold = thresholds.get('excellent', 4)
    good_threshold = thresholds.get('good', 3)
    
    if elements_found >= excellent_threshold:
        expected_cat = simple_config.get('category_names', {}).get('excellent', 'Excellent')
    elif elements_found >= good_threshold:
        expected_cat = simple_config.get('category_names', {}).get('good', 'Good')
    else:
        expected_cat = simple_config.get('category_names', {}).get('needs_improvement', 'Needs Improvement')
    
    actual_cat = result.get('simple_category', '')
    if actual_cat != expected_cat:
        print(f"❌ CATEGORY MISMATCH: Expected '{expected_cat}' but got '{actual_cat}'")
    else:
        print(f"✅ Category correct: {actual_cat}")
    
    print("\n" + "-"*100 + "\n")

# Now let's check the specific issue with missing elements calculation
print("\n🔍 Checking Missing Elements Calculation Logic\n")

# Test a control that should have all elements
full_control = {
    "id": "FULL",
    "description": "The Finance Manager reviews and reconciles monthly bank statements within 5 business days to ensure accuracy. Discrepancies exceeding $10,000 are escalated to the CFO."
}

result = analyzer.analyze_control(full_control['id'], full_control['description'])
print(f"Full control test:")
print(f"Elements found: {result.get('elements_found_count')}")
print(f"Missing elements: {result.get('missing_elements', [])}")
print(f"Simple category: {result.get('simple_category')}")

# Check the matched_keywords structure
print("\nMatched keywords structure:")
for element, keywords in result.get('matched_keywords', {}).items():
    print(f"  {element}: {keywords}")