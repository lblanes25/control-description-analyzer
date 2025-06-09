#!/usr/bin/env python3
"""Test the simple scoring feature"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Initialize analyzer with config
script_dir = os.path.dirname(os.path.abspath('src/gui/main_window.py'))
project_root = os.path.dirname(os.path.dirname(script_dir))
config_path = os.path.join(project_root, 'config', 'control_analyzer.yaml')

analyzer = EnhancedControlAnalyzer(config_path)

# Test controls
test_controls = [
    {
        "id": "TEST001",
        "description": "The Finance Manager reviews and reconciles monthly bank statements within 5 business days of month-end to ensure accuracy and identify any unauthorized transactions. Discrepancies exceeding $10,000 are investigated and escalated to the CFO within 2 business days.",
        "expected_elements": 5,
        "expected_category": "Meets Expectations"
    },
    {
        "id": "TEST002", 
        "description": "The Accounting Supervisor reviews monthly journal entries prior to posting to ensure accuracy. Errors are returned to the preparer for correction.",
        "expected_elements": 4,
        "expected_category": "Meets Expectations"  # With threshold of 4
    },
    {
        "id": "TEST003",
        "description": "Management reviews financial statements periodically and addresses issues as appropriate.",
        "expected_elements": 2,  # Should find WHO and WHAT
        "expected_category": "Needs Improvement"
    }
]

print("🧪 Testing Simple Scoring Feature\n")
print(f"Configuration loaded from: {config_path}")
print(f"Simple scoring enabled: {analyzer.config.get('simple_scoring', {}).get('enabled', True)}")
print(f"Thresholds: Meets Expectations={analyzer.config.get('simple_scoring', {}).get('thresholds', {}).get('excellent', 4)}, "
      f"Requires Attention={analyzer.config.get('simple_scoring', {}).get('thresholds', {}).get('good', 3)}")
print("\n" + "="*80 + "\n")

for test in test_controls:
    result = analyzer.analyze_control(
        control_id=test["id"],
        description=test["description"]
    )
    
    print(f"Control ID: {test['id']}")
    print(f"Description: {test['description'][:100]}...")
    print(f"\nWeighted Scoring:")
    print(f"  Total Score: {result.get('total_score', 0):.1f}")
    print(f"  Category: {result.get('category', 'Unknown')}")
    
    print(f"\nSimple Scoring:")
    print(f"  Elements Found: {result.get('elements_found_count', 'N/A')}")
    print(f"  Simple Category: {result.get('simple_category', 'N/A')}")
    
    # Check if expectations are met
    elements_found = result.get('elements_found_count', '0/5')
    actual_count = int(elements_found.split('/')[0])
    
    if actual_count == test["expected_elements"]:
        print(f"  ✅ Element count matches expected: {test['expected_elements']}")
    else:
        print(f"  ❌ Element count mismatch - Expected: {test['expected_elements']}, Actual: {actual_count}")
    
    if result.get('simple_category') == test["expected_category"]:
        print(f"  ✅ Category matches expected: {test['expected_category']}")
    else:
        print(f"  ❌ Category mismatch - Expected: {test['expected_category']}, Actual: {result.get('simple_category')}")
    
    print(f"\nDetailed Element Detection:")
    matched = result.get('matched_keywords', {})
    for element in ['WHO', 'WHEN', 'WHAT', 'WHY', 'ESCALATION']:
        keywords = matched.get(element, [])
        status = "✓ Found" if keywords else "✗ Missing"
        print(f"  {element}: {status} {keywords if keywords else ''}")
    
    print("\n" + "-"*80 + "\n")

# Test with simple scoring disabled
print("\n🧪 Testing with Simple Scoring Disabled")
analyzer.config['simple_scoring']['enabled'] = False
result = analyzer.analyze_control("TEST004", "Staff reviews reports monthly")
print(f"Elements Found: '{result.get('elements_found_count', 'Should be empty')}'")
print(f"Simple Category: '{result.get('simple_category', 'Should be empty')}'")