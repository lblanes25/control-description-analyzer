#!/usr/bin/env python3
"""Debug staff detection specifically"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Initialize analyzer
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config', 'control_analyzer.yaml')
analyzer = EnhancedControlAnalyzer(config_path)

test_description = "Staff reviews reports periodically and addresses issues as appropriate."

print(f"Testing: {test_description}")
print()

result = analyzer.analyze_control("TEST", test_description)

# Get detailed scores and keywords
weighted_scores = result.get('weighted_scores', {})
normalized_scores = result.get('normalized_scores', {}) 
matched_keywords = result.get('matched_keywords', {})
missing_elements = result.get('missing_elements', [])

print("Detailed Analysis:")
for element in ['WHO', 'WHEN', 'WHAT', 'WHY', 'ESCALATION']:
    weighted_score = weighted_scores.get(element, 0)
    normalized_score = normalized_scores.get(element, 0)
    keywords = matched_keywords.get(element, [])
    is_missing = element in missing_elements
    
    print(f"{element}:")
    print(f"  Keywords: {keywords}")
    print(f"  Normalized Score: {normalized_score:.2f}")
    print(f"  Weighted Score: {weighted_score:.2f}")
    print(f"  Missing: {is_missing}")
    print(f"  Has Keywords + Below Threshold: {bool(keywords and weighted_score < 5.0)}")
    print()

# Check if WHO should get subthreshold feedback
who_keywords = matched_keywords.get('WHO', [])
who_score = weighted_scores.get('WHO', 0)
print(f"WHO subthreshold check: Has keywords ({bool(who_keywords)}) + Score < 5.0 ({who_score < 5.0}) = {bool(who_keywords and who_score < 5.0)}")

enhancement_feedback = result.get('enhancement_feedback', {})
print(f"\nWHO Enhancement Feedback: {enhancement_feedback.get('WHO', 'None')}")