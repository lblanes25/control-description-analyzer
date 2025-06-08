#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

from src.core.analyzer import EnhancedControlAnalyzer

# Create analyzer
analyzer = EnhancedControlAnalyzer('config/control_analyzer_updated.yaml')

# Test WHY impact
print("=== WHY Element Impact Test ===")
control_with_why = """
Finance Manager reviews journal entries daily to ensure compliance 
with SOX requirements and maintain accurate financial records
"""
control_without_why = "Finance Manager reviews journal entries daily"

result_with = analyzer.analyze_control('WHY_WITH', control_with_why)
result_without = analyzer.analyze_control('WHY_WITHOUT', control_without_why)

print(f"With WHY: {result_with['total_score']:.2f}")
print(f"Without WHY: {result_without['total_score']:.2f}")
print(f"Difference: {abs(result_with['total_score'] - result_without['total_score']):.2f}")

print("\nScoring Breakdown WITH WHY:")
for element, score in result_with['scoring_breakdown'].items():
    print(f"  {element}: {score}")

print("\nScoring Breakdown WITHOUT WHY:")
for element, score in result_without['scoring_breakdown'].items():
    print(f"  {element}: {score}")

# Test ESCALATION impact
print("\n=== ESCALATION Element Impact Test ===")
control_with_escalation = """
Manager reviews exception reports daily and escalates significant 
variances to the Controller for resolution
"""
control_without_escalation = "Manager reviews exception reports daily"

result_with_esc = analyzer.analyze_control('ESC_WITH', control_with_escalation)
result_without_esc = analyzer.analyze_control('ESC_WITHOUT', control_without_escalation)

print(f"With ESCALATION: {result_with_esc['total_score']:.2f}")
print(f"Without ESCALATION: {result_without_esc['total_score']:.2f}")
print(f"Difference: {abs(result_with_esc['total_score'] - result_without_esc['total_score']):.2f}")

print("\nScoring Breakdown WITH ESCALATION:")
for element, score in result_with_esc['scoring_breakdown'].items():
    print(f"  {element}: {score}")

print("\nScoring Breakdown WITHOUT ESCALATION:")
for element, score in result_without_esc['scoring_breakdown'].items():
    print(f"  {element}: {score}")