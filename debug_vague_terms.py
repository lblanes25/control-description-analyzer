#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

from src.core.analyzer import EnhancedControlAnalyzer

# Create analyzer
analyzer = EnhancedControlAnalyzer('config/control_analyzer_updated.yaml')

# Test the exact vague control from the failing test
vague_control = """
Management periodically reviews reports as appropriate and ensures 
timely resolution of issues while maintaining adequate oversight
"""

result = analyzer.analyze_control('VAGUE_DEBUG', vague_control)

print("=== Vague Term Detection Debug ===")
print(f"Control text: {vague_control.strip()}")
print(f"Vague terms found: {result['vague_terms_found']}")
print(f"Number of vague terms: {len(result['vague_terms_found'])}")
print(f"Demerits applied: {result['scoring_breakdown']['demerits']}")
print(f"Expected demerits: {-2 * len(result['vague_terms_found'])}")

expected_vague = {'periodically', 'appropriate', 'timely', 'issues', 'adequate'}
found_vague = set(result['vague_terms_found'])
print(f"Expected terms: {expected_vague}")
print(f"Found terms: {found_vague}")
print(f"Overlap: {found_vague.intersection(expected_vague)}")
print(f"Extra terms found: {found_vague - expected_vague}")
print(f"Missing terms: {expected_vague - found_vague}")