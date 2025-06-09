#!/usr/bin/env python3
"""
Test script to verify WHERE percentage-based scoring implementation
"""

import sys
from src.core.analyzer import EnhancedControlAnalyzer

def test_where_scoring():
    # Initialize analyzer
    analyzer = EnhancedControlAnalyzer()
    
    # Test control examples
    test_controls = [
        ('System Control', 'The Finance Manager reconciles bank statements monthly in SAP to ensure accuracy.'),
        ('Location Control', 'The Accounting Supervisor reviews journal entries in the main office to verify completeness.'),
        ('Generic Control', 'Management reviews financial reports periodically to identify issues.')
    ]
    
    print('Testing new WHERE percentage-based scoring system:')
    print('=' * 70)
    
    for control_type, control_text in test_controls:
        print(f'\n{control_type}: {control_text}')
        result = analyzer.analyze_single_control(control_text, f'TEST_{control_type}')
        
        print('  Element Scores:')
        weighted_scores = result.get('weighted_scores', {})
        for element in ['WHO', 'WHAT', 'WHEN']:
            score = weighted_scores.get(element, 0)
            print(f'    {element}: {score:.1f}')
        
        where_impact = weighted_scores.get('WHERE', 0)
        final_score = result.get('total_score', 0)
        category = result.get('category', 'Unknown')
        detected_type = result.get('control_classification', {}).get('final_type', 'unknown')
        
        print(f'  WHERE Impact: {where_impact:.1f}')
        print(f'  Final Score: {final_score:.1f}')
        print(f'  Category: {category}')
        print(f'  Detected Type: {detected_type}')
        
        # Calculate and show the multiplier effect
        base_score = sum(weighted_scores.get(el, 0) for el in ['WHO', 'WHAT', 'WHEN'])
        if where_impact > 0:
            effective_multiplier = final_score / (final_score - where_impact) if (final_score - where_impact) > 0 else 1.0
            print(f'  Effective Multiplier: {effective_multiplier:.3f}x')

if __name__ == '__main__':
    test_where_scoring()