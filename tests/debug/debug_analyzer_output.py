#!/usr/bin/env python3
"""
Debug script to test the analyzer output format and identify GUI display issues.
"""

import sys
import os
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.analyzer import EnhancedControlAnalyzer

def test_analyzer_output():
    """Test what the analyzer actually returns"""
    
    # Initialize analyzer
    config_path = os.path.join(os.path.dirname(__file__), "config", "control_analyzer.yaml")
    if os.path.exists(config_path):
        analyzer = EnhancedControlAnalyzer(config_path)
    else:
        fallback_config = os.path.join(os.path.dirname(__file__), "config", "control_analyzer.yaml") 
        if os.path.exists(fallback_config):
            analyzer = EnhancedControlAnalyzer(fallback_config)
        else:
            analyzer = EnhancedControlAnalyzer()
    
    # Test with a sample control description
    test_description = "The Finance Manager reviews the monthly reconciliation between the subledger and general ledger by the 5th business day of the following month."
    
    print("Testing analyzer with sample control...")
    print(f"Description: {test_description}")
    print("-" * 80)
    
    # Analyze the control
    result = analyzer.analyze_control("TEST-001", test_description, "Monthly", "Detective")
    
    print("Result structure:")
    for key, value in result.items():
        print(f"  {key}: {type(value).__name__}")
        if key in ["matched_keywords", "weighted_scores", "normalized_scores"]:
            print(f"    Value: {value}")
        elif key in ["total_score", "category"]:
            print(f"    Value: {value}")
    
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS:")
    
    print(f"\nControl ID: {result.get('control_id')}")
    print(f"Total Score: {result.get('total_score')}")
    print(f"Category: {result.get('category')}")
    
    print(f"\nMatched Keywords:")
    matched_keywords = result.get('matched_keywords', {})
    for element, keywords in matched_keywords.items():
        print(f"  {element}: {keywords}")
    
    print(f"\nWeighted Scores:")
    weighted_scores = result.get('weighted_scores', {})
    for element, score in weighted_scores.items():
        print(f"  {element}: {score}")
        
    print(f"\nNormalized Scores:")
    normalized_scores = result.get('normalized_scores', {})
    for element, score in normalized_scores.items():
        print(f"  {element}: {score}")
    
    print(f"\nMissing Elements: {result.get('missing_elements', [])}")
    print(f"Vague Terms: {result.get('vague_terms_found', [])}")
    
    # Test if the GUI processing would work
    print("\n" + "=" * 80)
    print("GUI PROCESSING TEST:")
    
    print("Simulating GUI table population...")
    
    # Test the exact code from apply_result_filters
    normalized_scores = result.get("normalized_scores", {})
    weighted_scores = result.get("weighted_scores", {})
    
    print(f"WHO (normalized): {normalized_scores.get('WHO', 0):.1f}")
    print(f"WHEN (normalized): {normalized_scores.get('WHEN', 0):.1f}")
    print(f"WHAT (normalized): {normalized_scores.get('WHAT', 0):.1f}")
    print(f"WHY (normalized): {normalized_scores.get('WHY', 0):.1f}")
    print(f"ESCALATION (normalized): {normalized_scores.get('ESCALATION', 0):.1f}")
    
    print(f"\nWHO (weighted): {weighted_scores.get('WHO', 0):.1f}")
    print(f"WHEN (weighted): {weighted_scores.get('WHEN', 0):.1f}")
    print(f"WHAT (weighted): {weighted_scores.get('WHAT', 0):.1f}")
    print(f"WHY (weighted): {weighted_scores.get('WHY', 0):.1f}")
    print(f"ESCALATION (weighted): {weighted_scores.get('ESCALATION', 0):.1f}")
    
    return result

if __name__ == "__main__":
    test_analyzer_output()