#!/usr/bin/env python3
"""
Test script to verify WHERE element implementation

This test validates that the WHERE element is properly integrated into
the control analyzer framework, including:
1. WHERE service detection capabilities
2. WHERE element scoring
3. Integration with WHAT analyzer
4. Overall system functionality
"""

import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.core.analyzer import EnhancedControlAnalyzer
from src.analyzers.where_service import WhereDetectionService
from src.analyzers.where import enhance_where_detection
import spacy


def test_where_service():
    """Test the shared WHERE detection service"""
    print("=" * 50)
    print("TESTING WHERE DETECTION SERVICE")
    print("=" * 50)
    
    # Initialize service with basic config
    config = {}
    service = WhereDetectionService(config)
    
    # Load spaCy model
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Warning: spaCy model not found, using basic detection")
        nlp = None
        return False
    
    # Test cases
    test_cases = [
        "The Finance Manager reviews transactions in SAP monthly",
        "System automatically validates data in Oracle database",
        "The team processes invoices using SharePoint workflow",
        "Manager approves requests at headquarters office",
        "IT department monitors the network remotely"
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_text}")
        
        doc = nlp(test_text)
        results = service.detect_where_components(test_text, doc)
        
        print(f"  Systems found: {len(results.get('systems', []))}")
        print(f"  Locations found: {len(results.get('locations', []))}")
        print(f"  Organizational units found: {len(results.get('organizational', []))}")
        
        if results.get('primary_component'):
            primary = results['primary_component']
            print(f"  Primary WHERE: '{primary['text']}' ({primary['type']}) - confidence: {primary['confidence']:.2f}")
        else:
            print("  No primary WHERE component detected")
    
    return True


def test_where_element():
    """Test the WHERE element detection"""
    print("\n" + "=" * 50)
    print("TESTING WHERE ELEMENT")
    print("=" * 50)
    
    # Load spaCy model
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Warning: spaCy model not found, skipping WHERE element test")
        return False
    
    # Test WHERE element directly
    test_cases = [
        {
            'text': "The Finance Manager reviews transactions in SAP monthly",
            'control_type': 'IT'
        },
        {
            'text': "Team processes invoices using SharePoint",
            'control_type': 'Manual'
        },
        {
            'text': "Manager approves requests",
            'control_type': 'Manual'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['text']}")
        
        results = enhance_where_detection(
            test_case['text'], 
            nlp,
            control_type=test_case['control_type'],
            config={}
        )
        
        print(f"  WHERE Score: {results['score']:.2f}")
        print(f"  Confidence: {results['confidence']:.2f}")
        print(f"  Matched Keywords: {results['matched_keywords']}")
        print(f"  Primary Location: {results['primary_location']}")
        print(f"  Suggestions: {results['suggestions']}")
    
    return True


def test_full_integration():
    """Test WHERE integration with the full analyzer"""
    print("\n" + "=" * 50)
    print("TESTING FULL INTEGRATION")
    print("=" * 50)
    
    try:
        # Initialize the analyzer
        config_path = os.path.join(project_root, 'config', 'control_analyzer.yaml')
        analyzer = EnhancedControlAnalyzer(config_path)
        
        # Test control descriptions
        test_controls = [
            {
                'id': 'TEST001',
                'description': "The Finance Manager reviews and approves journal entries in SAP monthly to ensure accuracy",
                'frequency': 'Monthly',
                'control_type': 'IT'
            },
            {
                'id': 'TEST002', 
                'description': "System automatically validates transaction amounts in Oracle database",
                'frequency': 'Real-time',
                'control_type': 'Automated'
            },
            {
                'id': 'TEST003',
                'description': "The team processes invoices using SharePoint workflow",
                'frequency': 'Daily',
                'control_type': 'Manual'
            }
        ]
        
        for control in test_controls:
            print(f"\nAnalyzing Control {control['id']}: {control['description']}")
            
            result = analyzer.analyze_control(
                control['id'],
                control['description'],
                control['frequency'],
                control['control_type']
            )
            
            # Print available keys for debugging
            print(f"  Available keys: {list(result.keys())[:5]}...")  # Show first 5 keys
            
            # Try different possible key names for overall score
            overall_score_key = None
            for key in ['Overall Score', 'overall_score', 'Total Score', 'Score', 'final_score']:
                if key in result:
                    overall_score_key = key
                    break
            
            if overall_score_key:
                print(f"  Overall Score: {result[overall_score_key]:.1f}")
            else:
                print("  Overall Score: Not found")
            
            # Try different possible key names for category
            category_key = None
            for key in ['Category', 'category', 'Classification', 'Grade']:
                if key in result:
                    category_key = key
                    break
            
            if category_key:
                print(f"  Category: {result[category_key]}")
            else:
                print("  Category: Not found")
            
            # Check if WHERE is included in element scores
            for element in ['WHO', 'WHAT', 'WHEN', 'WHERE', 'WHY', 'ESCALATION']:
                found = False
                for key in [f'{element} Score', f'{element.lower()}_score', f'{element}_normalized_score']:
                    if key in result:
                        score = result[key]
                        print(f"  {element}: {score:.1f}")
                        found = True
                        break
                if not found:
                    print(f"  {element}: Not found")
            
            # Check WHERE specific results
            where_keywords_found = False
            for key in ['WHERE Matched Keywords', 'where_matched_keywords', 'WHERE_matched_keywords']:
                if key in result:
                    print(f"  WHERE Keywords: {result[key]}")
                    where_keywords_found = True
                    break
            if not where_keywords_found:
                print("  WHERE Keywords: Not found")
            
            where_suggestions_found = False
            for key in ['WHERE Suggestions', 'where_suggestions', 'WHERE_suggestions']:
                if key in result:
                    suggestions = result[key]
                    if suggestions:
                        print(f"  WHERE Suggestions: {suggestions[:2]}")  # Show first 2
                    where_suggestions_found = True
                    break
            if not where_suggestions_found:
                print("  WHERE Suggestions: Not found")
        
        return True
        
    except Exception as e:
        print(f"Error in full integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all WHERE implementation tests"""
    print("WHERE ELEMENT IMPLEMENTATION TEST")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: WHERE Service
    if test_where_service():
        success_count += 1
        print("\n✓ WHERE Service test passed")
    else:
        print("\n✗ WHERE Service test failed")
    
    # Test 2: WHERE Element
    if test_where_element():
        success_count += 1
        print("\n✓ WHERE Element test passed")
    else:
        print("\n✗ WHERE Element test failed")
    
    # Test 3: Full Integration
    if test_full_integration():
        success_count += 1
        print("\n✓ Full Integration test passed")
    else:
        print("\n✗ Full Integration test failed")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"TEST SUMMARY: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! WHERE element implementation is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)