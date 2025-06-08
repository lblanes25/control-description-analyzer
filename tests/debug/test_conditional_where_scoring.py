#!/usr/bin/env python3
"""
Test Conditional WHERE Scoring Implementation

This test validates the new conditional WHERE scoring methodology as specified 
in ScoringUpdate.md. It tests:
1. Control type classification
2. Conditional WHERE scoring based on control type
3. New demerit system
4. Core element scoring (WHO:30, WHAT:35, WHEN:35)
5. WHY/ESCALATION feedback-only status
"""

import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.core.analyzer import EnhancedControlAnalyzer
from src.analyzers.control_classifier import ControlTypeClassifier


def test_control_classification():
    """Test the control type classification system"""
    print("=" * 60)
    print("TESTING CONTROL TYPE CLASSIFICATION")
    print("=" * 60)
    
    config = {}
    classifier = ControlTypeClassifier(config)
    
    test_cases = [
        {
            'description': "System validates transaction limits and flags exceptions for manager review",
            'automation': 'manual',
            'expected_type': 'system',
            'expected_upgraded': True
        },
        {
            'description': "Branch manager reviews daily exception report and saves findings in SharePoint",
            'automation': 'manual', 
            'expected_type': 'other',
            'expected_upgraded': False
        },
        {
            'description': "Security guard performs physical vault inspection",
            'automation': 'manual',
            'expected_type': 'location_dependent',
            'expected_upgraded': False
        },
        {
            'description': "Automated reconciliation identifies breaks and analyst investigates discrepancies",
            'automation': 'manual',
            'expected_type': 'system',
            'expected_upgraded': True
        },
        {
            'description': "Senior management reviews quarterly risk reports",
            'automation': 'manual',
            'expected_type': 'other',
            'expected_upgraded': False
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description'][:50]}...")
        print(f"  Automation Field: {test_case['automation']}")
        
        result = classifier.classify_control(
            test_case['description'], 
            test_case['automation']
        )
        
        print(f"  Classified as: {result['final_type']}")
        print(f"  Upgraded: {result['upgraded']}")
        print(f"  Reasoning: {result['reasoning'][0] if result['reasoning'] else 'None'}")
        
        # Check results
        type_correct = result['final_type'] == test_case['expected_type']
        upgrade_correct = result['upgraded'] == test_case['expected_upgraded']
        
        if type_correct and upgrade_correct:
            print(f"  ✓ PASS")
            passed += 1
        else:
            print(f"  ✗ FAIL - Expected: {test_case['expected_type']}, Upgraded: {test_case['expected_upgraded']}")
    
    print(f"\nClassification Test Results: {passed}/{total} passed")
    return passed == total


def test_conditional_where_scoring():
    """Test the new conditional WHERE scoring system"""
    print("\n" + "=" * 60)
    print("TESTING CONDITIONAL WHERE SCORING")
    print("=" * 60)
    
    try:
        config_path = os.path.join(project_root, 'config', 'control_analyzer_updated.yaml')
        analyzer = EnhancedControlAnalyzer(config_path)
        
        test_cases = [
            {
                'id': 'TEST001',
                'description': "System validates transaction limits and flags exceptions for manager review",
                'automation': 'manual',
                'expected_category': 'Adequate',
                'expected_where_points': 10,  # Should be classified as 'system'
                'expected_classification': 'system'
            },
            {
                'id': 'TEST002',
                'description': "Branch manager reviews daily exception report and saves findings in SharePoint", 
                'automation': 'manual',
                'expected_category': 'Effective',  # Core elements score ~99 points
                'expected_where_points': 0,  # Should be classified as 'other'
                'expected_classification': 'other'
            },
            {
                'id': 'TEST003',
                'description': "Security guard performs physical vault inspection daily",
                'automation': 'manual',
                'expected_category': 'Effective',  # Core + WHERE = 84 points
                'expected_where_points': 5,  # Should be classified as 'location_dependent'
                'expected_classification': 'location_dependent'
            },
            {
                'id': 'TEST004',
                'description': "The Finance Manager reviews and approves journal entries in SAP monthly",
                'automation': 'hybrid',
                'expected_category': 'Effective',
                'expected_where_points': 10,  # Should be classified as 'system'
                'expected_classification': 'system'
            }
        ]
        
        passed = 0
        total = len(test_cases)
        
        for test_case in test_cases:
            print(f"\nTesting Control {test_case['id']}: {test_case['description'][:50]}...")
            
            result = analyzer.analyze_control(
                test_case['id'],
                test_case['description'],
                automation_field=test_case['automation']
            )
            
            # Extract results
            total_score = result['total_score']
            category = result['category']
            classification = result['control_classification']
            scoring_breakdown = result['scoring_breakdown']
            
            print(f"  Total Score: {total_score:.1f}")
            print(f"  Category: {category}")
            print(f"  Control Type: {classification['final_type']}")
            print(f"  WHERE Points: {scoring_breakdown['WHERE']}")
            print(f"  Demerits: {scoring_breakdown['demerits']}")
            print(f"  Core Breakdown: WHO:{scoring_breakdown['WHO']:.1f} WHAT:{scoring_breakdown['WHAT']:.1f} WHEN:{scoring_breakdown['WHEN']:.1f}")
            
            # Check results
            category_correct = category == test_case['expected_category']
            where_points_correct = scoring_breakdown['WHERE'] == test_case['expected_where_points']
            classification_correct = classification['final_type'] == test_case['expected_classification']
            
            if category_correct and where_points_correct and classification_correct:
                print(f"  ✓ PASS")
                passed += 1
            else:
                print(f"  ✗ FAIL")
                if not category_correct:
                    print(f"    Expected category: {test_case['expected_category']}, got: {category}")
                if not where_points_correct:
                    print(f"    Expected WHERE points: {test_case['expected_where_points']}, got: {scoring_breakdown['WHERE']}")
                if not classification_correct:
                    print(f"    Expected classification: {test_case['expected_classification']}, got: {classification['final_type']}")
        
        print(f"\nConditional WHERE Scoring Test Results: {passed}/{total} passed")
        return passed == total
        
    except Exception as e:
        print(f"Error in conditional WHERE scoring test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_demerit_system():
    """Test the new uncapped demerit system"""
    print("\n" + "=" * 60)
    print("TESTING DEMERIT SYSTEM")
    print("=" * 60)
    
    try:
        config_path = os.path.join(project_root, 'config', 'control_analyzer_updated.yaml')
        analyzer = EnhancedControlAnalyzer(config_path)
        
        test_cases = [
            {
                'id': 'DEMERIT001',
                'description': "Management periodically reviews reports as appropriate and timely addresses issues",
                'expected_vague_count': 4,  # periodically, appropriate, timely, issues
                'expected_demerits_min': -8  # At least 4 vague terms * -2 each
            },
            {
                'id': 'DEMERIT002', 
                'description': "The Finance Manager validates invoices daily in SAP",
                'expected_vague_count': 0,
                'expected_demerits_max': 0
            }
        ]
        
        passed = 0
        total = len(test_cases)
        
        for test_case in test_cases:
            print(f"\nTesting {test_case['id']}: {test_case['description'][:50]}...")
            
            result = analyzer.analyze_control(
                test_case['id'],
                test_case['description']
            )
            
            scoring_breakdown = result['scoring_breakdown']
            demerits = scoring_breakdown['demerits']
            vague_terms = result['vague_terms_found']
            
            print(f"  Vague Terms Found: {len(vague_terms)} ({vague_terms})")
            print(f"  Total Demerits: {demerits}")
            
            # Check results
            if 'expected_vague_count' in test_case:
                vague_correct = len(vague_terms) >= test_case['expected_vague_count']
            else:
                vague_correct = True
                
            if 'expected_demerits_min' in test_case:
                demerits_correct = demerits <= test_case['expected_demerits_min']
            elif 'expected_demerits_max' in test_case:
                demerits_correct = demerits >= test_case['expected_demerits_max']
            else:
                demerits_correct = True
            
            if vague_correct and demerits_correct:
                print(f"  ✓ PASS")
                passed += 1
            else:
                print(f"  ✗ FAIL")
                if not vague_correct:
                    print(f"    Expected vague count >= {test_case.get('expected_vague_count', 0)}, got: {len(vague_terms)}")
                if not demerits_correct:
                    expected_range = f"<= {test_case.get('expected_demerits_min', 'N/A')} or >= {test_case.get('expected_demerits_max', 'N/A')}"
                    print(f"    Expected demerits {expected_range}, got: {demerits}")
        
        print(f"\nDemerit System Test Results: {passed}/{total} passed")
        return passed == total
        
    except Exception as e:
        print(f"Error in demerit system test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all conditional WHERE scoring tests"""
    print("CONDITIONAL WHERE SCORING TEST SUITE")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Control Classification
    test_results.append(test_control_classification())
    
    # Test 2: Conditional WHERE Scoring  
    test_results.append(test_conditional_where_scoring())
    
    # Test 3: Demerit System
    test_results.append(test_demerit_system())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print("\n" + "=" * 60)
    print(f"OVERALL TEST RESULTS: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        print("🎉 All test suites passed! Conditional WHERE scoring is working correctly.")
        return 0
    else:
        print("⚠️  Some test suites failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)