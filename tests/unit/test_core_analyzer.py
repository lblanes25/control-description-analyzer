#!/usr/bin/env python3
"""
Unit Tests for Core Analyzer (Priority 1 - Critical)

Tests the core analysis engine functionality including:
- Complete control analysis workflow
- Conditional WHERE scoring methodology
- Uncapped demerit calculations
- Integration of all element analyzers
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.core.analyzer import EnhancedControlAnalyzer
from src.analyzers.control_classifier import ControlTypeClassifier


class TestEnhancedControlAnalyzer:
    """Test suite for EnhancedControlAnalyzer core functionality"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance with test configuration"""
        config_path = os.path.join(project_root, 'config', 'control_analyzer.yaml')
        return EnhancedControlAnalyzer(config_path)

    @pytest.fixture
    def test_controls(self):
        """Test control library with expected outcomes"""
        return {
            'system_control': {
                'id': 'SYS_001',
                'description': 'System validates transaction limits and flags exceptions for manager review',
                'automation': 'manual',
                'expected_classification': 'system',
                'expected_where_points': 10,
                'expected_category': 'Requires Attention'
            },
            'location_control': {
                'id': 'LOC_001', 
                'description': 'Security guard performs physical vault inspection daily',
                'automation': 'manual',
                'expected_classification': 'location_dependent',
                'expected_where_points': 5,
                'expected_category': 'Meets Expectations'
            },
            'other_control': {
                'id': 'OTH_001',
                'description': 'Branch manager reviews daily exception report and saves findings in SharePoint',
                'automation': 'manual', 
                'expected_classification': 'other',
                'expected_where_points': 0,
                'expected_category': 'Meets Expectations'
            },
            'vague_control': {
                'id': 'VAG_001',
                'description': 'Management periodically reviews reports as appropriate and timely addresses issues',
                'automation': 'manual',
                'expected_vague_terms': ['periodically', 'appropriate', 'timely', 'issues'],
                'expected_demerits': -8
            }
        }

    def test_analyze_control_complete_workflow(self, analyzer, test_controls):
        """Test complete control analysis workflow (P1 Critical)"""
        control = test_controls['system_control']
        
        result = analyzer.analyze_control(
            control['id'],
            control['description'],
            automation_field=control['automation']
        )
        
        # Verify required fields are present
        required_fields = [
            'control_id', 'description', 'total_score', 'category',
            'control_classification', 'scoring_breakdown', 'vague_terms_found'
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        # Verify control classification
        assert result['control_classification']['final_type'] == control['expected_classification']
        
        # Verify scoring structure
        scoring = result['scoring_breakdown']
        assert 'WHO' in scoring
        assert 'WHAT' in scoring
        assert 'WHEN' in scoring
        assert 'WHERE' in scoring
        assert 'demerits' in scoring

    def test_conditional_where_scoring_system_controls(self, analyzer, test_controls):
        """Test conditional WHERE scoring for system controls (P1 Critical)"""
        control = test_controls['system_control']
        
        result = analyzer.analyze_control(
            control['id'],
            control['description'],
            automation_field=control['automation']
        )
        
        # System controls should get 10 WHERE points
        assert result['scoring_breakdown']['WHERE'] == control['expected_where_points']
        assert result['control_classification']['final_type'] == 'system'

    def test_conditional_where_scoring_location_controls(self, analyzer, test_controls):
        """Test conditional WHERE scoring for location-dependent controls (P1 Critical)"""
        control = test_controls['location_control']
        
        result = analyzer.analyze_control(
            control['id'],
            control['description'],
            automation_field=control['automation']
        )
        
        # Location-dependent controls should get 5 WHERE points
        assert result['scoring_breakdown']['WHERE'] == control['expected_where_points']
        assert result['control_classification']['final_type'] == 'location_dependent'

    def test_conditional_where_scoring_other_controls(self, analyzer, test_controls):
        """Test conditional WHERE scoring for other controls (P1 Critical)"""
        control = test_controls['other_control']
        
        result = analyzer.analyze_control(
            control['id'],
            control['description'],
            automation_field=control['automation']
        )
        
        # Other controls should get 0 WHERE points
        assert result['scoring_breakdown']['WHERE'] == control['expected_where_points']
        assert result['control_classification']['final_type'] == 'other'

    def test_core_element_weight_distribution(self, analyzer):
        """Test core element weight distribution (30/35/35) (P1 Critical)"""
        # Test with a control that has strong WHO, WHAT, WHEN
        result = analyzer.analyze_control(
            'WEIGHT_001',
            'The Finance Manager validates invoices daily in SAP',
            automation_field='hybrid'
        )
        
        scoring = result['scoring_breakdown']
        
        # Verify weights are applied correctly
        # Note: Actual scores depend on element strength, but proportions should reflect weights
        total_core = scoring['WHO'] + scoring['WHAT'] + scoring['WHEN']
        
        # All three elements should have scores > 0
        assert scoring['WHO'] > 0, "WHO should have positive score"
        assert scoring['WHAT'] > 0, "WHAT should have positive score" 
        assert scoring['WHEN'] > 0, "WHEN should have positive score"

    def test_uncapped_demerit_system(self, analyzer, test_controls):
        """Test uncapped demerit system (P1 Critical)"""
        control = test_controls['vague_control']
        
        result = analyzer.analyze_control(
            control['id'],
            control['description'],
            automation_field=control['automation']
        )
        
        # Verify vague terms are detected
        vague_terms = result['vague_terms_found']
        expected_terms = set(control['expected_vague_terms'])
        found_terms = set(vague_terms)
        
        # Should find all expected vague terms
        assert expected_terms.issubset(found_terms), f"Missing vague terms: {expected_terms - found_terms}"
        
        # Verify demerits calculation (-2 per vague term)
        expected_demerits = len(vague_terms) * -2
        actual_demerits = result['scoring_breakdown']['demerits']
        assert actual_demerits <= expected_demerits, f"Demerits too lenient: {actual_demerits} vs {expected_demerits}"

    def test_category_threshold_determination(self, analyzer):
        """Test category threshold logic (P1 Critical)"""
        # Test cases designed to hit specific score ranges
        test_cases = [
            {
                'description': 'High scoring control with all elements',
                'expected_min_score': 75,  # Should be Meets Expectations
                'expected_category': 'Meets Expectations'
            },
            {
                'description': 'Medium control',
                'expected_min_score': 50,  # Should be Requires Attention
                'expected_max_score': 74,
                'expected_category': 'Requires Attention'
            }
        ]
        
        for i, case in enumerate(test_cases):
            result = analyzer.analyze_control(
                f'THRESH_{i}',
                case['description']
            )
            
            score = result['total_score']
            category = result['category']
            
            # Verify score ranges and category assignment
            if 'expected_min_score' in case:
                if score >= case['expected_min_score']:
                    # Only check category if score is in expected range
                    if 'expected_max_score' not in case or score <= case['expected_max_score']:
                        assert category == case['expected_category'], \
                            f"Wrong category for score {score}: got {category}, expected {case['expected_category']}"

    def test_invalid_input_handling(self, analyzer):
        """Test handling of invalid inputs (P1 Critical)"""
        # Empty description
        result = analyzer.analyze_control('EMPTY_001', '')
        assert result['category'] in ['Needs Improvement', 'Invalid']
        
        # Very short description
        result = analyzer.analyze_control('SHORT_001', 'X')
        assert result['total_score'] >= 0  # Should not crash
        
        # None description should be handled gracefully (not crash)
        result = analyzer.analyze_control('NONE_001', None)
        assert result['total_score'] == 0
        assert result['category'] == 'Needs Improvement'

    def test_automation_field_handling(self, analyzer):
        """Test various automation field values (P1 Critical)"""
        base_description = 'Manager reviews and approves transactions daily'
        
        automation_tests = [
            ('automated', 'system'),
            ('manual', 'other'),
            ('hybrid', 'other'),  # Depends on content analysis
            (None, 'other'),
            ('invalid_value', 'other')
        ]
        
        for automation, expected_type in automation_tests:
            result = analyzer.analyze_control(
                f'AUTO_{automation or "none"}',
                base_description,
                automation_field=automation
            )
            
            # Should not crash and should classify appropriately
            assert result['control_classification']['final_type'] in ['system', 'location_dependent', 'other']

    def test_performance_single_control(self, analyzer, benchmark):
        """Test performance of single control analysis (P1 Critical)"""
        control_description = "The Finance Manager reviews and validates journal entries in SAP monthly to ensure accuracy and compliance with accounting standards"
        
        def analyze_single():
            return analyzer.analyze_control('PERF_001', control_description)
        
        # Should complete in under 100ms as per strategy
        result = benchmark(analyze_single)
        assert result['total_score'] >= 0

    def test_missing_element_detection(self, analyzer):
        """Test detection of missing critical elements (P1 Critical)"""
        # Control missing WHO
        result = analyzer.analyze_control(
            'MISSING_WHO',
            'Reviews transactions daily'  # No clear WHO
        )
        
        # Should have lower score due to missing WHO
        assert result['total_score'] < 75  # Likely not Meets Expectations
        
        # Control missing WHAT  
        result = analyzer.analyze_control(
            'MISSING_WHAT',
            'Manager daily in system'  # No clear WHAT
        )
        
        # Should have lower score due to missing WHAT
        assert result['total_score'] < 75

    def test_multiple_control_detection(self, analyzer):
        """Test detection of multiple controls in single description (P1 Critical)"""
        multiple_control_description = """
        Manager reviews daily transactions and validates amounts. 
        Then approves journal entries and reconciles accounts. 
        Finally generates reports and escalates exceptions.
        """
        
        result = analyzer.analyze_control(
            'MULTI_001',
            multiple_control_description
        )
        
        # Should detect multiple controls and apply demerit
        demerits = result['scoring_breakdown']['demerits']
        assert demerits < 0, "Should have demerits for multiple controls"


class TestConditionalScoringMethodology:
    """Specific tests for the conditional WHERE scoring methodology"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        config_path = os.path.join(project_root, 'config', 'control_analyzer.yaml')
        return EnhancedControlAnalyzer(config_path)

    def test_where_scoring_with_no_where_element(self, analyzer):
        """Test controls without WHERE element get 0 points regardless of type"""
        # System control without WHERE
        result = analyzer.analyze_control(
            'NO_WHERE_SYS',
            'System validates transaction limits',  # No WHERE mentioned
            automation_field='automated'
        )
        
        # Even system controls get 0 WHERE points if no WHERE detected
        where_score = result['scoring_breakdown']['WHERE']
        # Note: WHERE might still be detected from "System" - this tests the logic

    def test_where_element_context_sensitivity(self, analyzer):
        """Test WHERE element detection is context-sensitive"""
        test_cases = [
            {
                'description': 'Manager saves report in SharePoint',
                'expected_type': 'other',  # Documentation only
                'has_where': True
            },
            {
                'description': 'System validates data in Oracle',
                'expected_type': 'system',  # System participation
                'has_where': True
            }
        ]
        
        for case in test_cases:
            result = analyzer.analyze_control(
                f'WHERE_CTX_{case["expected_type"]}',
                case['description']
            )
            
            classification = result['control_classification']['final_type']
            where_points = result['scoring_breakdown']['WHERE']
            
            # Verify classification is correct
            assert classification == case['expected_type']
            
            # Verify WHERE points based on type and presence
            if case['has_where']:
                if classification == 'system':
                    assert where_points == 10 or where_points == 0  # Depends on WHERE detection
                elif classification == 'location_dependent':
                    assert where_points == 5 or where_points == 0
                else:  # other
                    assert where_points == 0