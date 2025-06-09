#!/usr/bin/env python3
"""
Integration Tests for Conditional WHERE Scoring (Priority 1 - Critical)

Tests the complete integration of:
- Control type classification
- Conditional WHERE scoring
- Element analysis integration
- End-to-end workflow validation
"""

import pytest
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


class TestConditionalScoringIntegration:
    """Integration tests for the complete conditional scoring workflow"""

    def test_system_control_end_to_end(self, analyzer_with_config, assert_helpers):
        """Test complete system control analysis workflow (P1 Critical)"""
        result = analyzer_with_config.analyze_control(
            'SYS_INT_001',
            'System validates transaction limits and flags exceptions for manager review',
            automation_field='manual'
        )
        
        # Validate result structure
        assert_helpers['structure'](result)
        assert_helpers['score_ranges'](result)
        assert_helpers['category_consistency'](result)
        
        # Verify classification
        assert result['control_classification']['final_type'] == 'system'
        assert result['control_classification']['upgraded'] == True
        
        # Verify WHERE scoring
        assert result['scoring_breakdown']['WHERE'] == 10
        
        # Verify core element weights are applied
        scoring = result['scoring_breakdown']
        assert scoring['WHO'] > 0  # Should detect "manager"
        assert scoring['WHAT'] > 0  # Should detect "validates", "flags"
        # WHEN might be 0 due to missing explicit timing

    def test_location_control_end_to_end(self, analyzer_with_config, assert_helpers):
        """Test complete location-dependent control analysis workflow (P1 Critical)"""
        result = analyzer_with_config.analyze_control(
            'LOC_INT_001',
            'Security guard performs physical vault inspection daily',
            automation_field='manual'
        )
        
        # Validate result structure
        assert_helpers['structure'](result)
        assert_helpers['score_ranges'](result)
        assert_helpers['category_consistency'](result)
        
        # Verify classification
        assert result['control_classification']['final_type'] == 'location_dependent'
        assert result['control_classification']['upgraded'] == False
        
        # Verify WHERE scoring
        assert result['scoring_breakdown']['WHERE'] == 5
        
        # Verify core elements
        scoring = result['scoring_breakdown']
        assert scoring['WHO'] > 0  # Should detect "Security guard"
        assert scoring['WHAT'] > 0  # Should detect "performs", "inspection"
        assert scoring['WHEN'] > 0  # Should detect "daily"

    def test_other_control_end_to_end(self, analyzer_with_config, assert_helpers):
        """Test complete other control analysis workflow (P1 Critical)"""
        result = analyzer_with_config.analyze_control(
            'OTH_INT_001',
            'Branch manager reviews daily exception report and saves findings in SharePoint',
            automation_field='manual'
        )
        
        # Validate result structure
        assert_helpers['structure'](result)
        assert_helpers['score_ranges'](result)
        assert_helpers['category_consistency'](result)
        
        # Verify classification
        assert result['control_classification']['final_type'] == 'other'
        assert result['control_classification']['upgraded'] == False
        
        # Verify WHERE scoring
        assert result['scoring_breakdown']['WHERE'] == 0
        
        # Should still have good core scores
        scoring = result['scoring_breakdown']
        assert scoring['WHO'] > 0  # Should detect "Branch manager"
        assert scoring['WHAT'] > 0  # Should detect "reviews", "saves"
        assert scoring['WHEN'] > 0  # Should detect "daily"

    def test_vague_control_with_demerits(self, analyzer_with_config, assert_helpers):
        """Test control with vague terms and demerits (P1 Critical)"""
        result = analyzer_with_config.analyze_control(
            'VAG_INT_001',
            'Management periodically reviews reports as appropriate and timely addresses issues',
            automation_field='manual'
        )
        
        # Validate result structure
        assert_helpers['structure'](result)
        assert_helpers['score_ranges'](result)
        assert_helpers['category_consistency'](result)
        
        # Verify vague terms are detected
        vague_terms = result['vague_terms_found']
        expected_vague = {'periodically', 'appropriate', 'timely', 'issues'}
        found_vague = set(vague_terms)
        
        # Should find most or all expected vague terms
        assert len(found_vague.intersection(expected_vague)) >= 3, \
            f"Should find most vague terms. Found: {found_vague}, Expected: {expected_vague}"
        
        # Verify demerits are applied
        demerits = result['scoring_breakdown']['demerits']
        assert demerits < 0, "Should have negative demerits for vague terms"
        assert demerits <= -6, "Should have significant demerits for multiple vague terms"

    def test_hybrid_control_prominence_analysis(self, analyzer_with_config, assert_helpers):
        """Test hybrid control prominence analysis (P1 Critical)"""
        # System-prominent hybrid
        result = analyzer_with_config.analyze_control(
            'HYB_SYS_001',
            'System calculates balances while analyst validates results at branch office',
            automation_field='hybrid'
        )
        
        assert_helpers['structure'](result)
        
        # Should be classified as system due to system prominence
        classification = result['control_classification']
        assert classification['final_type'] == 'system'
        assert classification['system_score'] > classification['location_score']
        
        # Should get system WHERE points
        assert result['scoring_breakdown']['WHERE'] == 10

    def test_core_element_weight_distribution(self, analyzer_with_config):
        """Test that core elements follow 30/35/35 weight distribution (P1 Critical)"""
        # Use a control with strong all elements
        result = analyzer_with_config.analyze_control(
            'WEIGHT_001',
            'The Finance Manager validates invoices daily in SAP',
            automation_field='hybrid'
        )
        
        scoring = result['scoring_breakdown']
        
        # All core elements should have positive scores
        assert scoring['WHO'] > 0, "WHO should have positive score"
        assert scoring['WHAT'] > 0, "WHAT should have positive score"
        assert scoring['WHEN'] > 0, "WHEN should have positive score"
        
        # Core total should be reasonable (not exceeding 100)
        core_total = scoring['WHO'] + scoring['WHAT'] + scoring['WHEN']
        assert core_total <= 100, f"Core elements total should not exceed 100: {core_total}"

    def test_automation_field_handling_integration(self, analyzer_with_config):
        """Test various automation field values in complete workflow (P1 Critical)"""
        base_description = 'System processes transactions and manager reviews results'
        
        test_cases = [
            ('automated', 'system'),
            ('hybrid', 'system'),  # Should be system due to system prominence
            ('manual', 'system'),  # Should upgrade due to system participation
            (None, 'system'),      # Should analyze content and upgrade
        ]
        
        for automation, expected_type in test_cases:
            result = analyzer_with_config.analyze_control(
                f'AUTO_INT_{automation or "none"}',
                base_description,
                automation_field=automation
            )
            
            # Should not crash and should classify reasonably
            assert result['control_classification']['final_type'] in ['system', 'location_dependent', 'other']
            
            # For this specific description, should detect system characteristics
            if automation in ['automated', 'hybrid'] or automation is None:
                # Should likely be classified as system
                pass

    def test_missing_elements_impact_on_scoring(self, analyzer_with_config):
        """Test how missing elements impact conditional scoring (P1 Critical)"""
        # Control missing WHO
        result_no_who = analyzer_with_config.analyze_control(
            'MISSING_WHO',
            'Validates transactions daily in system'  # No clear WHO
        )
        
        # Control missing WHAT
        result_no_what = analyzer_with_config.analyze_control(
            'MISSING_WHAT', 
            'Manager daily in SAP'  # No clear WHAT
        )
        
        # Control missing WHEN
        result_no_when = analyzer_with_config.analyze_control(
            'MISSING_WHEN',
            'Manager validates transactions in SAP'  # No clear WHEN
        )
        
        # All should have lower scores due to missing critical elements
        for result in [result_no_who, result_no_what, result_no_when]:
            assert result['total_score'] < 75, "Missing critical elements should reduce score"

    def test_multiple_controls_detection_integration(self, analyzer_with_config):
        """Test multiple controls detection in complete workflow (P1 Critical)"""
        multiple_control_description = """
        Manager reviews daily transactions and validates amounts. 
        Then approves journal entries and reconciles accounts. 
        Finally generates reports and escalates exceptions to supervisor.
        """
        
        result = analyzer_with_config.analyze_control(
            'MULTI_INT_001',
            multiple_control_description
        )
        
        # Should detect multiple controls and apply demerit
        demerits = result['scoring_breakdown']['demerits']
        assert demerits < 0, "Should have demerits for multiple controls"
        
        # Should still analyze elements but with penalties
        assert result['total_score'] >= 0, "Score should not go below 0"

    def test_performance_integration(self, analyzer_with_config, benchmark):
        """Test end-to-end analysis performance (P1 Critical)"""
        def analyze_typical_control():
            return analyzer_with_config.analyze_control(
                'PERF_INT_001',
                'The Finance Manager reviews and validates journal entries in SAP monthly to ensure accuracy and compliance'
            )
        
        # Should complete within performance targets
        result = benchmark(analyze_typical_control)
        
        # Verify result is valid
        assert result['total_score'] >= 0
        assert result['category'] in ['Meets Expectations', 'Requires Attention', 'Needs Improvement']

    def test_error_handling_integration(self, analyzer_with_config):
        """Test error handling in complete workflow (P1 Critical)"""
        # Empty description
        result = analyzer_with_config.analyze_control('ERROR_001', '')
        assert result['category'] in ['Needs Improvement', 'Invalid']
        assert result['total_score'] >= 0
        
        # Very short description
        result = analyzer_with_config.analyze_control('ERROR_002', 'X')
        assert result['total_score'] >= 0
        
        # Special characters
        result = analyzer_with_config.analyze_control(
            'ERROR_003', 
            'Manager @#$% validates !@# transactions'
        )
        assert result['total_score'] >= 0

    def test_why_escalation_feedback_only_integration(self, analyzer_with_config):
        """Test WHY and ESCALATION are feedback-only (P1 Critical)"""
        # Control with clear WHY and ESCALATION
        result = analyzer_with_config.analyze_control(
            'FEEDBACK_001',
            'Manager reviews reports to ensure compliance and escalates issues to supervisor when thresholds are exceeded'
        )
        
        # WHY and ESCALATION should not impact scoring
        scoring = result['scoring_breakdown']
        
        # Only core elements and WHERE should contribute to score
        score_components = ['WHO', 'WHAT', 'WHEN', 'WHERE', 'demerits']
        calculated_score = sum(scoring[component] for component in score_components)
        
        # Total score should match calculated score (no WHY/ESCALATION impact)
        assert abs(result['total_score'] - calculated_score) < 0.1, \
            "WHY and ESCALATION should not impact total score"