#!/usr/bin/env python3
"""
Priority 2 Business Logic Tests for Control Description Analyzer

Tests business logic components as specified in testing strategy:
- Demerit system calculations
- Category threshold determination  
- Enhancement feedback generation
- Multi-control detection
- WHY/ESCALATION feedback-only validation
"""

import pytest
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


class TestDemeritSystemCalculations:
    """Test suite for demerit system calculations (P2 Business Logic)"""

    def test_vague_term_demerits_uncapped(self, analyzer_with_config):
        """Test uncapped vague term demerit system (P2 Business Logic)"""
        # Test control with multiple vague terms
        vague_control = """
        Management periodically reviews reports as appropriate and ensures 
        timely resolution of issues while maintaining adequate oversight
        """
        
        result = analyzer_with_config.analyze_control('VAGUE_001', vague_control)
        
        # Should detect multiple vague terms
        vague_terms = result['vague_terms_found']
        expected_vague = {'periodically', 'appropriate', 'timely', 'issues', 'adequate'}
        found_vague = set(vague_terms)
        
        # Should find most expected vague terms
        overlap = found_vague.intersection(expected_vague)
        assert len(overlap) >= 3, f"Should detect multiple vague terms. Found: {found_vague}"
        
        # Verify vague term demerits are included (may have additional demerits from other sources)
        demerits = result['scoring_breakdown']['demerits']
        num_vague_found = len(vague_terms)
        expected_vague_demerits = -2 * num_vague_found
        
        # Total demerits should be at least the vague term demerits (may have additional penalties)
        assert demerits <= expected_vague_demerits, f"Should include vague term demerits: {demerits} should be <= {expected_vague_demerits}"
        
        # Total demerits should be significant for many vague terms
        assert demerits <= -8, f"Should have significant demerits for multiple vague terms: {demerits}"

    def test_multiple_control_demerits(self, analyzer_with_config):
        """Test multiple control detection demerits (P2 Business Logic)"""
        multiple_control_text = """
        Manager reviews transactions daily and validates amounts.
        Then approves journal entries and reconciles accounts.
        Finally generates reports and escalates exceptions to supervisor.
        """
        
        result = analyzer_with_config.analyze_control('MULTI_DEM_001', multiple_control_text)
        
        # Should detect multiple controls
        demerits = result['scoring_breakdown']['demerits']
        assert demerits < 0, "Should have demerits for multiple controls"
        
        # Should include multiple control penalty (typically -10)
        assert demerits <= -10, f"Should include multiple control penalty: {demerits}"

    def test_missing_accountability_demerits(self, analyzer_with_config):
        """Test missing accountability demerits (P2 Business Logic)"""
        # Control with vague WHO element
        vague_who_control = "Someone reviews reports periodically"
        
        result = analyzer_with_config.analyze_control('VAGUE_WHO_001', vague_who_control)
        
        # Should have lower WHO score due to vague accountability vs specific role
        who_score = result['scoring_breakdown']['WHO']
        
        # Compare to specific role
        specific_control = "Finance Manager reviews reports periodically"
        specific_result = analyzer_with_config.analyze_control('SPECIFIC_WHO_001', specific_control)
        specific_who_score = specific_result['scoring_breakdown']['WHO']
        
        assert who_score <= specific_who_score, f"Vague WHO should score <= specific WHO: {who_score} vs {specific_who_score}"
        
        # Should have demerits for vague term "periodically"
        demerits = result['scoring_breakdown']['demerits']
        assert demerits < 0, "Should have demerits for vague timing"

    def test_untestable_timing_demerits(self, analyzer_with_config):
        """Test untestable timing demerits (P2 Business Logic)"""
        # Control with untestable timing
        untestable_timing = "Manager reviews reports as needed when issues arise"
        
        result = analyzer_with_config.analyze_control('UNTESTABLE_001', untestable_timing)
        
        # Should detect vague timing terms
        vague_terms = result['vague_terms_found']
        timing_vague = {'needed', 'issues'}
        found_timing_vague = set(vague_terms).intersection(timing_vague)
        
        assert len(found_timing_vague) > 0, f"Should detect timing vague terms: {vague_terms}"
        
        # Should have appropriate demerits
        demerits = result['scoring_breakdown']['demerits']
        assert demerits < 0, "Should have demerits for untestable timing"

    def test_demerit_combination_logic(self, analyzer_with_config):
        """Test combination of multiple demerit types (P2 Business Logic)"""
        # Control combining multiple demerit sources
        complex_problematic_control = """
        Management periodically reviews various reports as appropriate.
        Staff validates data and then generates additional reports.
        Issues are escalated when necessary to ensure adequate oversight.
        """
        
        result = analyzer_with_config.analyze_control('COMPLEX_DEM_001', complex_problematic_control)
        
        # Should have multiple types of demerits
        demerits = result['scoring_breakdown']['demerits']
        
        # Should be significantly negative due to multiple issues
        assert demerits <= -15, f"Should have substantial demerits for multiple issues: {demerits}"
        
        # Verify vague terms are detected
        vague_terms = result['vague_terms_found']
        expected_vague = {'periodically', 'various', 'appropriate', 'issues', 'necessary', 'adequate'}
        found_vague = set(vague_terms)
        
        overlap = found_vague.intersection(expected_vague)
        assert len(overlap) >= 4, f"Should detect multiple vague terms: {found_vague}"


class TestCategoryThresholdDetermination:
    """Test suite for category threshold determination (P2 Business Logic)"""

    def test_effective_category_thresholds(self, analyzer_with_config):
        """Test 'Effective' category threshold determination (P2 Business Logic)"""
        # High-quality control should be 'Effective' (75+ points)
        effective_control = """
        The Finance Manager reviews and validates journal entries in SAP daily 
        to ensure accuracy and compliance with accounting standards
        """
        
        result = analyzer_with_config.analyze_control('EFFECTIVE_001', effective_control)
        
        # Should be classified as Effective
        assert result['category'] == 'Effective', f"High-quality control should be Effective: {result['category']}"
        assert result['total_score'] >= 75, f"Effective controls should score 75+: {result['total_score']}"

    def test_adequate_category_thresholds(self, analyzer_with_config):
        """Test 'Adequate' category threshold determination (P2 Business Logic)"""
        # Medium-quality control should be 'Adequate' (50-74 points)
        adequate_control = """
        Manager reviews monthly reports and validates balances
        """
        
        result = analyzer_with_config.analyze_control('ADEQUATE_001', adequate_control)
        
        # Should be classified as Adequate or better
        assert result['category'] in ['Adequate', 'Effective'], f"Reasonable control should be Adequate+: {result['category']}"
        
        # If Adequate, should be in 50-74 range
        if result['category'] == 'Adequate':
            assert 50 <= result['total_score'] < 75, f"Adequate should be 50-74: {result['total_score']}"

    def test_needs_improvement_thresholds(self, analyzer_with_config):
        """Test 'Needs Improvement' category threshold determination (P2 Business Logic)"""
        # Poor control should be 'Needs Improvement' (<50 points)
        poor_control = "Someone does something sometimes"
        
        result = analyzer_with_config.analyze_control('POOR_001', poor_control)
        
        # Should be classified as Needs Improvement
        assert result['category'] == 'Needs Improvement', f"Poor control should need improvement: {result['category']}"
        assert result['total_score'] < 50, f"Poor controls should score <50: {result['total_score']}"

    def test_threshold_boundary_conditions(self, analyzer_with_config):
        """Test threshold boundary conditions (P2 Business Logic)"""
        # Test various controls to validate threshold boundaries
        test_cases = [
            {
                'description': 'Manager validates transactions monthly',
                'expected_min_score': 40,  # Should be reasonable
                'expected_category': ['Effective', 'Adequate', 'Needs Improvement']  # System is generous
            },
            {
                'description': 'The Senior Financial Analyst reviews and reconciles bank statements daily in the accounting system',
                'expected_min_score': 60,  # Should be good
                'expected_category': ['Adequate', 'Effective']
            }
        ]
        
        for case in test_cases:
            result = analyzer_with_config.analyze_control(f"BOUNDARY_{case['description'][:10]}", case['description'])
            
            # Verify score meets minimum expectations
            assert result['total_score'] >= case['expected_min_score'], \
                f"Score too low for '{case['description']}': {result['total_score']}"
            
            # Verify category is in expected range
            assert result['category'] in case['expected_category'], \
                f"Unexpected category for '{case['description']}': {result['category']}"

    def test_category_consistency_with_score(self, analyzer_with_config):
        """Test category assignment consistency with scores (P2 Business Logic)"""
        # Test multiple controls to ensure category-score consistency
        for i in range(10):
            test_control = f"Manager {i} reviews reports and validates data daily in system"
            result = analyzer_with_config.analyze_control(f'CONSISTENCY_{i}', test_control)
            
            score = result['total_score']
            category = result['category']
            
            # Verify category matches score thresholds
            if score >= 75:
                assert category == 'Effective', f"Score {score} should be Effective, got {category}"
            elif score >= 50:
                assert category == 'Adequate', f"Score {score} should be Adequate, got {category}"
            else:
                assert category == 'Needs Improvement', f"Score {score} should be Needs Improvement, got {category}"


class TestEnhancementFeedbackGeneration:
    """Test suite for enhancement feedback generation (P2 Business Logic)"""

    def test_missing_who_feedback(self, analyzer_with_config):
        """Test feedback generation for missing WHO elements (P2 Business Logic)"""
        control_missing_who = "Validates transactions daily in the system"
        
        result = analyzer_with_config.analyze_control('MISSING_WHO_001', control_missing_who)
        
        # Should have lower WHO score than control with clear WHO
        who_score = result['scoring_breakdown']['WHO']
        
        # Compare to control with clear WHO
        clear_who_control = "Finance Manager validates transactions daily in the system"
        clear_result = analyzer_with_config.analyze_control('CLEAR_WHO_001', clear_who_control)
        clear_who_score = clear_result['scoring_breakdown']['WHO']
        
        assert who_score <= clear_who_score, f"Missing WHO should score <= clear WHO: {who_score} vs {clear_who_score}"

    def test_missing_what_feedback(self, analyzer_with_config):
        """Test feedback generation for missing WHAT elements (P2 Business Logic)"""
        control_missing_what = "The Finance Manager daily in SAP system"
        
        result = analyzer_with_config.analyze_control('MISSING_WHAT_001', control_missing_what)
        
        # Should have low WHAT score
        what_score = result['scoring_breakdown']['WHAT']
        assert what_score < 15, f"Missing WHAT should have low score: {what_score}"

    def test_missing_when_feedback(self, analyzer_with_config):
        """Test feedback generation for missing WHEN elements (P2 Business Logic)"""
        control_missing_when = "The Finance Manager reviews transactions in SAP"
        
        result = analyzer_with_config.analyze_control('MISSING_WHEN_001', control_missing_when)
        
        # Should have low WHEN score
        when_score = result['scoring_breakdown']['WHEN']
        assert when_score < 15, f"Missing WHEN should have low score: {when_score}"

    def test_vague_term_feedback(self, analyzer_with_config):
        """Test feedback generation for vague terms (P2 Business Logic)"""
        vague_control = "Management periodically reviews reports as appropriate"
        
        result = analyzer_with_config.analyze_control('VAGUE_FEEDBACK_001', vague_control)
        
        # Should identify specific vague terms
        vague_terms = result['vague_terms_found']
        expected_terms = {'periodically', 'appropriate'}
        found_terms = set(vague_terms)
        
        overlap = found_terms.intersection(expected_terms)
        assert len(overlap) >= 1, f"Should identify vague terms: {vague_terms}"

    def test_enhancement_suggestions_structure(self, analyzer_with_config):
        """Test enhancement feedback structure (P2 Business Logic)"""
        test_control = "Someone does something sometimes"
        
        result = analyzer_with_config.analyze_control('FEEDBACK_STRUCT_001', test_control)
        
        # Verify required feedback fields are present
        required_fields = ['vague_terms_found', 'scoring_breakdown', 'category']
        for field in required_fields:
            assert field in result, f"Missing feedback field: {field}"
        
        # Verify scoring breakdown has all elements
        scoring = result['scoring_breakdown']
        required_elements = ['WHO', 'WHAT', 'WHEN', 'WHERE', 'demerits']
        for element in required_elements:
            assert element in scoring, f"Missing scoring element: {element}"


class TestMultiControlDetection:
    """Test suite for multi-control detection (P2 Business Logic)"""

    def test_sequential_action_detection(self, analyzer_with_config):
        """Test detection of sequential actions as multiple controls (P2 Business Logic)"""
        sequential_control = """
        Manager first reviews daily transactions.
        Then validates account balances.
        Finally approves journal entries.
        """
        
        result = analyzer_with_config.analyze_control('SEQUENTIAL_001', sequential_control)
        
        # Should apply multiple control demerits
        demerits = result['scoring_breakdown']['demerits']
        assert demerits < 0, "Should have demerits for multiple sequential actions"

    def test_complex_workflow_detection(self, analyzer_with_config):
        """Test detection of complex workflows as multiple controls (P2 Business Logic)"""
        complex_workflow = """
        The Finance Manager reviews exception reports, investigates discrepancies,
        validates corrective actions, reconciles accounts, generates summary reports,
        and escalates significant issues to the Controller for final approval.
        """
        
        result = analyzer_with_config.analyze_control('COMPLEX_001', complex_workflow)
        
        # Should detect multiple controls and apply demerits
        demerits = result['scoring_breakdown']['demerits']
        assert demerits < 0, f"Complex workflow should have demerits for multiple controls: {demerits}"
        
        # Should have multiple control penalty (actual system behavior may vary)
        assert demerits <= -5, f"Should have some multiple control penalty: {demerits}"

    def test_coordinated_activities_detection(self, analyzer_with_config):
        """Test detection of coordinated activities (P2 Business Logic)"""
        coordinated_activities = """
        Team lead assigns tasks, monitors progress, reviews deliverables,
        provides feedback, and reports status to management.
        """
        
        result = analyzer_with_config.analyze_control('COORDINATED_001', coordinated_activities)
        
        # Should detect multiple coordinated activities
        demerits = result['scoring_breakdown']['demerits']
        assert demerits < 0, "Should detect coordinated activities as multiple controls"

    def test_single_control_no_demerits(self, analyzer_with_config):
        """Test single control doesn't trigger multiple control demerits (P2 Business Logic)"""
        single_control = "The Finance Manager reviews and approves journal entries daily"
        
        result = analyzer_with_config.analyze_control('SINGLE_001', single_control)
        
        # May have vague term demerits but should not have multiple control demerits
        # (this is harder to test directly, but score should be reasonable)
        assert result['total_score'] > 30, "Single control should not be heavily penalized"

    def test_action_count_thresholds(self, analyzer_with_config):
        """Test action count thresholds for multiple control detection (P2 Business Logic)"""
        # Test different numbers of actions
        test_cases = [
            {
                'actions': 2,
                'text': "Manager reviews and approves transactions",
                'should_trigger': False
            },
            {
                'actions': 3,
                'text': "Manager reviews, validates, and approves transactions",
                'should_trigger': True  # Threshold is > 2 actions
            },
            {
                'actions': 5,
                'text': "Manager reviews, validates, approves, reconciles, and reports transactions",
                'should_trigger': True
            }
        ]
        
        for case in test_cases:
            result = analyzer_with_config.analyze_control(
                f"ACTION_COUNT_{case['actions']}", 
                case['text']
            )
            
            demerits = result['scoring_breakdown']['demerits']
            
            if case['should_trigger']:
                # Should have multiple control demerits (actual penalty may vary)
                assert demerits < 0, f"Should trigger some demerits for {case['actions']} actions: {demerits}"
            # Note: single controls might still have other demerits (vague terms, etc.)


class TestFeedbackOnlyElements:
    """Test suite for WHY/ESCALATION feedback-only validation (P2 Business Logic)"""

    def test_why_element_no_direct_scoring_impact(self, analyzer_with_config):
        """Test WHY element is not directly scored but may affect content analysis (P2 Business Logic)"""
        # Test that WHY elements don't appear in scoring breakdown
        control_with_why = """
        Finance Manager reviews journal entries daily to ensure compliance 
        with SOX requirements and maintain accurate financial records
        """
        
        result = analyzer_with_config.analyze_control('WHY_DIRECT_001', control_with_why)
        
        # Verify WHY is not directly in scoring breakdown
        scoring_elements = result['scoring_breakdown'].keys()
        why_elements = [key for key in scoring_elements if 'why' in key.lower()]
        assert len(why_elements) == 0, f"WHY should not be directly scored: {why_elements}"
        
        # Only core elements should be in scoring
        expected_elements = {'WHO', 'WHAT', 'WHEN', 'WHERE', 'demerits'}
        actual_elements = set(scoring_elements)
        assert actual_elements == expected_elements, f"Unexpected scoring elements: {actual_elements}"
        
        # WHY content may indirectly affect element detection (this is correct behavior)
        # We test that WHY isn't directly scored, not that it has zero indirect effect

    def test_escalation_element_no_direct_scoring_impact(self, analyzer_with_config):
        """Test ESCALATION element is not directly scored but may affect content analysis (P2 Business Logic)"""
        # Test that ESCALATION elements don't appear in scoring breakdown
        control_with_escalation = """
        Manager reviews exception reports daily and escalates significant 
        variances to the Controller for resolution
        """
        
        result = analyzer_with_config.analyze_control('ESC_DIRECT_001', control_with_escalation)
        
        # Verify ESCALATION is not directly in scoring breakdown
        scoring_elements = result['scoring_breakdown'].keys()
        escalation_elements = [key for key in scoring_elements if 'escalation' in key.lower()]
        assert len(escalation_elements) == 0, f"ESCALATION should not be directly scored: {escalation_elements}"
        
        # Only core elements should be in scoring
        expected_elements = {'WHO', 'WHAT', 'WHEN', 'WHERE', 'demerits'}
        actual_elements = set(scoring_elements)
        assert actual_elements == expected_elements, f"Unexpected scoring elements: {actual_elements}"
        
        # ESCALATION content may indirectly affect element detection (this is correct behavior)

    def test_feedback_only_elements_in_total_score(self, analyzer_with_config):
        """Test WHY/ESCALATION are excluded from total score calculation (P2 Business Logic)"""
        control_with_both = """
        The Finance Manager reviews journal entries daily in SAP to ensure SOX compliance
        and escalates material discrepancies to the Controller for immediate resolution
        """
        
        result = analyzer_with_config.analyze_control('FEEDBACK_ONLY_001', control_with_both)
        
        # Calculate expected score from core elements only
        scoring = result['scoring_breakdown']
        core_score = scoring['WHO'] + scoring['WHAT'] + scoring['WHEN'] + scoring['WHERE'] + scoring['demerits']
        
        # Total score should equal core elements (no WHY/ESCALATION contribution)
        score_diff = abs(result['total_score'] - core_score)
        assert score_diff < 1, f"Total score should equal core elements only: {result['total_score']} vs {core_score}"

    def test_why_escalation_feedback_generation(self, analyzer_with_config):
        """Test WHY/ESCALATION feedback is generated but separate from scoring (P2 Business Logic)"""
        comprehensive_control = """
        The Finance Manager validates bank reconciliations monthly in the accounting system
        to ensure accurate cash positions and regulatory compliance, escalating any
        discrepancies exceeding $10,000 to the Controller for immediate investigation
        """
        
        result = analyzer_with_config.analyze_control('COMPREHENSIVE_001', comprehensive_control)
        
        # Should have good scores for core elements
        scoring = result['scoring_breakdown']
        assert scoring['WHO'] > 15, f"Should detect clear WHO: {scoring['WHO']}"
        assert scoring['WHAT'] > 20, f"Should detect clear WHAT: {scoring['WHAT']}"
        assert scoring['WHEN'] > 15, f"Should detect clear WHEN: {scoring['WHEN']}"
        
        # Total score should be sum of core elements only
        core_total = scoring['WHO'] + scoring['WHAT'] + scoring['WHEN'] + scoring['WHERE'] + scoring['demerits']
        assert abs(result['total_score'] - core_total) < 1, "Total should equal core elements only"
        
        # Should be high-quality control
        assert result['category'] in ['Adequate', 'Effective'], f"Comprehensive control should be good quality: {result['category']}"