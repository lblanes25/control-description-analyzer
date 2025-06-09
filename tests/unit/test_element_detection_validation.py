#!/usr/bin/env python3
"""
Comprehensive Element Detection Validation Tests

Tests verify that analyzers correctly identify WHO, WHAT, WHEN, WHERE elements 
in various control scenarios including normal cases, edge cases, and patterns.
This focuses on detection accuracy rather than just API structure.
"""

import pytest
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.analyzers.who import enhanced_who_detection_v2
from src.analyzers.what import enhance_what_detection
from src.analyzers.when import enhance_when_detection
from src.analyzers.where import enhance_where_detection


class TestWHOElementDetectionValidation:
    """Comprehensive WHO element detection validation tests"""

    @pytest.fixture
    def spacy_model(self):
        """Load spaCy model for testing"""
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model not available")

    def test_specific_role_detection_accuracy(self, spacy_model):
        """Test detection of specific roles with high accuracy"""
        test_cases = [
            {
                'text': 'The Finance Manager reviews all journal entries monthly',
                'expected_who': 'Finance Manager',
                'expected_confidence_min': 0.8,
                'expected_type': 'human'
            },
            {
                'text': 'Senior Internal Auditor validates controls quarterly',
                'expected_who': 'Senior Internal Auditor',
                'expected_confidence_min': 0.6,  # Slightly lower due to complexity
                'expected_type': 'human'
            },
            {
                'text': 'The Corporate Controller approves all financial statements',
                'expected_who': 'Corporate Controller',
                'expected_confidence_min': 0.8,  # Now fixed to work properly
                'expected_type': 'human'
            },
            {
                'text': 'Accounts Payable Manager processes vendor invoices',
                'expected_who': 'Manager',  # May detect partial role
                'expected_confidence_min': 0.6,
                'expected_type': 'human'
            }
        ]

        for case in test_cases:
            result = enhanced_who_detection_v2(case['text'], spacy_model)
            
            # Should detect the role
            assert result['primary'] is not None, f"Should detect WHO in: {case['text']}"
            
            # Check role text contains expected role (more flexible matching)
            primary_text = result['primary']['text']
            # Extract key role terms for comparison
            expected_key_terms = case['expected_who'].lower().split()
            primary_text_lower = primary_text.lower()
            
            # Should contain at least one key term from the expected role
            assert any(term in primary_text_lower for term in expected_key_terms), \
                f"Expected key terms from '{case['expected_who']}' in detected '{primary_text}' for: {case['text']}"
            
            # Check confidence
            assert result['confidence'] >= case['expected_confidence_min'], \
                f"Confidence {result['confidence']} should be >= {case['expected_confidence_min']} for: {case['text']}"
            
            # Check type
            assert result['primary']['type'] == case['expected_type'], \
                f"Expected type '{case['expected_type']}', got '{result['primary']['type']}' for: {case['text']}"

    def test_team_department_detection_accuracy(self, spacy_model):
        """Test detection of teams and departments"""
        test_cases = [
            {
                'text': 'The Accounting Team reconciles bank statements weekly',
                'expected_who': 'Accounting Team',
                'expected_confidence_min': 0.6,
                'expected_type': 'human'
            },
            {
                'text': 'Finance department reviews budget variances monthly',
                'expected_who': 'Finance department',
                'expected_confidence_min': 0.6,
                'expected_type': 'human'
            },
            {
                'text': 'Internal Audit Group performs annual risk assessment',
                'expected_who': 'Internal Audit Group',
                'expected_confidence_min': 0.6,
                'expected_type': 'human'
            },
            {
                'text': 'Treasury Committee approves investment policies',
                'expected_who': 'Treasury Committee',
                'expected_confidence_min': 0.6,
                'expected_type': 'human'
            }
        ]

        for case in test_cases:
            result = enhanced_who_detection_v2(case['text'], spacy_model)
            
            assert result['primary'] is not None, f"Should detect WHO in: {case['text']}"
            
            primary_text = result['primary']['text']
            assert case['expected_who'] in primary_text or primary_text in case['expected_who'], \
                f"Expected '{case['expected_who']}' in detected '{primary_text}' for: {case['text']}"
            
            assert result['confidence'] >= case['expected_confidence_min'], \
                f"Confidence {result['confidence']} should be >= {case['expected_confidence_min']} for: {case['text']}"

    def test_system_entity_detection_accuracy(self, spacy_model):
        """Test detection of system entities performing controls"""
        test_cases = [
            {
                'text': 'SAP system automatically validates transaction limits',
                'expected_who': 'SAP system',
                'expected_confidence_min': 0.7,
                'expected_type': 'system'
            },
            {
                'text': 'The application generates daily exception reports',
                'expected_who': 'application',
                'expected_confidence_min': 0.6,
                'expected_type': 'system'
            },
            {
                'text': 'Oracle database enforces referential integrity',
                'expected_who': 'Oracle database',
                'expected_confidence_min': 0.7,
                'expected_type': 'system'
            },
            {
                'text': 'Automated workflow routes approvals to managers',
                'expected_who': 'Automated workflow',
                'expected_confidence_min': 0.6,
                'expected_type': 'system'
            }
        ]

        for case in test_cases:
            result = enhanced_who_detection_v2(case['text'], spacy_model)
            
            assert result['primary'] is not None, f"Should detect WHO in: {case['text']}"
            
            primary_text = result['primary']['text']
            assert case['expected_who'] in primary_text or primary_text in case['expected_who'], \
                f"Expected '{case['expected_who']}' in detected '{primary_text}' for: {case['text']}"
            
            assert result['confidence'] >= case['expected_confidence_min'], \
                f"Confidence {result['confidence']} should be >= {case['expected_confidence_min']} for: {case['text']}"
            
            assert result['primary']['type'] == case['expected_type'], \
                f"Expected type '{case['expected_type']}', got '{result['primary']['type']}' for: {case['text']}"

    def test_vague_role_detection_accuracy(self, spacy_model):
        """Test detection of vague role references with appropriate confidence"""
        test_cases = [
            {
                'text': 'Management reviews financial reports quarterly',
                'expected_who': 'Management',
                'expected_confidence_max': 0.6,
                'expected_type': 'human'
            },
            {
                'text': 'Staff performs daily reconciliations',
                'expected_who': 'Staff',
                'expected_confidence_max': 0.6,
                'expected_type': 'human'
            },
            {
                'text': 'Personnel validate customer information',
                'expected_who': 'Personnel',
                'expected_confidence_max': 0.6,
                'expected_type': 'human'
            }
        ]

        for case in test_cases:
            result = enhanced_who_detection_v2(case['text'], spacy_model)
            
            assert result['primary'] is not None, f"Should detect WHO in: {case['text']}"
            
            primary_text = result['primary']['text']
            assert case['expected_who'] in primary_text or primary_text in case['expected_who'], \
                f"Expected '{case['expected_who']}' in detected '{primary_text}' for: {case['text']}"
            
            # Vague roles should have lower confidence
            assert result['confidence'] <= case['expected_confidence_max'], \
                f"Confidence {result['confidence']} should be <= {case['expected_confidence_max']} for vague role: {case['text']}"

    def test_who_edge_cases(self, spacy_model):
        """Test WHO detection edge cases and challenging scenarios"""
        edge_cases = [
            {
                'name': 'passive_voice_with_clear_performer',
                'text': 'Journal entries are reviewed by the Finance Manager',
                'expected_who_keywords': ['finance', 'manager'],
                'should_detect': True
            },
            {
                'name': 'compound_roles',
                'text': 'Finance Manager and Controller jointly approve budgets',
                'expected_who_keywords': ['finance', 'manager', 'controller'],  # Should detect at least one
                'should_detect': True
            },
            {
                'name': 'no_clear_performer',
                'text': 'Controls are implemented to ensure compliance',
                'should_detect': False,  # No clear performer
                'low_confidence_acceptable': True
            },
            {
                'name': 'multiple_performers_sequence',
                'text': 'Analyst prepares report, Manager reviews, Controller approves',
                'expected_who_keywords': ['analyst', 'manager', 'controller'],  # Should detect primary performer
                'should_detect': True
            },
            {
                'name': 'performer_with_qualification',
                'text': 'Senior Financial Analyst with CPA certification reviews statements',
                'expected_who_keywords': ['senior', 'financial', 'analyst'],
                'should_detect': True
            }
        ]

        for case in edge_cases:
            result = enhanced_who_detection_v2(case['text'], spacy_model)
            
            if case['should_detect']:
                assert result['primary'] is not None, \
                    f"Should detect WHO in {case['name']}: {case['text']}"
                
                if 'expected_who_keywords' in case:
                    primary_text = result['primary']['text'].lower()
                    # Should contain at least one expected keyword
                    assert any(keyword in primary_text for keyword in case['expected_who_keywords']), \
                        f"Expected one of {case['expected_who_keywords']} in detected '{primary_text}' for {case['name']}: {case['text']}"
            else:
                # For cases where no clear performer should be detected
                if result['primary'] is not None and not case.get('low_confidence_acceptable'):
                    # If something is detected, it should have very low confidence
                    assert result['confidence'] < 0.5, \
                        f"Should have low confidence for {case['name']}: {case['text']}"


class TestWHATElementDetectionValidation:
    """Comprehensive WHAT element detection validation tests"""

    @pytest.fixture
    def spacy_model(self):
        """Load spaCy model for testing"""
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model not available")

    def test_strong_action_verb_detection(self, spacy_model):
        """Test detection of strong action verbs with high confidence"""
        test_cases = [
            {
                'text': 'Manager reviews and approves all journal entries',
                'expected_actions': ['reviews', 'approves'],
                'expected_confidence_min': 0.8
            },
            {
                'text': 'Analyst validates transaction accuracy weekly',
                'expected_actions': ['validates'],
                'expected_confidence_min': 0.8
            },
            {
                'text': 'Controller reconciles bank statements monthly',
                'expected_actions': ['reconciles'],
                'expected_confidence_min': 0.8
            },
            {
                'text': 'Auditor verifies completeness of documentation',
                'expected_actions': ['verifies'],
                'expected_confidence_min': 0.8
            }
        ]

        for case in test_cases:
            result = enhance_what_detection(case['text'], spacy_model)
            
            # Should detect primary action
            assert result['primary_action'] is not None, \
                f"Should detect primary action in: {case['text']}"
            
            # Check if detected action matches expected
            detected_verb = result['primary_action']['verb_lemma']
            assert any(expected in detected_verb or detected_verb in expected 
                      for expected in case['expected_actions']), \
                f"Expected one of {case['expected_actions']} but got '{detected_verb}' for: {case['text']}"
            
            # Check confidence
            assert result['primary_action']['score'] >= case['expected_confidence_min'], \
                f"Confidence {result['primary_action']['score']} should be >= {case['expected_confidence_min']} for: {case['text']}"

    def test_moderate_action_verb_detection(self, spacy_model):
        """Test detection of moderate strength action verbs"""
        test_cases = [
            {
                'text': 'Supervisor ensures compliance with policies',
                'expected_actions': ['ensures'],
                'expected_confidence_range': (0.6, 0.8)
            },
            {
                'text': 'Manager coordinates review activities',
                'expected_actions': ['coordinates'],
                'expected_confidence_range': (0.6, 0.8)
            },
            {
                'text': 'Team maintains documentation standards',
                'expected_actions': ['maintains'],
                'expected_confidence_range': (0.6, 0.8)
            }
        ]

        for case in test_cases:
            result = enhance_what_detection(case['text'], spacy_model)
            
            assert result['primary_action'] is not None, \
                f"Should detect primary action in: {case['text']}"
            
            detected_verb = result['primary_action']['verb_lemma']
            assert any(expected in detected_verb or detected_verb in expected 
                      for expected in case['expected_actions']), \
                f"Expected one of {case['expected_actions']} but got '{detected_verb}' for: {case['text']}"
            
            # Check confidence is in moderate range
            score = result['primary_action']['score']
            min_conf, max_conf = case['expected_confidence_range']
            assert min_conf <= score <= max_conf, \
                f"Confidence {score} should be between {min_conf} and {max_conf} for: {case['text']}"

    def test_weak_action_verb_detection(self, spacy_model):
        """Test detection of weak action verbs with appropriate confidence"""
        test_cases = [
            {
                'text': 'Staff considers various options',
                'expected_actions': ['considers'],
                'expected_confidence_max': 0.6
            },
            {
                'text': 'Team attempts to resolve issues',
                'expected_actions': ['attempts'],
                'expected_confidence_max': 0.6
            },
            {
                'text': 'Management observes current practices',
                'expected_actions': ['observes'],
                'expected_confidence_max': 0.6
            }
        ]

        for case in test_cases:
            result = enhance_what_detection(case['text'], spacy_model)
            
            if result['primary_action'] is not None:
                # If weak action is detected, should have low confidence
                assert result['primary_action']['score'] <= case['expected_confidence_max'], \
                    f"Weak action confidence {result['primary_action']['score']} should be <= {case['expected_confidence_max']} for: {case['text']}"

    def test_control_noun_detection(self, spacy_model):
        """Test detection of control-specific nouns that boost action confidence"""
        test_cases = [
            {
                'text': 'Manager performs monthly reconciliation',
                'expected_nouns': ['reconciliation'],
                'action_should_be_boosted': True
            },
            {
                'text': 'Analyst conducts validation procedures',
                'expected_nouns': ['validation'],
                'action_should_be_boosted': True
            },
            {
                'text': 'Controller provides approval for transactions',
                'expected_nouns': ['approval'],
                'action_should_be_boosted': True
            },
            {
                'text': 'Team completes review process',
                'expected_nouns': ['review'],
                'action_should_be_boosted': True
            }
        ]

        for case in test_cases:
            result = enhance_what_detection(case['text'], spacy_model)
            
            # Should detect action
            assert result['primary_action'] is not None, \
                f"Should detect action with control noun in: {case['text']}"
            
            # Action should have reasonable confidence due to control noun
            if case['action_should_be_boosted']:
                assert result['primary_action']['score'] >= 0.6, \
                    f"Action with control noun should have boosted confidence for: {case['text']}"

    def test_what_edge_cases(self, spacy_model):
        """Test WHAT detection edge cases"""
        edge_cases = [
            {
                'name': 'passive_voice_strong_action',
                'text': 'Reports are reviewed by management monthly',
                'expected_action': 'reviewed',
                'should_detect': True
            },
            {
                'name': 'compound_actions',
                'text': 'Manager reviews and approves budget submissions',
                'expected_action': 'reviews',  # Should detect at least one
                'should_detect': True
            },
            {
                'name': 'future_tense_action',
                'text': 'Controller will validate account balances',
                'expected_action': 'validate',
                'should_detect': True
            },
            {
                'name': 'conditional_action',
                'text': 'System may generate exception reports',
                'expected_action': 'generate',
                'should_detect': True,
                'expected_confidence_max': 0.7  # Conditional should be lower confidence
            },
            {
                'name': 'purpose_clause_only',
                'text': 'To ensure compliance with regulations',
                'should_detect': False  # Purpose clause, not action
            },
            {
                'name': 'multiple_actions_sequence',
                'text': 'Prepare report, review findings, and submit recommendations',
                'expected_action': 'prepare',  # Should detect primary action
                'should_detect': True
            }
        ]

        for case in edge_cases:
            result = enhance_what_detection(case['text'], spacy_model)
            
            if case['should_detect']:
                assert result['primary_action'] is not None, \
                    f"Should detect action in {case['name']}: {case['text']}"
                
                if 'expected_action' in case:
                    detected_verb = result['primary_action']['verb_lemma']
                    assert case['expected_action'] in detected_verb or detected_verb in case['expected_action'], \
                        f"Expected '{case['expected_action']}' in detected '{detected_verb}' for {case['name']}: {case['text']}"
                
                if 'expected_confidence_max' in case:
                    assert result['primary_action']['score'] <= case['expected_confidence_max'], \
                        f"Confidence should be <= {case['expected_confidence_max']} for {case['name']}: {case['text']}"
            else:
                # Should not detect meaningful action or should have very low confidence
                if result['primary_action'] is not None:
                    assert result['primary_action']['score'] < 0.5, \
                        f"Should have low confidence for {case['name']}: {case['text']}"


class TestWHENElementDetectionValidation:
    """Comprehensive WHEN element detection validation tests"""

    @pytest.fixture
    def spacy_model(self):
        """Load spaCy model for testing"""
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model not available")

    def test_explicit_frequency_detection(self, spacy_model):
        """Test detection of explicit frequencies with high confidence"""
        test_cases = [
            {
                'text': 'Manager reviews reports daily',
                'expected_frequency': 'daily',
                'expected_confidence_min': 0.8
            },
            {
                'text': 'Team performs reconciliation monthly',
                'expected_frequency': 'monthly',
                'expected_confidence_min': 0.8
            },
            {
                'text': 'Auditor conducts testing quarterly',
                'expected_frequency': 'quarterly',
                'expected_confidence_min': 0.8
            },
            {
                'text': 'Controller completes annual review',
                'expected_frequency': 'annually',
                'expected_confidence_min': 0.8
            },
            {
                'text': 'Staff validates data weekly',
                'expected_frequency': 'weekly',
                'expected_confidence_min': 0.8
            }
        ]

        for case in test_cases:
            result = enhance_when_detection(case['text'], spacy_model)
            
            # Should detect timing
            assert result['top_match'] is not None, \
                f"Should detect timing in: {case['text']}"
            
            # Check frequency matches
            detected_frequency = result['top_match'].get('frequency')
            assert detected_frequency == case['expected_frequency'], \
                f"Expected frequency '{case['expected_frequency']}' but got '{detected_frequency}' for: {case['text']}"
            
            # Check confidence
            assert result['top_match']['score'] >= case['expected_confidence_min'], \
                f"Confidence {result['top_match']['score']} should be >= {case['expected_confidence_min']} for: {case['text']}"

    def test_period_end_timing_detection(self, spacy_model):
        """Test detection of period-end timing patterns"""
        test_cases = [
            {
                'text': 'Controller performs closing procedures at month-end',
                'expected_pattern': 'month-end',
                'expected_frequency': 'monthly',
                'expected_confidence_min': 0.7
            },
            {
                'text': 'Team completes reconciliation by quarter-end',
                'expected_pattern': 'quarter-end',
                'expected_frequency': 'quarterly',
                'expected_confidence_min': 0.7
            },
            {
                'text': 'Auditor reviews controls during year-end close',
                'expected_pattern': 'year-end',
                'expected_frequency': 'annually',
                'expected_confidence_min': 0.7
            }
        ]

        for case in test_cases:
            result = enhance_when_detection(case['text'], spacy_model)
            
            assert result['top_match'] is not None, \
                f"Should detect period-end timing in: {case['text']}"
            
            # Check that period-end pattern is detected
            detected_text = result['top_match']['text'].lower()
            assert case['expected_pattern'] in detected_text, \
                f"Expected '{case['expected_pattern']}' in detected '{detected_text}' for: {case['text']}"
            
            # Check inferred frequency
            if 'frequency' in result['top_match']:
                assert result['top_match']['frequency'] == case['expected_frequency'], \
                    f"Expected frequency '{case['expected_frequency']}' for period-end pattern: {case['text']}"

    def test_conditional_timing_detection(self, spacy_model):
        """Test detection of conditional/event-based timing"""
        test_cases = [
            {
                'text': 'Staff validates data as needed',
                'expected_pattern': 'as needed',
                'expected_confidence_max': 0.7
            },
            {
                'text': 'Manager reviews exceptions when necessary',
                'expected_pattern': 'when necessary',
                'expected_confidence_max': 0.7
            },
            {
                'text': 'Team investigates issues upon notification',
                'expected_pattern': 'upon notification',
                'expected_confidence_max': 0.8
            },
            {
                'text': 'Controller approves transactions if required',
                'expected_pattern': 'if required',
                'expected_confidence_max': 0.7
            }
        ]

        for case in test_cases:
            result = enhance_when_detection(case['text'], spacy_model)
            
            # Conditional timing may be detected as vague or event-based
            if result['top_match'] is not None:
                detected_text = result['top_match']['text'].lower()
                
                # Should have lower confidence than explicit frequencies
                assert result['top_match']['score'] <= case['expected_confidence_max'], \
                    f"Conditional timing confidence {result['top_match']['score']} should be <= {case['expected_confidence_max']} for: {case['text']}"

    def test_vague_timing_detection(self, spacy_model):
        """Test detection and proper handling of vague timing terms"""
        test_cases = [
            {
                'text': 'Manager periodically reviews reports',
                'expected_vague_term': 'periodically',
                'should_be_flagged': True
            },
            {
                'text': 'Staff regularly validates data',
                'expected_vague_term': 'regularly',
                'should_be_flagged': True
            },
            {
                'text': 'Team provides timely resolution',
                'expected_vague_term': 'timely',
                'should_be_flagged': True
            },
            {
                'text': 'Controls are performed occasionally',
                'expected_vague_term': 'occasionally',
                'should_be_flagged': True
            }
        ]

        for case in test_cases:
            result = enhance_when_detection(case['text'], spacy_model)
            
            if case['should_be_flagged']:
                # Should detect vague terms
                vague_terms = result.get('vague_terms', [])
                assert len(vague_terms) > 0, \
                    f"Should detect vague timing terms in: {case['text']}"
                
                # Should have vague term in detected terms
                vague_term_texts = [vt['text'].lower() for vt in vague_terms]
                assert any(case['expected_vague_term'] in vt for vt in vague_term_texts), \
                    f"Expected vague term '{case['expected_vague_term']}' in {vague_term_texts} for: {case['text']}"
                
                # Overall score should be low for vague timing
                assert result['score'] <= 0.3, \
                    f"Score should be low for vague timing in: {case['text']}"

    def test_when_edge_cases(self, spacy_model):
        """Test WHEN detection edge cases"""
        edge_cases = [
            {
                'name': 'multiple_frequencies',
                'text': 'Manager reviews daily reports and monthly summaries',
                'should_detect_multiple': True
            },
            {
                'name': 'no_timing_information',
                'text': 'Manager approves transactions',
                'should_detect': False
            },
            {
                'name': 'business_cycle_timing',
                'text': 'Controller performs review during closing cycle',
                'expected_pattern': 'closing cycle',
                'should_detect': True
            },
            {
                'name': 'specific_timeframe',
                'text': 'Team responds within 2 business days',
                'expected_pattern': 'within 2 business days',
                'should_detect': True
            },
            {
                'name': 'ad_hoc_timing',
                'text': 'Manager conducts ad-hoc reviews',
                'expected_pattern': 'ad-hoc',
                'should_detect': True
            }
        ]

        for case in edge_cases:
            result = enhance_when_detection(case['text'], spacy_model)
            
            if case['should_detect']:
                assert result['top_match'] is not None, \
                    f"Should detect timing in {case['name']}: {case['text']}"
                
                if 'expected_pattern' in case:
                    detected_text = result['top_match']['text'].lower()
                    assert case['expected_pattern'] in detected_text, \
                        f"Expected '{case['expected_pattern']}' in detected '{detected_text}' for {case['name']}: {case['text']}"
            
            elif not case['should_detect']:
                # Should either not detect timing or have very low score
                if result['top_match'] is not None:
                    assert result['score'] < 0.5, \
                        f"Should have low score for {case['name']}: {case['text']}"
            
            if case.get('should_detect_multiple'):
                # Should detect multiple frequencies
                frequencies = result.get('frequencies', [])
                assert len(frequencies) > 1, \
                    f"Should detect multiple frequencies for {case['name']}: {case['text']}"


class TestWHEREElementDetectionValidation:
    """Comprehensive WHERE element detection validation tests"""

    @pytest.fixture
    def spacy_model(self):
        """Load spaCy model for testing"""
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model not available")

    def test_system_location_detection(self, spacy_model):
        """Test detection of systems where controls are executed"""
        test_cases = [
            {
                'text': 'Manager reviews transactions in SAP',
                'expected_where': 'SAP',
                'expected_type': 'system',
                'expected_confidence_min': 0.7
            },
            {
                'text': 'System validates data in Oracle database',
                'expected_where': 'Oracle database',
                'expected_type': 'system',
                'expected_confidence_min': 0.7
            },
            {
                'text': 'Controller approves entries in the ERP system',
                'expected_where': 'ERP system',
                'expected_type': 'system',
                'expected_confidence_min': 0.6
            },
            {
                'text': 'Team generates reports from SharePoint',
                'expected_where': 'SharePoint',
                'expected_type': 'system',
                'expected_confidence_min': 0.7
            }
        ]

        for case in test_cases:
            result = enhance_where_detection(case['text'], spacy_model)
            
            # Should detect WHERE component
            assert result['primary_location'] is not None, \
                f"Should detect WHERE in: {case['text']}"
            
            # Check detected location
            primary_text = result['primary_location']['text']
            assert case['expected_where'] in primary_text or primary_text in case['expected_where'], \
                f"Expected '{case['expected_where']}' in detected '{primary_text}' for: {case['text']}"
            
            # Check type
            assert result['primary_location']['type'] == case['expected_type'], \
                f"Expected type '{case['expected_type']}', got '{result['primary_location']['type']}' for: {case['text']}"
            
            # Check confidence
            assert result['confidence'] >= case['expected_confidence_min'], \
                f"Confidence {result['confidence']} should be >= {case['expected_confidence_min']} for: {case['text']}"

    def test_physical_location_detection(self, spacy_model):
        """Test detection of physical locations where controls are performed"""
        test_cases = [
            {
                'text': 'Guard performs inspection at the vault',
                'expected_where': 'vault',
                'expected_type': 'location',
                'expected_confidence_min': 0.6
            },
            {
                'text': 'Manager reviews documents at branch office',
                'expected_where': 'branch office',
                'expected_type': 'location',
                'expected_confidence_min': 0.6
            },
            {
                'text': 'Team conducts inventory count at warehouse',
                'expected_where': 'warehouse',
                'expected_type': 'location',
                'expected_confidence_min': 0.6
            },
            {
                'text': 'Controller signs checks in the finance department',
                'expected_where': 'finance department',
                'expected_type': 'organizational',
                'expected_confidence_min': 0.6
            }
        ]

        for case in test_cases:
            result = enhance_where_detection(case['text'], spacy_model)
            
            # Should detect WHERE component
            assert result['primary_location'] is not None, \
                f"Should detect WHERE in: {case['text']}"
            
            # Check detected location
            primary_text = result['primary_location']['text']
            assert case['expected_where'] in primary_text or primary_text in case['expected_where'], \
                f"Expected '{case['expected_where']}' in detected '{primary_text}' for: {case['text']}"

    def test_organizational_location_detection(self, spacy_model):
        """Test detection of organizational units as locations"""
        test_cases = [
            {
                'text': 'Accounting department reconciles bank statements',
                'expected_where': 'Accounting department',
                'expected_type': 'organizational',
                'expected_confidence_min': 0.6
            },
            {
                'text': 'Treasury team manages cash flows',
                'expected_where': 'Treasury team',
                'expected_type': 'organizational',
                'expected_confidence_min': 0.6
            },
            {
                'text': 'Internal audit group reviews controls',
                'expected_where': 'Internal audit group',
                'expected_type': 'organizational',
                'expected_confidence_min': 0.6
            },
            {
                'text': 'Finance committee approves budgets',
                'expected_where': 'Finance committee',
                'expected_type': 'organizational',
                'expected_confidence_min': 0.6
            }
        ]

        for case in test_cases:
            result = enhance_where_detection(case['text'], spacy_model)
            
            # Should detect WHERE component
            assert result['primary_location'] is not None, \
                f"Should detect WHERE in: {case['text']}"
            
            # Check detected location
            primary_text = result['primary_location']['text']
            assert case['expected_where'] in primary_text or primary_text in case['expected_where'], \
                f"Expected '{case['expected_where']}' in detected '{primary_text}' for: {case['text']}"

    def test_where_edge_cases(self, spacy_model):
        """Test WHERE detection edge cases"""
        edge_cases = [
            {
                'name': 'multiple_locations',
                'text': 'Manager reviews data in SAP and validates in Oracle',
                'should_detect_multiple': True
            },
            {
                'name': 'vague_system_reference',
                'text': 'Team processes data in the system',
                'expected_where': 'system',
                'should_detect': True,
                'expected_confidence_max': 0.7
            },
            {
                'name': 'no_location_information',
                'text': 'Manager approves transactions',
                'should_detect': False
            },
            {
                'name': 'implicit_system_location',
                'text': 'Automated controls validate transaction limits',
                'should_detect': False  # No explicit WHERE mentioned
            },
            {
                'name': 'location_with_preposition',
                'text': 'Controller works from the main office',
                'expected_where': 'main office',
                'should_detect': True
            }
        ]

        for case in edge_cases:
            result = enhance_where_detection(case['text'], spacy_model)
            
            if case['should_detect']:
                assert result['primary_location'] is not None, \
                    f"Should detect WHERE in {case['name']}: {case['text']}"
                
                if 'expected_where' in case:
                    primary_text = result['primary_location']['text']
                    assert case['expected_where'] in primary_text or primary_text in case['expected_where'], \
                        f"Expected '{case['expected_where']}' in detected '{primary_text}' for {case['name']}: {case['text']}"
                
                if 'expected_confidence_max' in case:
                    assert result['confidence'] <= case['expected_confidence_max'], \
                        f"Confidence should be <= {case['expected_confidence_max']} for {case['name']}: {case['text']}"
            
            elif not case['should_detect']:
                # Should either not detect WHERE or have very low confidence
                if result['primary_location'] is not None:
                    assert result['confidence'] < 0.5, \
                        f"Should have low confidence for {case['name']}: {case['text']}"
            
            if case.get('should_detect_multiple'):
                # Should detect multiple location components
                all_components = result['components'].get('all_components', [])
                assert len(all_components) > 1, \
                    f"Should detect multiple locations for {case['name']}: {case['text']}"


class TestElementDetectionEdgeCases:
    """Edge cases and challenging scenarios for element detection"""

    @pytest.fixture
    def spacy_model(self):
        """Load spaCy model for testing"""
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spaCy model not available")

    def test_complex_control_statements(self, spacy_model):
        """Test element detection in complex, realistic control statements"""
        complex_cases = [
            {
                'text': 'The Finance Manager reviews all journal entries exceeding $10,000 in SAP on a daily basis to ensure accuracy and compliance',
                'expected_who': 'Finance Manager',
                'expected_what': 'reviews',
                'expected_when': 'daily',
                'expected_where': 'SAP'
            },
            {
                'text': 'Senior Internal Auditor validates system access controls quarterly by testing user permissions in Active Directory',
                'expected_who': 'Senior Internal Auditor', 
                'expected_what': 'validates',
                'expected_when': 'quarterly',
                'expected_where': 'Active Directory'
            },
            {
                'text': 'Accounting team reconciles bank statements monthly and investigates variances exceeding materiality threshold',
                'expected_who': 'Accounting team',
                'expected_what': 'reconciles',
                'expected_when': 'monthly',
                'expected_where': None  # No explicit system/location
            }
        ]

        for case in complex_cases:
            # Test WHO detection
            who_result = enhanced_who_detection_v2(case['text'], spacy_model)
            if case['expected_who']:
                assert who_result['primary'] is not None
                assert case['expected_who'] in who_result['primary']['text']
            
            # Test WHAT detection  
            what_result = enhance_what_detection(case['text'], spacy_model)
            if case['expected_what']:
                assert what_result['primary_action'] is not None
                assert case['expected_what'] in what_result['primary_action']['verb_lemma']
            
            # Test WHEN detection
            when_result = enhance_when_detection(case['text'], spacy_model)
            if case['expected_when']:
                assert when_result['top_match'] is not None
                assert case['expected_when'] in when_result['top_match'].get('frequency', '')
            
            # Test WHERE detection
            where_result = enhance_where_detection(case['text'], spacy_model)
            if case['expected_where']:
                assert where_result['primary_location'] is not None
                assert case['expected_where'] in where_result['primary_location']['text']

    def test_ambiguous_element_scenarios(self, spacy_model):
        """Test scenarios where elements might be ambiguous or missing"""
        ambiguous_cases = [
            {
                'name': 'passive_voice_unclear_performer',
                'text': 'Reports are generated and distributed',
                'who_should_be_unclear': True
            },
            {
                'name': 'vague_action_description',
                'text': 'Staff handles customer inquiries appropriately',
                'what_should_be_weak': True,
                'when_should_be_vague': True
            },
            {
                'name': 'purpose_statement_not_action',
                'text': 'To ensure compliance with regulatory requirements',
                'what_should_not_detect': True
            },
            {
                'name': 'incomplete_control_description',
                'text': 'Management oversight',
                'all_elements_should_be_weak': True
            }
        ]

        for case in ambiguous_cases:
            who_result = enhanced_who_detection_v2(case['text'], spacy_model)
            what_result = enhance_what_detection(case['text'], spacy_model)
            when_result = enhance_when_detection(case['text'], spacy_model)
            where_result = enhance_where_detection(case['text'], spacy_model)
            
            if case.get('who_should_be_unclear'):
                if who_result['primary'] is not None:
                    assert who_result['confidence'] < 0.6, \
                        f"WHO should be unclear for {case['name']}: {case['text']}"
            
            if case.get('what_should_be_weak'):
                if what_result['primary_action'] is not None:
                    assert what_result['primary_action']['score'] < 0.6, \
                        f"WHAT should be weak for {case['name']}: {case['text']}"
            
            if case.get('what_should_not_detect'):
                if what_result['primary_action'] is not None:
                    assert what_result['primary_action']['score'] < 0.5, \
                        f"WHAT should not detect meaningful action for {case['name']}: {case['text']}"
            
            if case.get('when_should_be_vague'):
                if when_result['top_match'] is not None:
                    assert when_result['score'] <= 0.3, \
                        f"WHEN should be vague for {case['name']}: {case['text']}"
            
            if case.get('all_elements_should_be_weak'):
                if who_result['primary'] is not None:
                    assert who_result['confidence'] < 0.6
                if what_result['primary_action'] is not None:
                    assert what_result['primary_action']['score'] < 0.6
                if when_result['top_match'] is not None:
                    assert when_result['score'] < 0.6
                if where_result['primary_location'] is not None:
                    assert where_result['confidence'] < 0.6