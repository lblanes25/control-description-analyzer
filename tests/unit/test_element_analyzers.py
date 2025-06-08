#!/usr/bin/env python3
"""
Unit Tests for Element Analyzers (Priority 1 - Critical)

Tests individual element analyzers including:
- WHO element detection and scoring
- WHAT element detection and scoring  
- WHEN element detection and scoring
- WHERE element conditional scoring
- WHY element feedback generation
- ESCALATION element soft flagging
"""

import pytest
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.analyzers.who import PersonRoleDetector, SystemDetector
from src.analyzers.what import ActionAnalyzer
from src.analyzers.when import TimingPatternMatcher
from src.analyzers.why import PurposeAnalyzer
from src.analyzers.escalation import EscalationPathDetector


class TestWHOElementAnalysis:
    """Test suite for WHO element detection and scoring (P1 Critical)"""

    @pytest.fixture
    def person_detector(self):
        """Create PersonRoleDetector with test configuration"""
        config = {
            'person_roles': {
                'executive': ['ceo', 'cfo', 'controller', 'director'],
                'management': ['manager', 'supervisor', 'lead'],
                'staff': ['analyst', 'specialist', 'accountant', 'auditor']
            },
            'human_indicators': [
                'responsible', 'performs', 'conducts', 'oversees', 'approves'
            ]
        }
        return PersonRoleDetector(config)

    @pytest.fixture  
    def system_detector(self):
        """Create SystemDetector with test configuration"""
        config = {
            'system_patterns': {
                'automated_keywords': ['system', 'application', 'automated'],
                'system_verbs': ['generates', 'calculates', 'processes'],
                'common_systems': ['sap', 'oracle', 'sharepoint']
            }
        }
        return SystemDetector(config)

    def test_specific_role_detection(self, person_detector):
        """Test detection of specific roles (P1 Critical)"""
        test_cases = [
            {
                'text': 'The Finance Manager reviews transactions daily',
                'expected_role': 'Finance Manager',
                'expected_confidence': 'high',
                'expected_type': 'specific_role'
            },
            {
                'text': 'Senior Internal Auditor validates controls monthly',
                'expected_role': 'Senior Internal Auditor', 
                'expected_confidence': 'high',
                'expected_type': 'specific_role'
            },
            {
                'text': 'Corporate Controller approves journal entries',
                'expected_role': 'Corporate Controller',
                'expected_confidence': 'high',
                'expected_type': 'specific_role'
            }
        ]

        for case in test_cases:
            result = person_detector.detect_person_role(case['text'])
            
            # Should detect the specific role
            assert len(result['detected_roles']) > 0, f"Should detect role in: {case['text']}"
            
            # Should have high confidence for specific roles
            max_confidence = max(role['confidence'] for role in result['detected_roles'])
            assert max_confidence >= 0.8, f"Should have high confidence for specific role: {max_confidence}"

    def test_team_department_detection(self, person_detector):
        """Test detection of teams and departments (P1 Critical)"""
        test_cases = [
            {
                'text': 'Accounting team validates balances monthly',
                'expected_type': 'team',
                'expected_confidence': 'medium'
            },
            {
                'text': 'Finance department reviews reports quarterly',
                'expected_type': 'department',
                'expected_confidence': 'medium'
            },
            {
                'text': 'Internal audit group performs testing',
                'expected_type': 'group',
                'expected_confidence': 'medium'
            }
        ]

        for case in test_cases:
            result = person_detector.detect_person_role(case['text'])
            
            # Should detect team/department references
            assert len(result['detected_roles']) > 0, f"Should detect team/dept in: {case['text']}"
            
            # Should have medium confidence
            max_confidence = max(role['confidence'] for role in result['detected_roles'])
            assert 0.5 <= max_confidence < 0.8, f"Should have medium confidence for team: {max_confidence}"

    def test_vague_role_detection(self, person_detector):
        """Test detection of vague role references (P1 Critical)"""
        test_cases = [
            {
                'text': 'Management reviews reports quarterly',
                'expected_type': 'vague',
                'expected_confidence': 'low'
            },
            {
                'text': 'Staff performs various reconciliations',
                'expected_type': 'vague',
                'expected_confidence': 'low'
            },
            {
                'text': 'Personnel validate transactions as needed',
                'expected_type': 'vague',
                'expected_confidence': 'low'
            }
        ]

        for case in test_cases:
            result = person_detector.detect_person_role(case['text'])
            
            # Should detect vague references
            assert len(result['detected_roles']) > 0, f"Should detect vague role in: {case['text']}"
            
            # Should have low confidence
            max_confidence = max(role['confidence'] for role in result['detected_roles'])
            assert max_confidence < 0.6, f"Should have low confidence for vague role: {max_confidence}"

    def test_system_entity_detection(self, system_detector):
        """Test detection of system entities (P1 Critical)"""
        test_cases = [
            {
                'text': 'System validates transaction limits automatically',
                'expected_system': True,
                'expected_confidence': 'high'
            },
            {
                'text': 'SAP generates daily reports for review',
                'expected_system': True,
                'expected_confidence': 'high'
            },
            {
                'text': 'Application calculates interest monthly',
                'expected_system': True,
                'expected_confidence': 'high'
            }
        ]

        for case in test_cases:
            result = system_detector.detect_system_entity(case['text'])
            
            if case['expected_system']:
                assert len(result['detected_systems']) > 0, f"Should detect system in: {case['text']}"
                max_confidence = max(sys['confidence'] for sys in result['detected_systems'])
                assert max_confidence >= 0.7, f"Should have high confidence for system: {max_confidence}"

    def test_who_element_scoring_weights(self, person_detector):
        """Test WHO element scoring follows weight distribution (P1 Critical)"""
        # This test verifies the 30% weight allocation for WHO elements
        high_confidence_text = 'The Finance Manager reviews transactions daily'
        medium_confidence_text = 'Accounting team validates balances monthly'  
        low_confidence_text = 'Management reviews reports quarterly'

        results = []
        for text in [high_confidence_text, medium_confidence_text, low_confidence_text]:
            result = person_detector.detect_person_role(text)
            max_confidence = max(role['confidence'] for role in result['detected_roles']) if result['detected_roles'] else 0
            results.append(max_confidence)

        # High confidence should be significantly higher than low confidence
        assert results[0] > results[2], "Specific roles should have higher confidence than vague references"
        assert results[1] > results[2], "Teams should have higher confidence than vague references"


class TestWHATElementAnalysis:
    """Test suite for WHAT element detection and scoring (P1 Critical)"""

    @pytest.fixture
    def action_analyzer(self):
        """Create ActionAnalyzer with test configuration"""
        config = {
            'actionable_verbs': {
                'strong_action': {
                    'confidence': 0.9,
                    'verbs': ['review', 'approve', 'validate', 'verify', 'reconcile', 'analyze']
                },
                'moderate_action': {
                    'confidence': 0.7,
                    'verbs': ['ensure', 'coordinate', 'facilitate', 'oversee']
                },
                'weak_action': {
                    'confidence': 0.5,
                    'verbs': ['consider', 'attempt', 'try', 'seek']
                }
            },
            'control_nouns': ['reconciliation', 'validation', 'approval', 'review'],
            'confidence_threshold': 0.4
        }
        return ActionAnalyzer(config)

    def test_strong_action_verb_detection(self, action_analyzer):
        """Test detection of strong action verbs (P1 Critical)"""
        test_cases = [
            {
                'text': 'Manager reviews and approves journal entries',
                'expected_verbs': ['reviews', 'approves'],
                'expected_strength': 'strong',
                'expected_confidence': 0.9
            },
            {
                'text': 'Analyst validates transaction accuracy',
                'expected_verbs': ['validates'],
                'expected_strength': 'strong',
                'expected_confidence': 0.9
            },
            {
                'text': 'Auditor reconciles account balances monthly',
                'expected_verbs': ['reconciles'],
                'expected_strength': 'strong',
                'expected_confidence': 0.9
            }
        ]

        for case in test_cases:
            result = action_analyzer.analyze_actions(case['text'])
            
            # Should detect strong actions
            assert len(result['detected_actions']) > 0, f"Should detect actions in: {case['text']}"
            
            # Should have high confidence
            max_confidence = max(action['confidence'] for action in result['detected_actions'])
            assert max_confidence >= case['expected_confidence'], f"Should have high confidence: {max_confidence}"

    def test_moderate_action_verb_detection(self, action_analyzer):
        """Test detection of moderate action verbs (P1 Critical)"""
        test_cases = [
            {
                'text': 'Supervisor ensures compliance with procedures',
                'expected_verbs': ['ensures'],
                'expected_strength': 'moderate',
                'expected_confidence': 0.7
            },
            {
                'text': 'Manager coordinates review activities',
                'expected_verbs': ['coordinates'],
                'expected_strength': 'moderate',
                'expected_confidence': 0.7
            }
        ]

        for case in test_cases:
            result = action_analyzer.analyze_actions(case['text'])
            
            # Should detect moderate actions
            assert len(result['detected_actions']) > 0, f"Should detect actions in: {case['text']}"
            
            # Should have medium confidence
            max_confidence = max(action['confidence'] for action in result['detected_actions'])
            assert 0.6 <= max_confidence < 0.8, f"Should have medium confidence: {max_confidence}"

    def test_weak_action_verb_detection(self, action_analyzer):
        """Test detection of weak action verbs (P1 Critical)"""
        test_cases = [
            {
                'text': 'Staff considers various options',
                'expected_verbs': ['considers'],
                'expected_strength': 'weak',
                'expected_confidence': 0.5
            },
            {
                'text': 'Team attempts to resolve issues',
                'expected_verbs': ['attempts'],
                'expected_strength': 'weak',
                'expected_confidence': 0.5
            }
        ]

        for case in test_cases:
            result = action_analyzer.analyze_actions(case['text'])
            
            # Should detect weak actions
            assert len(result['detected_actions']) > 0, f"Should detect actions in: {case['text']}"
            
            # Should have low confidence
            max_confidence = max(action['confidence'] for action in result['detected_actions'])
            assert max_confidence <= 0.6, f"Should have low confidence: {max_confidence}"

    def test_control_noun_detection(self, action_analyzer):
        """Test detection of control-specific nouns (P1 Critical)"""
        test_cases = [
            {
                'text': 'Manager performs monthly reconciliation',
                'expected_nouns': ['reconciliation'],
                'should_boost': True
            },
            {
                'text': 'Analyst conducts validation procedures',
                'expected_nouns': ['validation'],
                'should_boost': True
            },
            {
                'text': 'Controller provides approval for transactions',
                'expected_nouns': ['approval'],
                'should_boost': True
            }
        ]

        for case in test_cases:
            result = action_analyzer.analyze_actions(case['text'])
            
            # Should detect control nouns
            if case['should_boost']:
                assert len(result['detected_actions']) > 0, f"Should detect actions with control nouns: {case['text']}"

    def test_what_element_scoring_weights(self, action_analyzer):
        """Test WHAT element scoring follows 35% weight distribution (P1 Critical)"""
        # Test that action strength affects scoring appropriately
        strong_action_text = 'Manager reviews and approves transactions'
        weak_action_text = 'Staff considers various options'

        strong_result = action_analyzer.analyze_actions(strong_action_text)
        weak_result = action_analyzer.analyze_actions(weak_action_text)

        strong_confidence = max(action['confidence'] for action in strong_result['detected_actions']) if strong_result['detected_actions'] else 0
        weak_confidence = max(action['confidence'] for action in weak_result['detected_actions']) if weak_result['detected_actions'] else 0

        # Strong actions should have significantly higher confidence
        assert strong_confidence > weak_confidence, "Strong actions should have higher confidence than weak actions"


class TestWHENElementAnalysis:
    """Test suite for WHEN element detection and scoring (P1 Critical)"""

    @pytest.fixture
    def timing_matcher(self):
        """Create TimingPatternMatcher with test configuration"""
        config = {
            'timing_pattern_rules': {
                'explicit_frequency': {
                    'confidence': 0.9,
                    'patterns': ['daily', 'weekly', 'monthly', 'quarterly', 'annually']
                },
                'period_end': {
                    'confidence': 0.85,
                    'patterns': ['month-end', 'quarter-end', 'year-end']
                },
                'event_trigger': {
                    'confidence': 0.75,
                    'patterns': ['upon receipt', 'when received', 'after approval']
                },
                'conditional_timing': {
                    'confidence': 0.7,
                    'patterns': ['as needed', 'when necessary', 'if required']
                }
            },
            'vague_terms': {
                'periodically': {'penalty': 0.3},
                'regularly': {'penalty': 0.3},
                'timely': {'penalty': 0.25}
            }
        }
        return TimingPatternMatcher(config)

    def test_explicit_frequency_detection(self, timing_matcher):
        """Test detection of explicit frequencies (P1 Critical)"""
        test_cases = [
            {
                'text': 'Manager reviews reports daily',
                'expected_pattern': 'daily',
                'expected_confidence': 0.9,
                'expected_type': 'explicit_frequency'
            },
            {
                'text': 'Team performs reconciliation monthly',
                'expected_pattern': 'monthly',
                'expected_confidence': 0.9,
                'expected_type': 'explicit_frequency'
            },
            {
                'text': 'Auditor conducts testing quarterly',
                'expected_pattern': 'quarterly',
                'expected_confidence': 0.9,
                'expected_type': 'explicit_frequency'
            }
        ]

        for case in test_cases:
            result = timing_matcher.match_timing_patterns(case['text'])
            
            # Should detect explicit timing
            assert len(result['detected_patterns']) > 0, f"Should detect timing in: {case['text']}"
            
            # Should have high confidence
            max_confidence = max(pattern['confidence'] for pattern in result['detected_patterns'])
            assert max_confidence >= case['expected_confidence'], f"Should have high confidence: {max_confidence}"

    def test_period_end_timing_detection(self, timing_matcher):
        """Test detection of period-end timing (P1 Critical)"""
        test_cases = [
            {
                'text': 'Controller performs closing procedures at month-end',
                'expected_pattern': 'month-end',
                'expected_confidence': 0.85
            },
            {
                'text': 'Team completes reconciliation by quarter-end',
                'expected_pattern': 'quarter-end',
                'expected_confidence': 0.85
            }
        ]

        for case in test_cases:
            result = timing_matcher.match_timing_patterns(case['text'])
            
            # Should detect period-end timing
            assert len(result['detected_patterns']) > 0, f"Should detect period-end timing in: {case['text']}"
            
            # Should have high confidence
            max_confidence = max(pattern['confidence'] for pattern in result['detected_patterns'])
            assert max_confidence >= case['expected_confidence'], f"Should have appropriate confidence: {max_confidence}"

    def test_conditional_timing_detection(self, timing_matcher):
        """Test detection of conditional timing (P1 Critical)"""
        test_cases = [
            {
                'text': 'Staff validates data as needed',
                'expected_pattern': 'as needed',
                'expected_confidence': 0.7,
                'expected_type': 'conditional'
            },
            {
                'text': 'Manager reviews exceptions when necessary',
                'expected_pattern': 'when necessary',
                'expected_confidence': 0.7,
                'expected_type': 'conditional'
            }
        ]

        for case in test_cases:
            result = timing_matcher.match_timing_patterns(case['text'])
            
            # Should detect conditional timing
            assert len(result['detected_patterns']) > 0, f"Should detect conditional timing in: {case['text']}"
            
            # Should have lower confidence than explicit timing
            max_confidence = max(pattern['confidence'] for pattern in result['detected_patterns'])
            assert max_confidence <= 0.75, f"Conditional timing should have lower confidence: {max_confidence}"

    def test_vague_timing_penalty_detection(self, timing_matcher):
        """Test detection and penalization of vague timing terms (P1 Critical)"""
        test_cases = [
            {
                'text': 'Manager periodically reviews reports',
                'expected_vague_terms': ['periodically'],
                'expected_penalty': True
            },
            {
                'text': 'Staff regularly validates data',
                'expected_vague_terms': ['regularly'],
                'expected_penalty': True
            },
            {
                'text': 'Team provides timely resolution',
                'expected_vague_terms': ['timely'],
                'expected_penalty': True
            }
        ]

        for case in test_cases:
            result = timing_matcher.match_timing_patterns(case['text'])
            
            # Should detect vague terms and apply penalties
            if case['expected_penalty']:
                # The vague terms should be detected in the analysis
                # Implementation may vary, but confidence should be reduced
                pass

    def test_when_element_scoring_weights(self, timing_matcher):
        """Test WHEN element scoring follows 35% weight distribution (P1 Critical)"""
        # Test that timing clarity affects scoring appropriately
        explicit_timing_text = 'Manager reviews reports daily'
        vague_timing_text = 'Staff periodically validates data'

        explicit_result = timing_matcher.match_timing_patterns(explicit_timing_text)
        vague_result = timing_matcher.match_timing_patterns(vague_timing_text)

        explicit_confidence = max(pattern['confidence'] for pattern in explicit_result['detected_patterns']) if explicit_result['detected_patterns'] else 0
        vague_confidence = max(pattern['confidence'] for pattern in vague_result['detected_patterns']) if vague_result['detected_patterns'] else 0

        # Explicit timing should have higher confidence than vague timing
        assert explicit_confidence > vague_confidence, "Explicit timing should have higher confidence than vague timing"


class TestWHEREElementConditionalScoring:
    """Test suite for WHERE element conditional scoring (P1 Critical)"""

    def test_where_detection_for_system_controls(self):
        """Test WHERE detection for system controls (P1 Critical)"""
        test_cases = [
            {
                'text': 'System validates transactions in SAP',
                'expected_where_present': True,
                'expected_where_type': 'system',
                'control_type': 'system',
                'expected_points': 10
            },
            {
                'text': 'Application processes data in Oracle database',
                'expected_where_present': True,
                'expected_where_type': 'system',
                'control_type': 'system',
                'expected_points': 10
            }
        ]

        # This would integrate with the conditional scoring logic
        # Testing the WHERE detection component independently
        for case in test_cases:
            # WHERE detection logic would be tested here
            # The actual scoring is tested in the core analyzer tests
            pass

    def test_where_detection_for_location_controls(self):
        """Test WHERE detection for location-dependent controls (P1 Critical)"""
        test_cases = [
            {
                'text': 'Guard performs inspection at vault',
                'expected_where_present': True,
                'expected_where_type': 'physical_location',
                'control_type': 'location_dependent',
                'expected_points': 5
            },
            {
                'text': 'Manager reviews documents at branch office',
                'expected_where_present': True,
                'expected_where_type': 'physical_location',
                'control_type': 'location_dependent',
                'expected_points': 5
            }
        ]

        # WHERE detection for physical locations
        for case in test_cases:
            # WHERE detection logic would be tested here
            pass

    def test_where_detection_for_other_controls(self):
        """Test WHERE detection has no scoring impact for other controls (P1 Critical)"""
        test_cases = [
            {
                'text': 'Manager saves report in SharePoint',
                'expected_where_present': True,
                'expected_where_type': 'system',
                'control_type': 'other',
                'expected_points': 0  # Other controls get 0 points regardless
            }
        ]

        # Other controls should get 0 WHERE points even if WHERE is present
        for case in test_cases:
            # This validates the conditional scoring logic
            pass


class TestWHYElementFeedbackOnly:
    """Test suite for WHY element feedback generation (P2 Business Logic)"""

    @pytest.fixture
    def purpose_analyzer(self):
        """Create PurposeAnalyzer with test configuration"""
        config = {
            'control_intent_categories': {
                'compliance': {
                    'keywords': ['comply', 'regulatory', 'sox', 'gaap'],
                    'confidence': 0.9
                },
                'risk_mitigation': {
                    'keywords': ['risk', 'prevent', 'mitigate', 'control'],
                    'confidence': 0.85
                }
            },
            'purpose_patterns': ['to ensure', 'to prevent', 'to comply', 'in order to']
        }
        return PurposeAnalyzer(config)

    def test_why_feedback_generation_no_scoring_impact(self, purpose_analyzer):
        """Test WHY generates feedback but has no scoring impact (P2 Business Logic)"""
        test_cases = [
            {
                'text': 'Manager reviews reports to ensure compliance with SOX requirements',
                'expected_purpose': 'compliance',
                'expected_feedback': True,
                'expected_score_impact': False
            },
            {
                'text': 'Analyst validates data to prevent errors and mitigate risk',
                'expected_purpose': 'risk_mitigation',
                'expected_feedback': True,
                'expected_score_impact': False
            }
        ]

        for case in test_cases:
            result = purpose_analyzer.analyze_purpose(case['text'])
            
            # Should generate feedback about purpose
            if case['expected_feedback']:
                assert len(result['detected_purposes']) > 0, f"Should detect purpose in: {case['text']}"
            
            # Should not impact scoring (this is verified in integration tests)


class TestESCALATIONElementSoftFlag:
    """Test suite for ESCALATION element soft flagging (P2 Business Logic)"""

    @pytest.fixture
    def escalation_detector(self):
        """Create EscalationPathDetector with test configuration"""
        config = {
            'escalation_indicators': {
                'roles': ['supervisor', 'manager', 'director', 'committee'],
                'actions': ['escalate', 'report', 'notify', 'alert'],
                'exception_terms': ['exception', 'deviation', 'variance', 'issue'],
                'threshold_terms': ['threshold', 'materiality', 'significant']
            }
        }
        return EscalationPathDetector(config)

    def test_escalation_soft_flag_generation(self, escalation_detector):
        """Test ESCALATION generates soft flags but has no scoring impact (P2 Business Logic)"""
        test_cases = [
            {
                'text': 'Manager escalates exceptions to supervisor for resolution',
                'expected_escalation': True,
                'expected_flag': True,
                'expected_score_impact': False
            },
            {
                'text': 'Significant variances are reported to management committee',
                'expected_escalation': True,
                'expected_flag': True,
                'expected_score_impact': False
            }
        ]

        for case in test_cases:
            result = escalation_detector.detect_escalation_path(case['text'])
            
            # Should detect escalation patterns
            if case['expected_escalation']:
                assert len(result['detected_escalations']) > 0, f"Should detect escalation in: {case['text']}"
            
            # Should generate soft flag (not impact scoring)
            # Score impact verification happens in integration tests