#!/usr/bin/env python3
"""
Unit Tests for Control Type Classification (Priority 1 - Critical)

Tests the control type classification system including:
- Control type determination logic
- Manual control upgrade logic  
- System context detection
- Location context scoring
"""

import pytest
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.analyzers.control_classifier import ControlTypeClassifier, classify_control_type


class TestControlTypeClassifier:
    """Test suite for ControlTypeClassifier"""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance with test configuration"""
        config = {
            'classification': {
                'control_participating_verbs': [
                    'calculates', 'validates', 'approves', 'alerts', 'flags',
                    'reconciles', 'generates', 'processes', 'identifies', 'matches',
                    'automatically'
                ],
                'documentation_verbs': [
                    'saves', 'stores', 'documents', 'records', 'enters',
                    'uploads', 'maintains', 'tracks', 'files'
                ],
                'system_names': [
                    'sap', 'oracle', 'peoplesoft', 'sharepoint', 'teams', 'application', 'system'
                ],
                'system_context_weight': 2,
                'location_context_weight': 1
            }
        }
        return ControlTypeClassifier(config)

    @pytest.fixture
    def test_controls(self):
        """Test control library for classification testing"""
        return {
            'automated_control': {
                'description': 'Automated system generates daily reports',
                'automation': 'automated',
                'expected_type': 'system',
                'expected_upgraded': False
            },
            'system_upgrade': {
                'description': 'System validates transaction limits and flags exceptions for manager review',
                'automation': 'manual',
                'expected_type': 'system',
                'expected_upgraded': True
            },
            'documentation_only': {
                'description': 'Branch manager reviews daily exception report and saves findings in SharePoint',
                'automation': 'manual',
                'expected_type': 'other',
                'expected_upgraded': False
            },
            'location_dependent': {
                'description': 'Security guard performs physical vault inspection',
                'automation': 'manual',
                'expected_type': 'location_dependent',
                'expected_upgraded': False
            },
            'hybrid_system_prominent': {
                'description': 'System calculates balances and analyst validates results at branch office',
                'automation': 'hybrid',
                'expected_type': 'system'
            },
            'hybrid_location_prominent': {
                'description': 'Manager physically inspects vault using system checklist',
                'automation': 'hybrid',
                'expected_type': 'location_dependent'
            },
            'plain_manual': {
                'description': 'Senior management reviews quarterly risk reports',
                'automation': 'manual',
                'expected_type': 'other',
                'expected_upgraded': False
            }
        }

    def test_automated_control_classification(self, classifier, test_controls):
        """Test automated controls are classified as system (P1 Critical)"""
        control = test_controls['automated_control']
        
        result = classifier.classify_control(
            control['description'],
            control['automation']
        )
        
        assert result['final_type'] == control['expected_type']
        assert result['upgraded'] == control['expected_upgraded']
        assert 'Automated' in result['reasoning'][0]

    def test_manual_control_upgrade_logic(self, classifier, test_controls):
        """Test manual control upgrade to hybrid (P1 Critical)"""
        # Test system upgrade case
        control = test_controls['system_upgrade']
        
        result = classifier.classify_control(
            control['description'],
            control['automation']
        )
        
        assert result['final_type'] == control['expected_type']
        assert result['upgraded'] == control['expected_upgraded']
        assert 'upgraded' in result['reasoning'][0].lower()

    def test_documentation_only_no_upgrade(self, classifier, test_controls):
        """Test documentation-only controls remain manual (P1 Critical)"""
        control = test_controls['documentation_only']
        
        result = classifier.classify_control(
            control['description'],
            control['automation']
        )
        
        assert result['final_type'] == control['expected_type']
        assert result['upgraded'] == control['expected_upgraded']

    def test_location_dependent_classification(self, classifier, test_controls):
        """Test location-dependent control classification (P1 Critical)"""
        control = test_controls['location_dependent']
        
        result = classifier.classify_control(
            control['description'],
            control['automation']
        )
        
        assert result['final_type'] == control['expected_type']
        assert result['upgraded'] == control['expected_upgraded']

    def test_hybrid_prominence_analysis(self, classifier, test_controls):
        """Test hybrid control prominence determination (P1 Critical)"""
        # Test system-prominent hybrid
        control = test_controls['hybrid_system_prominent']
        
        result = classifier.classify_control(
            control['description'],
            control['automation']
        )
        
        assert result['final_type'] == control['expected_type']
        assert result['system_score'] > result['location_score']
        
        # Test location-prominent hybrid
        control = test_controls['hybrid_location_prominent']
        
        result = classifier.classify_control(
            control['description'],
            control['automation']
        )
        
        assert result['final_type'] == control['expected_type']

    def test_control_participating_verb_detection(self, classifier):
        """Test detection of control-participating verbs (P1 Critical)"""
        test_cases = [
            {
                'description': 'System validates transaction amounts',
                'automation': 'manual',
                'should_upgrade': True,
                'verb': 'validates'
            },
            {
                'description': 'System calculates interest daily',
                'automation': 'manual', 
                'should_upgrade': True,
                'verb': 'calculates'
            },
            {
                'description': 'Application flags suspicious transactions',
                'automation': 'manual',
                'should_upgrade': True,
                'verb': 'flags'
            },
            {
                'description': 'System automatically processes payments',
                'automation': 'manual',
                'should_upgrade': True,
                'verb': 'automatically'
            }
        ]
        
        for case in test_cases:
            result = classifier.classify_control(
                case['description'],
                case['automation']
            )
            
            if case['should_upgrade']:
                assert result['upgraded'], f"Should upgrade for verb '{case['verb']}'"
                assert result['final_type'] in ['system', 'location_dependent']

    def test_documentation_verb_filtering(self, classifier):
        """Test documentation verbs don't trigger upgrade (P1 Critical)"""
        test_cases = [
            {
                'description': 'Manager saves report in SharePoint',
                'automation': 'manual',
                'should_upgrade': False,
                'verb': 'saves'
            },
            {
                'description': 'Analyst stores documents in system',
                'automation': 'manual',
                'should_upgrade': False,
                'verb': 'stores'
            },
            {
                'description': 'User records transactions in database',
                'automation': 'manual',
                'should_upgrade': False,
                'verb': 'records'
            },
            {
                'description': 'Staff uploads files to portal',
                'automation': 'manual',
                'should_upgrade': False,
                'verb': 'uploads'
            }
        ]
        
        for case in test_cases:
            result = classifier.classify_control(
                case['description'],
                case['automation']
            )
            
            if not case['should_upgrade']:
                assert not result['upgraded'], f"Should not upgrade for documentation verb '{case['verb']}'"

    def test_system_context_scoring(self, classifier):
        """Test system context prominence scoring (P1 Critical)"""
        high_system_context = 'System automatically calculates balances and generates reports in SAP'
        
        result = classifier.classify_control(high_system_context, 'hybrid')
        
        # Should have high system score due to:
        # - 'system' (weight 2)
        # - 'automatically' (weight 2) 
        # - 'calculates' (weight 2)
        # - 'generates' (weight 2)
        # - 'sap' (weight 2)
        assert result['system_score'] >= 6  # At least 3 indicators * weight 2

    def test_location_context_scoring(self, classifier):
        """Test location context prominence scoring (P1 Critical)"""
        high_location_context = 'Security guard performs physical vault inspection at branch office'
        
        result = classifier.classify_control(high_location_context, 'hybrid')
        
        # Should have location score due to:
        # - 'guard performs' (weight 1)
        # - 'physical vault' (weight 1)
        # - 'at branch' would match if patterns configured
        assert result['location_score'] >= 1

    def test_edge_case_handling(self, classifier):
        """Test edge cases and error handling (P1 Critical)"""
        # Empty description
        result = classifier.classify_control('', 'manual')
        assert result['final_type'] == 'other'
        assert result['upgraded'] == False
        
        # None description
        result = classifier.classify_control(None, 'manual')
        assert result['final_type'] == 'other'
        
        # Invalid automation field
        result = classifier.classify_control('Manager reviews reports', 'invalid')
        assert result['final_type'] in ['system', 'location_dependent', 'other']

    def test_confidence_calculation(self, classifier):
        """Test classification confidence scoring (P1 Critical)"""
        # High confidence case - automated control
        result = classifier.classify_control(
            'Automated system processes transactions',
            'automated'
        )
        assert result['classification_confidence'] >= 0.8
        
        # Lower confidence case - unknown automation
        result = classifier.classify_control(
            'Manager reviews reports',
            None
        )
        assert result['classification_confidence'] <= 0.7

    def test_mixed_verb_scenarios(self, classifier):
        """Test scenarios with both control and documentation verbs (P1 Critical)"""
        # Should upgrade - has control verb even with documentation verb
        result = classifier.classify_control(
            'System validates transactions and saves results to database',
            'manual'
        )
        # Should upgrade because 'validates' is control-participating
        
        # Should not upgrade - only documentation verbs
        result = classifier.classify_control(
            'Manager saves and stores reports in SharePoint',
            'manual'
        )
        assert not result['upgraded']

    def test_system_name_detection(self, classifier):
        """Test system name detection with context (P1 Critical)"""
        test_cases = [
            {
                'description': 'SAP calculates balances automatically',
                'should_upgrade': True,
                'system': 'SAP'
            },
            {
                'description': 'Oracle validates data integrity',
                'should_upgrade': True,
                'system': 'Oracle'
            },
            {
                'description': 'Results saved in SharePoint folder',
                'should_upgrade': False,  # Documentation only
                'system': 'SharePoint'
            }
        ]
        
        for case in test_cases:
            result = classifier.classify_control(
                case['description'],
                'manual'
            )
            
            if case['should_upgrade']:
                assert result['upgraded'], f"Should upgrade for {case['system']} with control verb"
            else:
                assert not result['upgraded'], f"Should not upgrade for {case['system']} with documentation only"


class TestConvenienceFunction:
    """Test the convenience function for single control classification"""

    def test_classify_control_type_function(self):
        """Test the standalone classify_control_type function (P1 Critical)"""
        result = classify_control_type(
            'System validates transaction limits and flags exceptions',
            'manual'
        )
        
        # Should return same structure as class method
        required_fields = [
            'final_type', 'automation_field', 'upgraded', 
            'system_score', 'location_score', 'reasoning'
        ]
        
        for field in required_fields:
            assert field in result
        
        # Should classify as system due to upgrade
        assert result['final_type'] == 'system'
        assert result['upgraded'] == True

    def test_convenience_function_with_config(self):
        """Test convenience function with custom config (P1 Critical)"""
        custom_config = {
            'classification': {
                'control_participating_verbs': ['validates', 'processes'],
                'documentation_verbs': ['saves', 'stores']
            }
        }
        
        result = classify_control_type(
            'System validates data',
            'manual',
            config=custom_config
        )
        
        assert result['final_type'] == 'system'
        assert result['upgraded'] == True


class TestPerformanceAndReliability:
    """Test performance and reliability aspects"""

    @pytest.fixture
    def classifier(self):
        return ControlTypeClassifier()

    def test_classification_performance(self, classifier, benchmark):
        """Test classification performance (P1 Critical)"""
        long_description = """
        The Finance Manager performs comprehensive monthly reconciliation of all general ledger accounts 
        by comparing system-generated reports from SAP with external bank statements and subsidiary ledgers. 
        The process involves detailed validation of transaction accuracy, identification of discrepancies, 
        and resolution of any exceptions through investigation and corrective journal entries. 
        All findings are documented in SharePoint and escalated to the Controller for review and approval.
        """
        
        def classify_long_control():
            return classifier.classify_control(long_description, 'manual')
        
        # Should complete quickly even for long descriptions
        result = benchmark(classify_long_control)
        assert result['final_type'] in ['system', 'location_dependent', 'other']

    def test_repeated_classification_consistency(self, classifier):
        """Test that repeated classifications are consistent (P1 Critical)"""
        description = 'System validates transaction limits and flags exceptions'
        automation = 'manual'
        
        results = []
        for _ in range(5):
            result = classifier.classify_control(description, automation)
            results.append((result['final_type'], result['upgraded']))
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "Classification should be deterministic"