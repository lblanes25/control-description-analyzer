#!/usr/bin/env python3
"""
Test Configuration and Fixtures for Control Description Analyzer

Provides shared fixtures and helpers for all test modules as specified
in the testing strategy document.
"""

import pytest
import sys
import os
import yaml
import tempfile
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.core.analyzer import EnhancedControlAnalyzer
from src.analyzers.control_classifier import ControlTypeClassifier


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path"""
    return Path(project_root)


@pytest.fixture(scope="session")
def test_config_path(project_root_path):
    """Return path to test configuration file"""
    return project_root_path / 'config' / 'control_analyzer_updated.yaml'


@pytest.fixture
def analyzer_with_config(test_config_path):
    """
    Analyzer instance with test configuration
    
    Creates a fresh analyzer instance for each test to ensure isolation.
    Uses the actual configuration file for realistic testing.
    """
    return EnhancedControlAnalyzer(str(test_config_path))


@pytest.fixture
def control_classifier():
    """
    Control classifier instance with test configuration
    
    Provides a classifier instance with default test configuration
    for component-level testing.
    """
    config = {
        'classification': {
            'control_participating_verbs': [
                'calculates', 'validates', 'approves', 'alerts', 'flags',
                'reconciles', 'generates', 'processes', 'identifies', 'matches',
                'automatically', 'system'
            ],
            'documentation_verbs': [
                'saves', 'stores', 'documents', 'records', 'enters',
                'uploads', 'maintains', 'tracks', 'files'
            ],
            'system_names': [
                'sap', 'oracle', 'peoplesoft', 'jde', 'dynamics', 'netsuite',
                'sharepoint', 'teams', 'slack', 'confluence', 'servicenow',
                'tableau', 'power bi', 'excel', 'access'
            ],
            'system_context_weight': 2,
            'location_context_weight': 1
        }
    }
    return ControlTypeClassifier(config)


@pytest.fixture(scope="session")
def sample_control_library():
    """
    Load test controls from YAML fixture file
    
    Provides comprehensive test control library as specified in 
    testing strategy Section 5.1.
    """
    fixtures_path = Path(__file__).parent / 'fixtures' / 'test_controls.yaml'
    
    with open(fixtures_path, 'r', encoding='utf-8') as f:
        controls = yaml.safe_load(f)
    
    return controls


@pytest.fixture
def good_controls(sample_control_library):
    """Extract good controls from test library"""
    return sample_control_library['good_controls']


@pytest.fixture
def problematic_controls(sample_control_library):
    """Extract problematic controls from test library"""
    return sample_control_library['problematic_controls']


@pytest.fixture
def edge_cases(sample_control_library):
    """Extract edge case controls from test library"""
    return sample_control_library['edge_cases']


@pytest.fixture
def classification_test_cases(sample_control_library):
    """Extract classification test cases from test library"""
    return sample_control_library['classification_test_cases']


@pytest.fixture
def vague_term_test_cases(sample_control_library):
    """Extract vague term test cases from test library"""
    return sample_control_library['vague_term_test_cases']


@pytest.fixture
def performance_test_controls(sample_control_library):
    """Extract performance test controls from test library"""
    return sample_control_library['performance_test_controls']


@pytest.fixture
def element_detection_tests(sample_control_library):
    """Extract element detection test cases from test library"""
    return sample_control_library['element_detection_test_cases']


@pytest.fixture
def multi_control_tests(sample_control_library):
    """Extract multi-control test cases from test library"""
    return sample_control_library['multi_control_test_cases']


@pytest.fixture
def mock_excel_file(tmp_path):
    """
    Generate test Excel files with various structures
    
    Creates temporary Excel files for integration testing with
    different column layouts and data scenarios.
    """
    def _create_excel(controls_data: List[Dict[str, Any]], 
                     filename: str = "test_controls.xlsx",
                     column_mapping: Dict[str, str] = None) -> Path:
        """
        Create Excel file with specified controls data
        
        Args:
            controls_data: List of control dictionaries
            filename: Output filename  
            column_mapping: Custom column names (default uses standard names)
        
        Returns:
            Path to created Excel file
        """
        if column_mapping is None:
            column_mapping = {
                'control_id': 'Control ID',
                'description': 'Control Description', 
                'automation': 'Control_Automation',
                'process': 'Process',
                'owner': 'Control Owner'
            }
        
        # Create DataFrame
        df_data = []
        for i, control in enumerate(controls_data):
            row = {
                column_mapping['control_id']: control.get('id', f'CTRL_{i:03d}'),
                column_mapping['description']: control.get('description', ''),
                column_mapping['automation']: control.get('automation_field', 'manual'),
                column_mapping['process']: control.get('process', 'General'),
                column_mapping['owner']: control.get('owner', 'Finance Manager')
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save to Excel
        excel_path = tmp_path / filename
        df.to_excel(excel_path, index=False)
        
        return excel_path
    
    return _create_excel


@pytest.fixture
def sample_excel_files(mock_excel_file, good_controls, problematic_controls):
    """
    Create various sample Excel files for integration testing
    
    Returns dictionary of file paths for different test scenarios.
    """
    files = {}
    
    # Good controls file
    files['good_controls'] = mock_excel_file(
        good_controls, 
        'good_controls.xlsx'
    )
    
    # Problematic controls file
    files['problematic_controls'] = mock_excel_file(
        problematic_controls,
        'problematic_controls.xlsx'
    )
    
    # Mixed controls file
    mixed_controls = good_controls[:2] + problematic_controls[:2]
    files['mixed_controls'] = mock_excel_file(
        mixed_controls,
        'mixed_controls.xlsx'
    )
    
    # Alternative column names
    alt_columns = {
        'control_id': 'Control #',
        'description': 'Control Statement',
        'automation': 'Automation Type',
        'process': 'Business Process', 
        'owner': 'Responsible Party'
    }
    files['alt_columns'] = mock_excel_file(
        good_controls[:3],
        'alt_columns.xlsx',
        alt_columns
    )
    
    return files


@pytest.fixture
def performance_dataset():
    """
    Generate large dataset for performance testing
    
    Creates datasets of varying sizes as specified in testing strategy
    performance constraints.
    """
    def _generate_controls(count: int) -> List[Dict[str, Any]]:
        """Generate specified number of test controls"""
        controls = []
        
        templates = [
            "The {role} {action} {object} {frequency} in {system}",
            "{role} performs {action} of {object} {frequency}",
            "System {action} {object} and {role} validates results {frequency}",
            "{role} {action} {object} at {location} {frequency}"
        ]
        
        roles = ["Finance Manager", "Accounting Supervisor", "Internal Auditor", "Risk Analyst", "Controller"]
        actions = ["reviews", "validates", "reconciles", "analyzes", "approves"]
        objects = ["transactions", "reports", "balances", "statements", "entries"]
        frequencies = ["daily", "weekly", "monthly", "quarterly", "annually"]
        systems = ["SAP", "Oracle", "Excel", "SharePoint", "Teams"]
        locations = ["branch office", "headquarters", "data center", "vault", "facility"]
        
        for i in range(count):
            template = templates[i % len(templates)]
            
            control = {
                'id': f'PERF_{i:04d}',
                'description': template.format(
                    role=roles[i % len(roles)],
                    action=actions[i % len(actions)],
                    object=objects[i % len(objects)],
                    frequency=frequencies[i % len(frequencies)],
                    system=systems[i % len(systems)],
                    location=locations[i % len(locations)]
                ),
                'automation_field': ['manual', 'hybrid', 'automated'][i % 3]
            }
            controls.append(control)
        
        return controls
    
    return _generate_controls


@pytest.fixture
def benchmark_controls(performance_dataset):
    """
    Pre-generated control sets for benchmark testing
    
    Provides control sets of different sizes for performance validation.
    """
    return {
        'single': performance_dataset(1)[0],
        'small_batch': performance_dataset(10),
        'medium_batch': performance_dataset(100),
        'large_batch': performance_dataset(1000)
    }


# Helper functions for test assertions

def assert_control_result_structure(result: Dict[str, Any]):
    """
    Assert that control analysis result has required structure
    
    Validates the result dictionary contains all required fields
    as specified in the testing strategy.
    """
    required_fields = [
        'control_id', 'description', 'total_score', 'category',
        'control_classification', 'scoring_breakdown', 'vague_terms_found',
        'elements_found_count', 'weighted_scores', 'normalized_scores'
    ]
    
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    # Validate scoring breakdown structure
    scoring = result['scoring_breakdown']
    required_scoring_fields = ['WHO', 'WHAT', 'WHEN', 'WHERE', 'demerits']
    
    for field in required_scoring_fields:
        assert field in scoring, f"Missing scoring field: {field}"
    
    # Validate classification structure
    classification = result['control_classification']
    required_classification_fields = [
        'final_type', 'automation_field', 'upgraded', 
        'system_score', 'location_score', 'reasoning'
    ]
    
    for field in required_classification_fields:
        assert field in classification, f"Missing classification field: {field}"


def assert_score_ranges(result: Dict[str, Any]):
    """
    Assert that scores are within expected ranges
    
    Validates that all scores are within logical bounds.
    """
    # Total score should be non-negative (with demerits, could be negative but capped at 0)
    assert result['total_score'] >= 0, f"Total score should be non-negative: {result['total_score']}"
    
    # Individual element scores should be reasonable
    scoring = result['scoring_breakdown']
    
    # Core element scores should be between 0 and their weights
    assert 0 <= scoring['WHO'] <= 30, f"WHO score out of range: {scoring['WHO']}"
    assert 0 <= scoring['WHAT'] <= 35, f"WHAT score out of range: {scoring['WHAT']}"
    assert 0 <= scoring['WHEN'] <= 35, f"WHEN score out of range: {scoring['WHEN']}"
    
    # WHERE points should match conditional scoring rules
    control_type = result['control_classification']['final_type']
    where_points = scoring['WHERE']
    
    if control_type == 'system':
        assert where_points in [0, 10], f"System control WHERE points should be 0 or 10: {where_points}"
    elif control_type == 'location_dependent':
        assert where_points in [0, 5], f"Location control WHERE points should be 0 or 5: {where_points}"
    else:  # other
        assert where_points == 0, f"Other control WHERE points should be 0: {where_points}"
    
    # Demerits should be non-positive
    assert scoring['demerits'] <= 0, f"Demerits should be non-positive: {scoring['demerits']}"


def assert_category_consistency(result: Dict[str, Any]):
    """
    Assert that category assignment is consistent with score
    
    Validates that the assigned category matches the score thresholds.
    """
    score = result['total_score']
    category = result['category']
    
    if score >= 75:
        assert category == 'Effective', f"Score {score} should be Effective, got {category}"
    elif score >= 50:
        assert category == 'Adequate', f"Score {score} should be Adequate, got {category}"
    else:
        assert category == 'Needs Improvement', f"Score {score} should be Needs Improvement, got {category}"


@pytest.fixture
def assert_helpers():
    """
    Provide assertion helper functions for tests
    
    Returns dictionary of assertion functions for common validations.
    """
    return {
        'structure': assert_control_result_structure,
        'score_ranges': assert_score_ranges,
        'category_consistency': assert_category_consistency
    }


# Pytest configuration hooks

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add unit marker to unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to benchmark tests
        if "benchmark" in item.keywords or "performance" in item.name:
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker to tests that might take time
        if any(keyword in item.name.lower() for keyword in ['large', 'batch', '1000']):
            item.add_marker(pytest.mark.slow)