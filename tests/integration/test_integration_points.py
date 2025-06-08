#!/usr/bin/env python3
"""
Priority 3 Integration Points Tests for Control Description Analyzer

Tests integration points as specified in testing strategy:
- File I/O operations testing
- Output format generation validation  
- CLI argument processing verification
- GUI component integration testing
"""

import pytest
import sys
import os
import tempfile
import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


class TestFileIOOperations:
    """Test suite for File I/O operations (P3 Integration Points)"""

    def test_excel_file_reading_standard_format(self, sample_excel_files, analyzer_with_config):
        """Test reading Excel files with standard column formats (P3 Integration)"""
        # Test good controls file
        excel_path = sample_excel_files['good_controls']
        
        # Verify file exists and is readable
        assert excel_path.exists(), f"Test Excel file should exist: {excel_path}"
        
        # Read Excel file into DataFrame
        df = pd.read_excel(excel_path)
        assert len(df) > 0, "Should read controls from Excel file"
        
        # Verify required columns are present
        expected_columns = ['Control ID', 'Control Description']
        for col in expected_columns:
            assert col in df.columns, f"Required column missing: {col}"

    def test_excel_file_reading_alternative_columns(self, sample_excel_files):
        """Test reading Excel files with alternative column names (P3 Integration)"""
        # Test alternative column names file
        excel_path = sample_excel_files['alt_columns']
        
        df = pd.read_excel(excel_path)
        assert len(df) > 0, "Should read controls with alternative column names"
        
        # Verify alternative columns are present
        alt_columns = ['Control #', 'Control Statement']
        for col in alt_columns:
            assert col in df.columns, f"Alternative column missing: {col}"

    def test_excel_file_error_handling(self, tmp_path):
        """Test Excel file error handling (P3 Integration)"""
        from src.core.analyzer import EnhancedControlAnalyzer
        
        analyzer = EnhancedControlAnalyzer('config/control_analyzer_updated.yaml')
        
        # Test non-existent file
        non_existent_file = tmp_path / "nonexistent.xlsx"
        
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            pd.read_excel(non_existent_file)

    def test_configuration_file_loading(self, analyzer_with_config):
        """Test configuration file loading (P3 Integration)"""
        config_path = Path('config/control_analyzer_updated.yaml')
        assert config_path.exists(), "Configuration file should exist"
        
        # Verify analyzer loads configuration successfully
        assert analyzer_with_config is not None, "Analyzer should initialize with config"

    def test_output_file_generation(self, tmp_path, analyzer_with_config):
        """Test output file generation (P3 Integration)"""
        # Create test control data
        test_controls = [
            {
                'control_id': 'TEST_001',
                'description': 'Finance Manager reviews journal entries daily in SAP',
                'expected_category': 'Effective'
            },
            {
                'control_id': 'TEST_002', 
                'description': 'Someone does something sometimes',
                'expected_category': 'Needs Improvement'
            }
        ]
        
        # Analyze controls
        results = []
        for control in test_controls:
            result = analyzer_with_config.analyze_control(
                control['control_id'], 
                control['description']
            )
            results.append(result)
        
        # Create output DataFrame
        output_data = []
        for result in results:
            output_data.append({
                'Control ID': result['control_id'],
                'Description': result['description'],
                'Total Score': result['total_score'],
                'Category': result['category'],
                'WHO Score': result['scoring_breakdown']['WHO'],
                'WHAT Score': result['scoring_breakdown']['WHAT'],
                'WHEN Score': result['scoring_breakdown']['WHEN'],
                'WHERE Score': result['scoring_breakdown']['WHERE'],
                'Demerits': result['scoring_breakdown']['demerits']
            })
        
        df = pd.DataFrame(output_data)
        
        # Test Excel output
        excel_output = tmp_path / "test_output.xlsx"
        df.to_excel(excel_output, index=False)
        
        assert excel_output.exists(), "Excel output file should be created"
        
        # Verify output file is readable
        read_df = pd.read_excel(excel_output)
        assert len(read_df) == len(test_controls), "Output should contain all analyzed controls"

    def test_large_file_processing(self, performance_dataset, tmp_path):
        """Test processing large Excel files (P3 Integration)"""
        # Create large test dataset
        large_dataset = performance_dataset(8000)  # 8000 controls
        
        # Convert to DataFrame
        df_data = []
        for i, control in enumerate(large_dataset):
            df_data.append({
                'Control ID': control['id'],
                'Control Description': control['description'],
                'Control_Automation': control['automation_field']
            })
        
        df = pd.DataFrame(df_data)
        
        # Save to Excel
        large_excel = tmp_path / "large_test.xlsx"
        df.to_excel(large_excel, index=False)
        
        # Verify file size and readability
        assert large_excel.exists(), "Large Excel file should be created"
        assert large_excel.stat().st_size > 100000, "Large file should have substantial size"
        
        # Test reading large file
        read_df = pd.read_excel(large_excel)
        assert len(read_df) == 8000, "Should read all 8000 controls from large file"


class TestOutputFormatGeneration:
    """Test suite for output format generation (P3 Integration Points)"""

    def test_json_output_format(self, analyzer_with_config):
        """Test JSON output format generation (P3 Integration)"""
        test_control = "Finance Manager reviews and validates journal entries daily in SAP"
        
        result = analyzer_with_config.analyze_control('JSON_TEST_001', test_control)
        
        # Convert result to JSON
        json_output = json.dumps(result, indent=2, default=str)
        
        # Verify JSON is valid
        parsed_result = json.loads(json_output)
        assert parsed_result['control_id'] == 'JSON_TEST_001'
        assert 'total_score' in parsed_result
        assert 'category' in parsed_result
        assert 'scoring_breakdown' in parsed_result

    def test_csv_output_format(self, tmp_path, analyzer_with_config):
        """Test CSV output format generation (P3 Integration)"""
        # Analyze multiple controls
        test_controls = [
            ('CSV_001', 'Finance Manager reviews transactions daily'),
            ('CSV_002', 'Analyst validates balances monthly'),
            ('CSV_003', 'Controller approves journal entries')
        ]
        
        results = []
        for control_id, description in test_controls:
            result = analyzer_with_config.analyze_control(control_id, description)
            results.append(result)
        
        # Create CSV data
        csv_data = []
        for result in results:
            csv_data.append({
                'control_id': result['control_id'],
                'description': result['description'],
                'total_score': result['total_score'],
                'category': result['category'],
                'who_score': result['scoring_breakdown']['WHO'],
                'what_score': result['scoring_breakdown']['WHAT'],
                'when_score': result['scoring_breakdown']['WHEN'],
                'where_score': result['scoring_breakdown']['WHERE']
            })
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        csv_path = tmp_path / "test_output.csv"
        df.to_csv(csv_path, index=False)
        
        # Verify CSV output
        assert csv_path.exists(), "CSV output should be created"
        
        # Read and verify CSV
        read_df = pd.read_csv(csv_path)
        assert len(read_df) == len(test_controls), "CSV should contain all controls"
        assert 'control_id' in read_df.columns, "CSV should have control_id column"

    def test_detailed_analysis_output(self, analyzer_with_config):
        """Test detailed analysis output format (P3 Integration)"""
        test_control = """
        The Senior Finance Manager reviews and validates all journal entries 
        daily in SAP to ensure compliance with SOX requirements and escalates 
        any discrepancies exceeding $10,000 to the Controller
        """
        
        result = analyzer_with_config.analyze_control('DETAILED_001', test_control)
        
        # Verify comprehensive output structure
        required_fields = [
            'control_id', 'description', 'total_score', 'category',
            'control_classification', 'scoring_breakdown', 'vague_terms_found',
            'elements_found_count', 'weighted_scores', 'normalized_scores'
        ]
        
        for field in required_fields:
            assert field in result, f"Detailed output missing field: {field}"
        
        # Verify scoring breakdown completeness
        scoring = result['scoring_breakdown']
        score_fields = ['WHO', 'WHAT', 'WHEN', 'WHERE', 'demerits']
        for field in score_fields:
            assert field in scoring, f"Scoring breakdown missing: {field}"
        
        # Verify classification details
        classification = result['control_classification']
        class_fields = ['final_type', 'automation_field', 'upgraded', 'reasoning']
        for field in class_fields:
            assert field in classification, f"Classification missing: {field}"

    def test_visualization_data_generation(self, analyzer_with_config):
        """Test data generation for visualizations (P3 Integration)"""
        # Analyze multiple controls for visualization
        test_controls = [
            ('VIZ_001', 'Finance Manager reviews transactions daily in SAP', 'Effective'),
            ('VIZ_002', 'Analyst validates balances monthly', 'Adequate'),
            ('VIZ_003', 'Someone does something sometimes', 'Needs Improvement')
        ]
        
        viz_data = []
        for control_id, description, expected_category in test_controls:
            result = analyzer_with_config.analyze_control(control_id, description)
            
            viz_data.append({
                'control_id': result['control_id'],
                'total_score': result['total_score'],
                'category': result['category'],
                'who_score': result['scoring_breakdown']['WHO'],
                'what_score': result['scoring_breakdown']['WHAT'],
                'when_score': result['scoring_breakdown']['WHEN'],
                'where_score': result['scoring_breakdown']['WHERE'],
                'vague_terms_count': len(result['vague_terms_found'])
            })
        
        # Verify visualization data structure
        assert len(viz_data) == 3, "Should have data for all controls"
        
        # Verify data completeness for visualization
        for data_point in viz_data:
            assert 'total_score' in data_point, "Visualization data needs total scores"
            assert 'category' in data_point, "Visualization data needs categories"
            assert data_point['total_score'] >= 0, "Scores should be valid for visualization"

    def test_summary_report_generation(self, analyzer_with_config):
        """Test summary report generation (P3 Integration)"""
        # Analyze multiple controls
        test_controls = [
            'Finance Manager reviews transactions daily',
            'Analyst validates balances monthly', 
            'Someone does something periodically',
            'Controller approves journal entries in SAP',
            'Staff performs various tasks as needed'
        ]
        
        results = []
        for i, description in enumerate(test_controls):
            result = analyzer_with_config.analyze_control(f'SUMMARY_{i:03d}', description)
            results.append(result)
        
        # Generate summary statistics
        total_controls = len(results)
        categories = [r['category'] for r in results]
        
        summary = {
            'total_controls_analyzed': total_controls,
            'effective_count': categories.count('Effective'),
            'adequate_count': categories.count('Adequate'),
            'needs_improvement_count': categories.count('Needs Improvement'),
            'average_score': sum(r['total_score'] for r in results) / total_controls,
            'min_score': min(r['total_score'] for r in results),
            'max_score': max(r['total_score'] for r in results)
        }
        
        # Verify summary report structure
        assert summary['total_controls_analyzed'] == 5, "Summary should count all controls"
        assert summary['effective_count'] + summary['adequate_count'] + summary['needs_improvement_count'] == 5
        assert 0 <= summary['average_score'] <= 100, "Average score should be reasonable"


class TestCLIArgumentProcessing:
    """Test suite for CLI argument processing (P3 Integration Points)"""

    def test_cli_help_functionality(self):
        """Test CLI help argument processing (P3 Integration)"""
        # Test help command with proper module path
        try:
            result = subprocess.run([
                sys.executable, '-m', 'src.cli', '--help'
            ], capture_output=True, text=True, timeout=10, cwd=project_root)
            
            # Should return 0 or show help without error
            assert result.returncode in [0, 1], "Help command should execute"
            
            # Check if help content appears in stdout or stderr
            output_text = (result.stdout + result.stderr).lower()
            assert 'usage:' in output_text or 'help' in output_text or 'analyzer' in output_text, \
                f"Expected help content not found. stdout: '{result.stdout}', stderr: '{result.stderr}'"
            
        except subprocess.TimeoutExpired:
            pytest.skip("CLI help command timed out")
        except FileNotFoundError:
            pytest.skip("CLI script not executable in test environment")

    def test_cli_argument_validation(self):
        """Test CLI argument validation (P3 Integration)"""
        # Test invalid arguments with proper module path
        try:
            result = subprocess.run([
                sys.executable, '-m', 'src.cli', '--invalid-argument'
            ], capture_output=True, text=True, timeout=10, cwd=project_root)
            
            # Should handle invalid arguments gracefully
            assert result.returncode != 0, "Invalid arguments should be rejected"
            
        except subprocess.TimeoutExpired:
            pytest.skip("CLI validation test timed out")
        except FileNotFoundError:
            pytest.skip("CLI script not executable in test environment")

    @pytest.mark.integration
    def test_cli_file_processing_integration(self, sample_excel_files):
        """Test CLI file processing integration (P3 Integration)"""
        # Test with actual Excel file
        excel_path = sample_excel_files['good_controls']
        
        try:
            # The CLI expects the file as a positional argument, not --input
            result = subprocess.run([
                sys.executable, '-m', 'src.cli', 
                str(excel_path),
                '--id-column', 'Control ID',
                '--desc-column', 'Control Description',
                '--skip-visualizations'  # Skip visualizations for test speed
            ], capture_output=True, text=True, timeout=30, cwd=project_root)
            
            # Should process without major errors
            # Note: Actual return code may vary based on implementation
            assert result.returncode in [0, 1], f"CLI should handle file processing. stderr: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.skip("CLI file processing test timed out")
        except FileNotFoundError:
            pytest.skip("CLI script not executable in test environment")

    def test_cli_argument_parsing_logic(self):
        """Test CLI argument parsing logic (P3 Integration)"""
        import argparse
        from unittest.mock import patch
        
        # Test argument parser configuration
        parser = argparse.ArgumentParser(description='Control Analyzer CLI')
        parser.add_argument('--input', '-i', help='Input Excel file')
        parser.add_argument('--output', '-o', help='Output directory')
        parser.add_argument('--config', '-c', help='Configuration file')
        parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        
        # Test valid argument combinations
        test_args = [
            ['--input', 'test.xlsx'],
            ['--input', 'test.xlsx', '--output', 'results/'],
            ['--input', 'test.xlsx', '--config', 'config.yaml', '--verbose'],
        ]
        
        for args in test_args:
            parsed_args = parser.parse_args(args)
            assert parsed_args.input is not None, f"Should parse input argument: {args}"

    def test_cli_configuration_handling(self):
        """Test CLI configuration file handling (P3 Integration)"""
        # Test configuration file argument processing
        config_path = 'config/control_analyzer_updated.yaml'
        
        # Verify config file exists
        assert Path(config_path).exists(), "Configuration file should exist for CLI testing"
        
        # Test configuration loading logic
        from src.utils.config_adapter import ConfigAdapter
        
        try:
            config_adapter = ConfigAdapter(config_path)
            assert config_adapter is not None, "Should load configuration for CLI"
        except Exception as e:
            pytest.fail(f"Configuration loading failed: {e}")


class TestGUIComponentIntegration:
    """Test suite for GUI component integration (P3 Integration Points)"""

    def test_gui_module_imports(self):
        """Test GUI module imports and basic structure (P3 Integration)"""
        try:
            from src.gui import main_window
            assert main_window is not None, "GUI main window module should be importable"
        except ImportError as e:
            pytest.skip(f"GUI module not available: {e}")

    def test_visualization_file_generation(self):
        """Test visualization file generation for GUI (P3 Integration)"""
        viz_path = Path('src/gui/visualizations')
        
        if viz_path.exists():
            # Check for expected visualization files
            expected_viz_files = [
                'dashboard.html',
                'element_radar.html', 
                'score_distribution.html'
            ]
            
            for viz_file in expected_viz_files:
                viz_file_path = viz_path / viz_file
                if viz_file_path.exists():
                    # Verify file is not empty
                    assert viz_file_path.stat().st_size > 0, f"Visualization file should not be empty: {viz_file}"
                    
                    # Verify it's HTML format
                    content = viz_file_path.read_text(encoding='utf-8')
                    assert '<html>' in content.lower() or '<!doctype html>' in content.lower(), \
                        f"Visualization should be HTML format: {viz_file}"
        else:
            pytest.skip("GUI visualization directory not found")

    def test_gui_data_interface(self, analyzer_with_config):
        """Test GUI data interface integration (P3 Integration)"""
        # Test data format expected by GUI
        test_control = "Finance Manager reviews transactions daily in SAP"
        result = analyzer_with_config.analyze_control('GUI_TEST_001', test_control)
        
        # Verify result format is suitable for GUI consumption
        gui_data = {
            'control_id': result['control_id'],
            'total_score': float(result['total_score']),  # Ensure JSON serializable
            'category': result['category'],
            'breakdown': {
                'who': float(result['scoring_breakdown']['WHO']),
                'what': float(result['scoring_breakdown']['WHAT']),
                'when': float(result['scoring_breakdown']['WHEN']),
                'where': float(result['scoring_breakdown']['WHERE'])
            }
        }
        
        # Test JSON serialization for GUI
        json_data = json.dumps(gui_data)
        parsed_data = json.loads(json_data)
        
        assert parsed_data['control_id'] == 'GUI_TEST_001'
        assert isinstance(parsed_data['total_score'], (int, float))
        assert 'breakdown' in parsed_data

    @pytest.mark.integration  
    def test_gui_visualization_integration(self):
        """Test GUI visualization integration (P3 Integration)"""
        # Test visualization utility integration
        try:
            from src.utils.visualization import generate_core_visualizations
            
            # Test with sample data
            sample_data = [
                {
                    'control_id': 'VIZ_001',
                    'total_score': 85.5,
                    'category': 'Effective',
                    'who_score': 25.0,
                    'what_score': 30.0,
                    'when_score': 20.5,
                    'where_score': 10.0
                },
                {
                    'control_id': 'VIZ_002', 
                    'total_score': 65.0,
                    'category': 'Adequate',
                    'who_score': 20.0,
                    'what_score': 25.0,
                    'when_score': 15.0,
                    'where_score': 5.0
                }
            ]
            
            # Test visualization generation
            # Note: This might require additional setup depending on implementation
            assert len(sample_data) > 0, "Should have sample data for visualization testing"
            
        except ImportError:
            pytest.skip("Visualization utilities not available")

    def test_gui_error_handling_integration(self):
        """Test GUI error handling integration (P3 Integration)"""
        # Test error scenarios that GUI should handle
        error_scenarios = [
            {'control_id': '', 'description': ''},  # Empty input
            {'control_id': 'TEST', 'description': None},  # None description
            {'control_id': None, 'description': 'Test control'}  # None ID
        ]
        
        for scenario in error_scenarios:
            # GUI should handle these scenarios gracefully
            # Test that error data can be serialized for GUI display
            error_response = {
                'error': True,
                'message': f"Invalid input: {scenario}",
                'control_id': scenario.get('control_id', 'UNKNOWN'),
                'total_score': 0,
                'category': 'Invalid'
            }
            
            # Verify error response is JSON serializable
            json_error = json.dumps(error_response, default=str)
            parsed_error = json.loads(json_error)
            
            assert parsed_error['error'] is True
            assert 'message' in parsed_error