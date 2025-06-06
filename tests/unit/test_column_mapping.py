#!/usr/bin/env python3
"""
Test script specifically for column mapping functionality
"""

from src.utils.config_adapter import ConfigAdapter
from control_analyzer import EnhancedControlAnalyzer
import pandas as pd
import os
import sys


def test_column_mapping(config_file, excel_file):
    """Test column mapping functionality"""
    print("\n=== Testing Column Mapping ===")

    print(f"Config file: {config_file}")
    print(f"Excel file: {excel_file}")

    # Create ConfigAdapter and test column defaults
    print("\n--- ConfigAdapter Test ---")
    cm = ConfigAdapter(config_file)
    column_defaults = cm.get_column_defaults()
    print(f"Column defaults from ConfigAdapter: {column_defaults}")

    # Create EnhancedControlAnalyzer and test its initialization
    print("\n--- EnhancedControlAnalyzer Test ---")
    analyzer = EnhancedControlAnalyzer(config_file)

    # Print analyzer's column mappings
    print(f"Analyzer column_mappings: {analyzer.column_mappings}")
    print(f"Analyzer default_column_mappings: {analyzer.default_column_mappings}")

    # Test _get_column_name method
    print("\n--- _get_column_name Method Test ---")
    columns_to_test = ["id", "description", "frequency", "type", "risk", "audit_leader"]

    for column in columns_to_test:
        mapped_name = analyzer._get_column_name(column)
        print(f"Column '{column}' maps to '{mapped_name}'")

    # Test with override
    print("\nTesting with override values:")
    for column in columns_to_test:
        override = f"Custom_{column.capitalize()}"
        mapped_name = analyzer._get_column_name(column, override)
        print(f"Column '{column}' with override '{override}' maps to '{mapped_name}'")

    # Test with Excel file
    if os.path.exists(excel_file):
        print("\n--- Excel File Test ---")
        try:
            # Read the Excel file to check columns
            df = pd.read_excel(excel_file)
            print(f"Excel file columns: {df.columns.tolist()}")

            # Test actual mapping to Excel columns
            mapped_columns = {}
            for column in columns_to_test:
                mapped_name = analyzer._get_column_name(column)
                found = mapped_name in df.columns
                mapped_columns[column] = (mapped_name, found)

            print("\nMapping to actual Excel columns:")
            for column, (mapped_name, found) in mapped_columns.items():
                status = "✓" if found else "✗"
                print(f"{status} Column '{column}' maps to '{mapped_name}'")

            # Count failures
            failures = sum(1 for _, found in mapped_columns.values() if not found)
            if failures > 0:
                print(f"\nWarning: {failures} column mappings not found in Excel file")
            else:
                print("\nAll mapped columns found in Excel file!")

        except Exception as e:
            print(f"Error testing with Excel file: {e}")
    else:
        print(f"\nExcel file '{excel_file}' not found, skipping Excel test")

    return True


def main():
    # Default values
    config_file = "control_analyzer_config_final_with_columns.yaml"
    excel_file = "who_test_controls.xlsx"

    # Override with command line arguments if provided
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    if len(sys.argv) > 2:
        excel_file = sys.argv[2]

    print("Column Mapping Test")
    print("==================")

    try:
        # Test column mapping
        success = test_column_mapping(config_file, excel_file)

        if success:
            print("\nTest completed successfully!")
            return 0
        else:
            print("\nTest failed!")
            return 1
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())