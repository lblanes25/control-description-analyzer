#!/usr/bin/env python3
"""
Simple diagnostic script to check Control Analyzer functionality
"""

import os
import sys
import pandas as pd
import traceback


def main():
    """Simple diagnostic to check Control Analyzer"""
    print("Control Analyzer Diagnostic")
    print("==========================")

    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python simple_diagnostic.py <excel_file> [config_file]")
        return 1

    excel_file = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Excel file: {excel_file}")
    print(f"Config file: {config_file}")

    # Check files exist
    if not os.path.exists(excel_file):
        print(f"ERROR: Excel file does not exist: {excel_file}")
        return 1

    if config_file and not os.path.exists(config_file):
        print(f"ERROR: Config file does not exist: {config_file}")
        return 1

    # Try to read Excel file
    try:
        print(f"Reading Excel file: {excel_file}")
        df = pd.read_excel(excel_file)
        print(f"Successfully read Excel file with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {', '.join(df.columns.tolist())}")

        # Try to access the first row to test column names
        if len(df) > 0:
            first_row = df.iloc[0]
            print("\nFirst row data:")
            for col in df.columns:
                print(f"  {col}: {first_row[col]}")

    except Exception as e:
        print(f"ERROR reading Excel file: {e}")
        traceback.print_exc()
        return 1

    # Try to read config file if provided
    if config_file:
        try:
            print(f"\nReading config file: {config_file}")
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Successfully read config file")

            # Check for column mappings
            if 'columns' in config:
                print("Column mappings in config:")
                for key, value in config['columns'].items():
                    if key in ['id', 'description', 'frequency', 'type', 'risk', 'audit_leader']:
                        print(f"  {key}: {value}")
            else:
                print("No column mappings found in config")

        except Exception as e:
            print(f"ERROR reading config file: {e}")
            traceback.print_exc()
            return 1

    # Try to import Control Analyzer
    try:
        print("\nImporting ControlAnalyzer...")
        from control_analyzer import EnhancedControlAnalyzer
        print("Successfully imported EnhancedControlAnalyzer")

        # Try to create analyzer instance
        print("Creating analyzer instance...")
        analyzer = EnhancedControlAnalyzer(config_file)
        print("Successfully created analyzer instance")

        # Check analyze_control method
        print("Testing analyze_control method...")
        has_analyze_control = hasattr(analyzer, 'analyze_control')
        print(f"Has analyze_control method: {has_analyze_control}")

        # Check analyze_file method
        print("Testing analyze_file method...")
        has_analyze_file = hasattr(analyzer, 'analyze_file')
        print(f"Has analyze_file method: {has_analyze_file}")

    except Exception as e:
        print(f"ERROR with Control Analyzer: {e}")
        traceback.print_exc()
        return 1

    print("\nDiagnostic complete - everything looks good!")
    print("To run a simple test, try:")

    # Suggest a command based on column names
    id_col = "Control_ID"
    desc_col = "Control_Description"

    if 'columns' in locals() and 'config' in locals():
        if 'id' in config.get('columns', {}):
            id_col = config['columns']['id']
        if 'description' in config.get('columns', {}):
            desc_col = config['columns']['description']

    output_file = os.path.splitext(excel_file)[0] + "_test_results.xlsx"

    if config_file:
        print(
            f"python -c \"from control_analyzer import EnhancedControlAnalyzer; analyzer = EnhancedControlAnalyzer('{config_file}'); analyzer.analyze_file('{excel_file}', '{id_col}', '{desc_col}', output_file='{output_file}')\"")
    else:
        print(
            f"python -c \"from control_analyzer import EnhancedControlAnalyzer; analyzer = EnhancedControlAnalyzer(); analyzer.analyze_file('{excel_file}', '{id_col}', '{desc_col}', output_file='{output_file}')\"")

    return 0


if __name__ == "__main__":
    sys.exit(main())