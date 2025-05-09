#!/usr/bin/env python3
# Enhanced integration script with batch processing and debug output for control analysis

import argparse
import os
import sys
import pandas as pd
import traceback
from control_analyzer import EnhancedControlAnalyzer
from visualization import generate_core_visualizations


def main():
    """Main function to run control analysis with batch processing support"""
    print("Starting Control Analyzer Integration Script")
    print("============================================")

    parser = argparse.ArgumentParser(
        description='Enhanced Control Description Analyzer with batch processing support')
    parser.add_argument('file', nargs='?', help='Excel file with control descriptions')
    parser.add_argument('--id-column', default=None, help='Column containing control IDs')
    parser.add_argument('--desc-column', default=None, help='Column containing control descriptions')
    parser.add_argument('--freq-column', help='Column containing frequency values for validation')
    parser.add_argument('--type-column', help='Column containing control type values for validation')
    parser.add_argument('--risk-column', help='Column containing risk descriptions for alignment')
    parser.add_argument('--audit-leader-column', help='Column containing Audit Leader information')
    parser.add_argument('--output-file', help='Output Excel file path')
    parser.add_argument('--config', help='Path to configuration file (YAML)')
    parser.add_argument('--open-dashboard', action='store_true',
                        help='Automatically open the dashboard in default browser')
    parser.add_argument('--disable-enhanced', action='store_true',
                        help='Disable enhanced detection modules')
    parser.add_argument('--debug', action='store_true', help='Show additional debug output')

    args = parser.parse_args()
    debug = args.debug

    print(f"Arguments parsed: {args}")

    # Set default output filename if not provided
    if args.file and not args.output_file:
        base_name = os.path.splitext(args.file)[0]
        args.output_file = f"{base_name}_analysis_results.xlsx"
        print(f"Setting default output file to: {args.output_file}")

    if not args.file:
        parser.print_help()
        return 1

    try:
        print(f"Creating analyzer with config: {args.config}")
        # Create analyzer instance
        analyzer = EnhancedControlAnalyzer(args.config)
        print("Analyzer created successfully")

        # Apply command-line flags
        if args.disable_enhanced:
            analyzer.use_enhanced_detection = False
            print("Enhanced detection modules disabled. Using base analysis only.")

        # Standard processing workflow
        print(f"Analyzing controls from {args.file}...")

        # Get column mappings for reference
        if debug and hasattr(analyzer, 'column_mappings'):
            print("Column mappings from config:")
            for key, value in analyzer.column_mappings.items():
                print(f"  {key}: {value}")

        # Check if Excel file exists
        if not os.path.exists(args.file):
            print(f"Error: Excel file '{args.file}' not found")
            return 1

        # Read Excel file to show columns
        try:
            df = pd.read_excel(args.file)
            print(f"Excel file columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error reading Excel file: {e}")

        # Run the analysis
        print("Starting analysis...")
        results = analyzer.analyze_file(
            args.file,
            args.id_column,
            args.desc_column,
            args.freq_column,
            args.type_column,
            args.risk_column,
            args.audit_leader_column,
            args.output_file
        )

        if results:
            print(f"Analysis complete - found {len(results)} results")

            # Print first result for debugging
            if debug and results:
                print("\nFirst result:")
                first = results[0]
                print(f"Control ID: {first.get('control_id')}")
                print(f"Score: {first.get('total_score'):.1f}")
                print(f"Category: {first.get('category')}")
                print(f"Missing Elements: {first.get('missing_elements')}")

            # Generate visualizations
            print("Generating visualizations...")
            vis_dir = os.path.splitext(args.output_file)[0] + "_visualizations"

            # Generate visualizations
            generate_core_visualizations(results, vis_dir)
            print(f"Visualizations saved to {vis_dir}")

            # Open dashboard if requested
            if args.open_dashboard:
                dashboard_path = os.path.join(vis_dir, "dashboard.html")
                if os.path.exists(dashboard_path):
                    print(f"Opening dashboard: {dashboard_path}")
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
                else:
                    print("Dashboard not found. Skipping open request.")

            print(f"Analysis complete. Results saved to {args.output_file}")
        else:
            print("No results returned from analyzer")

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    result = main()
    print(f"\nExiting with code {result}")
    sys.exit(result)