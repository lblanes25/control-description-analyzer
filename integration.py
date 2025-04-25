#!/usr/bin/env python3
# Example integration of enhanced visualization with audit control analysis

import argparse
import os
import sys
from control_analyzer import EnhancedControlAnalyzer
from visualization import generate_core_visualizations  # Import the enhanced visualization module


def main():
    """Main function to run control analysis with enhanced visualizations"""
    parser = argparse.ArgumentParser(
        description='Enhanced Control Description Analyzer with interactive visualizations')
    parser.add_argument('file', help='Excel file with control descriptions')
    parser.add_argument('--id-column', default='Control_ID', help='Column containing control IDs')
    parser.add_argument('--desc-column', default='Control_Description', help='Column containing control descriptions')
    parser.add_argument('--freq-column', help='Column containing frequency values for validation')
    parser.add_argument('--type-column', help='Column containing control type values for validation')
    parser.add_argument('--risk-column', help='Column containing risk descriptions for alignment')
    parser.add_argument('--output-file', help='Output Excel file path')
    parser.add_argument('--config', help='Path to configuration file (YAML)')
    parser.add_argument('--open-dashboard', action='store_true',
                        help='Automatically open the dashboard in default browser')

    args = parser.parse_args()

    # Set default output filename if not provided
    if not args.output_file:
        base_name = os.path.splitext(args.file)[0]
        args.output_file = f"{base_name}_analysis_results.xlsx"

    # Create analyzer instance
    analyzer = EnhancedControlAnalyzer(args.config)

    try:
        # Run the analysis
        print(f"Analyzing controls from {args.file}...")

        results = analyzer.analyze_file(
            args.file,
            args.id_column,
            args.desc_column,
            args.freq_column,
            args.type_column,
            args.risk_column,
            args.output_file
        )

        # Generate enhanced visualizations with filters
        print("Generating interactive visualizations...")
        vis_dir = os.path.splitext(args.output_file)[0] + "_visualizations"

        # Add metadata for audit leaders if missing
        # In a real implementation, this might come from a config file or be extracted from the Excel
        if results:
            # Check if we need to add audit leaders
            need_leaders = True
            for r in results:
                if r.get("Audit Leader") or (r.get("metadata") and r.get("metadata").get("Audit Leader")):
                    need_leaders = False
                    break

            # Add sample audit leaders if needed
            if need_leaders:
                print("No audit leader information found. Adding sample audit leaders for demonstration.")
                leaders = ["Sarah Johnson", "Michael Chen", "David Rodriguez", "Jennifer Taylor"]
                for i, r in enumerate(results):
                    leader_index = i % len(leaders)
                    if not r.get("metadata"):
                        r["metadata"] = {}
                    r["metadata"]["Audit Leader"] = leaders[leader_index]

        # Generate the visualizations
        output_files = generate_core_visualizations(results, vis_dir)

        print(f"Analysis complete. Results saved to {args.output_file}")
        print(f"Visualizations saved to {vis_dir}")

        # Open dashboard if requested
        if args.open_dashboard and "dashboard" in output_files:
            dashboard_path = output_files["dashboard"]
            print(f"Opening dashboard: {dashboard_path}")

            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())