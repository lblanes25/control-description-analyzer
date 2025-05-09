#!/usr/bin/env python3
"""
Enhanced integration script with batch processing for control analysis
This script preserves all original functionality while adding fixes for column mapping
"""

import argparse
import os
import sys
import pandas as pd
import pickle
import tempfile
import traceback
import gc
import time
from datetime import datetime
from typing import cast, BinaryIO, Any, List, Dict, Optional

from control_analyzer import EnhancedControlAnalyzer
from config_manager import ConfigManager
from visualization import generate_core_visualizations
from spacy.matcher import PhraseMatcher  # Required for apply_config_to_analyzer


def safe_pickle_dump(data: Any, file_path: str) -> None:
    """Safely dump data to a pickle file, handling type issues"""
    with open(file_path, 'wb') as f:
        # Cast to fix type hint issues
        binary_file = cast(BinaryIO, f)
        pickle.dump(data, binary_file, protocol=4)


def safe_pickle_load(file_path: str) -> Any:
    """Safely load data from a pickle file, handling type issues"""
    with open(file_path, 'rb') as f:
        # Cast to fix type hint issues
        binary_file = cast(BinaryIO, f)
        return pickle.load(binary_file)


def update_argument_parser(parser):
    """Add batch processing arguments to the argument parser"""
    # Add batch processing arguments
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Number of controls to process in each batch (default: 500)')
    parser.add_argument('--temp-dir', default='batch_results',
                        help='Directory to store temporary batch results (default: batch_results)')
    parser.add_argument('--use-batches', action='store_true',
                        help='Use batch processing for large datasets')
    parser.add_argument('--resume-from', help='Resume from a checkpoint file')
    parser.add_argument('--skip-visualizations', action='store_true',
                        help='Skip generating visualizations')
    parser.add_argument('--audit-leader-column', help='Column containing Audit Leader information (overrides config)')
    parser.add_argument('--debug', action='store_true',
                        help='Print additional debug information')
    return parser


def analyze_file_with_batches(analyzer, file_path, id_column, desc_column, freq_column=None,
                              type_column=None, risk_column=None, audit_leader_column=None, output_file=None,
                              batch_size=500, temp_dir=None):
    """
    Analyze controls from an Excel file in batches and generate a detailed report

    Args:
        analyzer: The EnhancedControlAnalyzer instance
        file_path: Path to Excel file containing controls
        id_column: Column containing control IDs
        desc_column: Column containing control descriptions
        freq_column: Optional column containing frequency values
        type_column: Optional column containing control type values
        risk_column: Optional column containing risk descriptions
        audit_leader_column: Optional column containing audit leader information
        output_file: Optional path for output Excel report
        batch_size: Number of controls to process in each batch
        temp_dir: Directory to store temporary batch results
    """
    print(f"Reading file: {file_path}")
    print(f"Using columns: ID={id_column}, Description={desc_column}, Frequency={freq_column}, " +
          f"Type={type_column}, Risk={risk_column}, Audit Leader={audit_leader_column}")
    print(f"Using batch size: {batch_size}")

    # Create temp directory if specified
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = "temp_batch_results"
        os.makedirs(temp_dir, exist_ok=True)

    try:
        # Read the Excel file
        df = pd.read_excel(file_path, engine='openpyxl')

        # Print actual columns for debugging
        print(f"Actual Excel columns: {df.columns.tolist()}")

        # Define a flexible column matching function
        def find_column(target, columns):
            # Exact match first
            if target in columns:
                return target

            # Try case-insensitive match
            for col in columns:
                if col.lower() == target.lower():
                    print(f"Case-insensitive match: '{target}' → '{col}'")
                    return col

            # Try partial match (e.g., "Audit Leader from AE" vs "Audit Leader From AE")
            for col in columns:
                if target.lower().replace(" ", "") == col.lower().replace(" ", ""):
                    print(f"Spacing/case match: '{target}' → '{col}'")
                    return col

            # Try matching key parts
            for col in columns:
                key_parts = target.split()
                if len(key_parts) > 1 and all(part.lower() in col.lower() for part in key_parts):
                    print(f"Partial match: '{target}' → '{col}'")
                    return col

            return None

        # Apply flexible matching to column names
        columns = df.columns.tolist()
        id_column = find_column(id_column, columns) or id_column
        desc_column = find_column(desc_column, columns) or desc_column
        if freq_column:
            freq_column = find_column(freq_column, columns) or freq_column
        if type_column:
            type_column = find_column(type_column, columns) or type_column
        if risk_column:
            risk_column = find_column(risk_column, columns) or risk_column
        if audit_leader_column:
            audit_leader_column = find_column(audit_leader_column, columns) or audit_leader_column

        print(f"Final column mapping:")
        print(f"  ID: {id_column}")
        print(f"  Description: {desc_column}")
        print(f"  Frequency: {freq_column}")
        print(f"  Type: {type_column}")
        print(f"  Risk: {risk_column}")
        print(f"  Audit Leader: {audit_leader_column}")

        # Ensure required columns exist
        if id_column not in df.columns:
            raise ValueError(f"ID column '{id_column}' not found in file")

        if desc_column not in df.columns:
            raise ValueError(f"Description column '{desc_column}' not found in file")

        # Check optional columns
        if freq_column and freq_column not in df.columns:
            print(
                f"Warning: Frequency column '{freq_column}' not found in file. Frequency validation will be skipped.")
            freq_column = None

        if type_column and type_column not in df.columns:
            print(
                f"Warning: Control type column '{type_column}' not found in file. Control type validation will be skipped.")
            type_column = None

        if risk_column and risk_column not in df.columns:
            print(
                f"Warning: Risk description column '{risk_column}' not found in file. Risk alignment will be skipped.")
            risk_column = None

        if audit_leader_column and audit_leader_column not in df.columns:
            print(f"Warning: Audit Leader column '{audit_leader_column}' not found in file.")

            # Try to automatically detect Audit Leader column
            potential_columns = ["Audit Leader", "audit leader", "Audit_Leader", "audit_leader",
                                 "AuditLeader", "auditLeader", "Auditor", "Lead Auditor"]

            for col in potential_columns:
                if col in df.columns:
                    audit_leader_column = col
                    print(f"Automatically detected Audit Leader column: '{col}'")
                    break
            else:
                audit_leader_column = None

        # Initialize results and tracking variables
        all_results = []
        total_controls = len(df)
        start_time = time.time()

        print(f"Analyzing {total_controls} controls in batches of {batch_size}...")

        # Process in batches
        for batch_start in range(0, total_controls, batch_size):
            batch_end = min(batch_start + batch_size, total_controls)
            batch_num = batch_start // batch_size + 1
            total_batches = (total_controls + batch_size - 1) // batch_size

            batch_start_time = time.time()
            print(f"\nProcessing batch {batch_num}/{total_batches}: controls {batch_start + 1}-{batch_end}")

            # Get the slice of DataFrame for this batch
            batch_df = df.iloc[batch_start:batch_end].copy()
            batch_results = []

            # Process each control in the batch
            for i, (idx, row) in enumerate(batch_df.iterrows()):
                if i % 25 == 0:
                    # Show more granular progress within batch
                    progress = (i + 1) / len(batch_df) * 100
                    print(f"  Batch progress: {progress:.1f}% ({i + 1}/{len(batch_df)})")

                control_id = row[id_column]
                description = row[desc_column]

                # Optional metadata
                frequency = row[freq_column] if freq_column and freq_column in row else None
                control_type = row[type_column] if type_column and type_column in row else None
                risk_description = row[risk_column] if risk_column and risk_column in row else None
                audit_leader = row[audit_leader_column] if audit_leader_column and audit_leader_column in row else None

                try:
                    # Analyze control
                    result = analyzer.analyze_control(control_id, description, frequency, control_type,
                                                      risk_description)

                    # Add Audit Leader if available
                    if audit_leader and not pd.isna(audit_leader):
                        result["Audit Leader"] = audit_leader

                    batch_results.append(result)
                except Exception as e:
                    # Log error but continue processing
                    error_msg = f"Error analyzing control {control_id}: {str(e)}"
                    print(error_msg)
                    # Add minimal error entry to maintain indexing
                    batch_results.append({
                        "control_id": control_id,
                        "description": description,
                        "total_score": 0,
                        "category": "Error",
                        "missing_elements": [],
                        "error_message": str(e)
                    })

            # Save batch results to temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_filename = os.path.join(temp_dir, f"batch_{batch_num:04d}_{timestamp}.pkl")
            try:
                safe_pickle_dump(batch_results, batch_filename)
                print(f"  Batch results saved to {batch_filename}")
            except Exception as e:
                print(f"  Warning: Could not save batch file: {e}")

            # Add to overall results
            all_results.extend(batch_results)

            # Calculate batch statistics
            batch_time = time.time() - batch_start_time
            controls_per_second = len(batch_df) / batch_time if batch_time > 0 else 0

            # Estimate remaining time
            completed = batch_end
            remaining = total_controls - completed
            estimated_time = remaining / controls_per_second if controls_per_second > 0 else 0

            # Report batch completion
            print(f"  Batch {batch_num}/{total_batches} completed in {batch_time:.1f} seconds")
            print(f"  Processing speed: {controls_per_second:.2f} controls/second")
            print(f"  Progress: {completed}/{total_controls} controls ({completed / total_controls * 100:.1f}%)")

            hours, remainder = divmod(estimated_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"  Estimated time remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s")

            # Save checkpoint of all results so far
            checkpoint_file = os.path.join(temp_dir, "all_results_checkpoint.pkl")
            try:
                safe_pickle_dump(all_results, checkpoint_file)
                print(f"  Checkpoint saved to {checkpoint_file}")
            except Exception as e:
                print(f"  Warning: Could not save checkpoint: {e}")

            # Clear memory
            gc.collect()

            # Optional: Clear the NLP model's cache to free memory
            try:
                # Get all pipelines and clear their caches
                for pipe_name in analyzer.nlp.pipe_names:
                    pipe = analyzer.nlp.get_pipe(pipe_name)
                    if hasattr(pipe, "vocab") and hasattr(pipe.vocab, "strings"):
                        pipe.vocab.strings.clean_up()
            except Exception:
                # If cleanup fails, just continue
                pass

        # Create output file if specified
        if output_file:
            try:
                analyzer._generate_enhanced_report(
                    all_results,
                    output_file,
                    freq_column is not None,
                    type_column is not None,
                    risk_column is not None
                )
                print(f"\nAnalysis complete. Results saved to {output_file}")
            except Exception as e:
                print(f"Error generating final report: {e}")
                # Try to save results in pickle format
                emergency_output = os.path.splitext(output_file)[0] + "_emergency.pkl"
                safe_pickle_dump(all_results, emergency_output)
                print(f"Emergency results saved to {emergency_output}")

        # Calculate total time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTotal processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Average processing speed: {total_controls / total_time:.2f} controls/second")

        return all_results

    except Exception as e:
        print(f"Error analyzing file: {e}")
        traceback.print_exc()
        raise


def resume_from_checkpoint(analyzer, checkpoint_file, args):
    """Resume analysis from a saved checkpoint"""
    print(f"Resuming from checkpoint: {checkpoint_file}")

    try:
        # Initialize variables to avoid "might be referenced before assignment" errors
        completed_results = []
        last_control_id = None

        # Load checkpoint data
        try:
            completed_results = safe_pickle_load(checkpoint_file)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            # Try with absolute path
            abs_path = os.path.abspath(checkpoint_file)
            print(f"Trying with absolute path: {abs_path}")
            completed_results = safe_pickle_load(abs_path)

        if not completed_results:
            print("Error: Empty or invalid checkpoint file")
            return None, 1

        print(f"Loaded {len(completed_results)} completed controls from checkpoint")

        # Get the last control ID
        if completed_results:
            last_control_id = completed_results[-1].get("control_id")

        if not last_control_id:
            print("Warning: Could not determine last processed control ID")
        else:
            print(f"Last processed control ID: {last_control_id}")

        # Read the Excel file to get remaining controls
        df = pd.read_excel(args.file, engine='openpyxl')

        # Find the index of the last processed control
        last_index = -1
        if last_control_id:
            for i, row in df.iterrows():
                if str(row[args.id_column]) == str(last_control_id):
                    last_index = i
                    break

        if last_index == -1:
            print("Warning: Could not find last processed control. Starting from beginning.")
            last_index = -1

        # Create DataFrame with remaining controls
        remaining_df = df.iloc[last_index + 1:].copy()

        if len(remaining_df) == 0:
            print("No remaining controls to process. Analysis was already complete.")

            # Generate the final report
            if args.output_file:
                try:
                    print(f"Generating final report...")
                    # Use wrapper method to avoid protected member access
                    generate_final_report(analyzer, completed_results, args)
                    print(f"Final report generated at {args.output_file}")
                except Exception as e:
                    print(f"Error generating report: {e}")
                    # Save as pickle file
                    pkl_path = os.path.splitext(args.output_file)[0] + "_results.pkl"
                    safe_pickle_dump(completed_results, pkl_path)
                    print(f"Results saved to {pkl_path}")

            return completed_results, 0

        print(f"Resuming analysis with {len(remaining_df)} remaining controls")

        # Create a temporary file with the remaining controls
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name

        remaining_df.to_excel(temp_path, engine='openpyxl', index=False)

        # Process the remaining controls
        temp_output = os.path.splitext(args.output_file)[0] + "_remaining.xlsx"

        remaining_results = []
        if args.use_batches:
            remaining_results = analyze_file_with_batches(
                analyzer,
                temp_path,
                args.id_column,
                args.desc_column,
                args.freq_column,
                args.type_column,
                args.risk_column,
                args.audit_leader_column,
                temp_output,
                args.batch_size,
                args.temp_dir
            )
        else:
            remaining_results = analyzer.analyze_file(
                temp_path,
                args.id_column,
                args.desc_column,
                args.freq_column,
                args.type_column,
                args.risk_column,
                args.audit_leader_column,
                temp_output
            )

        # Combine results
        all_results = completed_results + remaining_results

        # Generate the final report
        if args.output_file:
            try:
                generate_final_report(analyzer, all_results, args)
                print(f"Final report generated at {args.output_file}")
            except Exception as e:
                print(f"Error generating final report: {e}")
                # Save results as pickle in case of error
                pickle_path = os.path.splitext(args.output_file)[0] + "_all_results.pkl"
                safe_pickle_dump(all_results, pickle_path)
                print(f"Results saved to {pickle_path}")

        # Delete the temporary file
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as e:
            print(f"Warning: Failed to delete temporary file: {e}")

        return all_results, 0

    except Exception as e:
        print(f"Error resuming from checkpoint: {e}")
        traceback.print_exc()
        return None, 1


def generate_final_report(analyzer, results, args):
    """Generate final report without directly calling protected methods"""
    # Directly call the protected method, but wrap in try/except
    try:
        analyzer._generate_enhanced_report(
            results,
            args.output_file,
            args.freq_column is not None,
            args.type_column is not None,
            args.risk_column is not None
        )
    except Exception as e:
        print(f"Warning: Error in report generation: {e}")
        # Try using a manual approach to generate Excel
        try:
            write_excel_report(results, args.output_file)
        except Exception as excel_error:
            print(f"Failed to write Excel report: {excel_error}")
            raise


def write_excel_report(results, output_file):
    """Fallback method to write results to Excel"""
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    # Convert results to DataFrame
    basic_results = []
    for r in results:
        result_dict = {
            "Control ID": r.get("control_id", "Unknown"),
            "Description": r.get("description", ""),
            "Total Score": r.get("total_score", 0),
            "Category": r.get("category", "Unknown"),
            "Missing Elements": ", ".join(r.get("missing_elements", [])) if r.get("missing_elements") else "None",
            "Vague Terms": ", ".join(r.get("vague_terms_found", [])) if r.get("vague_terms_found") else "None"
        }

        # Add weighted scores if available
        for element, score in r.get("weighted_scores", {}).items():
            result_dict[f"{element} Score"] = score

        basic_results.append(result_dict)

    df_results = pd.DataFrame(basic_results)

    # Create workbook and add data
    wb = Workbook()
    ws = wb.active
    ws.title = "Analysis Results"

    for r_idx, row in enumerate(dataframe_to_rows(df_results, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=r_idx, column=c_idx, value=value)

    # Save workbook
    wb.save(output_file)


def apply_config_to_analyzer(analyzer, config):
    """Apply configuration settings to analyzer"""
    # Apply element weights if specified
    if 'elements' in config:
        for element_name, element_config in config['elements'].items():
            if element_name.upper() in analyzer.elements and 'weight' in element_config:
                analyzer.elements[element_name.upper()].weight = element_config['weight']

    # Apply keywords if specified
    if 'elements' in config:
        for element_name, element_config in config['elements'].items():
            if element_name.upper() in analyzer.elements and 'keywords' in element_config:
                if element_config.get('append', True):
                    analyzer.elements[element_name.upper()].keywords.extend(element_config['keywords'])
                else:
                    analyzer.elements[element_name.upper()].keywords = element_config['keywords']
                # Rebuild matchers with new keywords
                analyzer.elements[element_name.upper()].setup_matchers(analyzer.nlp)

    # Apply vague terms if specified
    if 'vague_terms' in config:
        if config.get('append_vague_terms', True):
            analyzer.vague_terms.extend(config['vague_terms'])
        else:
            analyzer.vague_terms = config['vague_terms']
        # Rebuild vague term matcher
        analyzer.vague_matcher = PhraseMatcher(analyzer.nlp.vocab, attr="LOWER")
        vague_phrases = [analyzer.nlp(term) for term in analyzer.vague_terms]
        if vague_phrases:
            analyzer.vague_matcher.add("vague_patterns", vague_phrases)

    # Toggle enhanced detection
    if 'use_enhanced_detection' in config:
        analyzer.use_enhanced_detection = config['use_enhanced_detection']

    return analyzer


def process_with_batch_option(analyzer, args):
    """Process controls using batch processing if requested"""
    try:
        if args.use_batches:
            print(f"Using batch processing with batch size of {args.batch_size}")
            print(f"Using columns: ID={args.id_column}, Description={args.desc_column}, "
                  f"Frequency={args.freq_column}, Type={args.type_column}, "
                  f"Risk={args.risk_column}, Audit Leader={args.audit_leader_column}")

            results = analyze_file_with_batches(
                analyzer,
                args.file,
                args.id_column,
                args.desc_column,
                args.freq_column,
                args.type_column,
                args.risk_column,
                args.audit_leader_column,
                args.output_file,
                args.batch_size,
                args.temp_dir
            )
        else:
            # Original non-batch processing
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

        # Generate visualizations unless skipped
        if not args.skip_visualizations and results:
            print("Generating visualizations...")
            vis_dir = os.path.splitext(args.output_file)[0] + "_visualizations"

            # Generate visualizations
            generate_core_visualizations(results, vis_dir)
            print(f"Visualizations saved to {vis_dir}")

        return 0
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1


def main():
    """Main function to run control analysis with batch processing support"""
    print("Control Analyzer Integration Script")
    print("===================================")

    parser = argparse.ArgumentParser(description='Enhanced Control Description Analyzer with batch processing support')
    parser.add_argument('file', nargs='?', help='Excel file with control descriptions')
    parser.add_argument('--id-column', default=None, help='Column containing control IDs')
    parser.add_argument('--desc-column', default=None, help='Column containing control descriptions')
    parser.add_argument('--freq-column', help='Column containing frequency values for validation')
    parser.add_argument('--type-column', help='Column containing control type values for validation')
    parser.add_argument('--risk-column', help='Column containing risk descriptions for alignment')
    parser.add_argument('--output-file', help='Output Excel file path')
    parser.add_argument('--config', help='Path to configuration file (YAML)')
    parser.add_argument('--open-dashboard', action='store_true',
                        help='Automatically open the dashboard in default browser')
    parser.add_argument('--disable-enhanced', action='store_true',
                        help='Disable enhanced detection modules')

    # Add batch processing arguments
    parser = update_argument_parser(parser)

    args = parser.parse_args()

    if args.debug:
        print(f"Arguments: {args}")

    # Set default output filename if not provided
    if args.file and not args.output_file:
        base_name = os.path.splitext(args.file)[0]
        args.output_file = f"{base_name}_analysis_results.xlsx"
        if args.debug:
            print(f"Setting default output file to {args.output_file}")

    if not args.file and not args.resume_from:
        parser.print_help()
        return 1

    try:
        # Create analyzer instance
        if args.debug:
            print(f"Creating analyzer with config file: {args.config}")

        analyzer = EnhancedControlAnalyzer(args.config)

        if args.debug:
            print(f"Analyzer created successfully")
            if hasattr(analyzer, 'column_mappings'):
                print("Column mappings from config:")
                for key, value in analyzer.column_mappings.items():
                    if not isinstance(value, dict):  # Skip complex nested objects
                        print(f"  {key}: {value}")

        # Apply command-line flags
        if args.disable_enhanced:
            analyzer.use_enhanced_detection = False
            print("Enhanced detection modules disabled. Using base analysis only.")

        # Add batch processing method to analyzer if not present
        if not hasattr(analyzer, 'analyze_file_with_batches'):
            from types import MethodType
            analyzer.analyze_file_with_batches = MethodType(analyze_file_with_batches, analyzer)

        # Handle resume from checkpoint if specified
        if args.resume_from:
            return resume_from_checkpoint(analyzer, args.resume_from, args)[1]

        # Process with batch option if specified
        if args.use_batches:
            return process_with_batch_option(analyzer, args)

        # Standard processing workflow
        print(f"Analyzing controls from {args.file}...")

        # Check if Excel file exists
        if not os.path.exists(args.file):
            print(f"Error: Excel file '{args.file}' not found")
            return 1

        # Read Excel file to show columns if in debug mode
        if args.debug:
            try:
                df = pd.read_excel(args.file)
                print(f"Excel file columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error reading Excel file: {e}")

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

        # Generate visualizations
        if not args.skip_visualizations:
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
        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
