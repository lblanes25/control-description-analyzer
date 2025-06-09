#!/usr/bin/env python3
"""
Enhanced integration script with batch processing for control analysis - VERBOSE VERSION.

This script coordinates the execution of control description analysis,
supporting batch processing, checkpoint functionality, and visualization
generation. It serves as the main entry point for the control analyzer tool.

This is the verbose version with debug output enabled by default.
"""

import argparse
import gc
import logging
import os
import pickle
import platform
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union, cast

import pandas as pd

from src.core.analyzer import EnhancedControlAnalyzer
from src.utils.config_adapter import ConfigAdapter
from src.utils.visualization import generate_core_visualizations

# Set up module logger
logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str] = None, verbose: bool = False) -> None:
    """
    Configure the logging system for the application.

    Args:
        log_file: Optional path to log file
        verbose: Whether to use verbose (DEBUG) logging
    """
    # Determine log level - DEFAULT TO DEBUG FOR VERBOSE VERSION
    log_level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log file specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"Logging to file: {log_file}")
        except Exception as e:
            logging.warning(f"Could not set up file logging: {e}")

    # Set module logger level
    logger.setLevel(log_level)


def safe_pickle_dump(data: Any, file_path: str) -> None:
    """
    Safely dump data to a pickle file, handling type issues.

    Args:
        data: Data to pickle
        file_path: Path to save the pickle file
    """
    with open(file_path, 'wb') as f:
        # Cast to fix type hint issues
        binary_file = cast(BinaryIO, f)
        pickle.dump(data, binary_file, protocol=4)


def safe_pickle_load(file_path: str) -> Any:
    """
    Safely load data from a pickle file, handling type issues.

    Args:
        file_path: Path to the pickle file to load

    Returns:
        The unpickled data
    """
    with open(file_path, 'rb') as f:
        # Cast to fix type hint issues
        binary_file = cast(BinaryIO, f)
        return pickle.load(binary_file)


def validate_and_get_columns(
        df: pd.DataFrame,
        id_column: str,
        desc_column: str,
        freq_column: Optional[str],
        type_column: Optional[str],
        risk_column: Optional[str],
        audit_leader_column: Optional[str],
        audit_entity_column: Optional[str]
) -> Dict[str, Optional[str]]:
    """
    Validate required columns and check optional columns in the DataFrame.

    Args:
        df: DataFrame to validate
        id_column: Column name for control IDs
        desc_column: Column name for control descriptions
        freq_column: Optional column name for frequency values
        type_column: Optional column name for control type values
        risk_column: Optional column name for risk descriptions
        audit_leader_column: Optional column name for audit leader information
        audit_entity_column: Optional column name for audit entity information

    Returns:
        Dictionary of validated column names

    Raises:
        ValueError: If required columns are missing
    """
    # Print actual columns for debugging
    logger.debug(f"Excel file columns: {df.columns.tolist()}")

    # Check required columns
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in file")

    if desc_column not in df.columns:
        raise ValueError(f"Description column '{desc_column}' not found in file")

    # Check optional columns
    validated_columns = {
        "id_column": id_column,
        "desc_column": desc_column,
        "freq_column": freq_column,
        "type_column": type_column,
        "risk_column": risk_column,
        "audit_leader_column": audit_leader_column,
        "audit_entity_column": audit_entity_column
    }

    # Validate frequency column
    if freq_column and freq_column not in df.columns:
        logger.warning(
            f"Frequency column '{freq_column}' not found in file. "
            "Frequency validation will be skipped."
        )
        validated_columns["freq_column"] = None

    # Validate control type column
    if type_column and type_column not in df.columns:
        logger.warning(
            f"Control type column '{type_column}' not found in file. "
            "Control type validation will be skipped."
        )
        validated_columns["type_column"] = None

    # Validate risk column
    if risk_column and risk_column not in df.columns:
        logger.warning(
            f"Risk description column '{risk_column}' not found in file. "
            "Risk alignment will be skipped."
        )
        validated_columns["risk_column"] = None

    # Check and try to auto-detect audit leader column
    if audit_leader_column and audit_leader_column not in df.columns:
        logger.warning(f"Audit Leader column '{audit_leader_column}' not found in file.")

        # Try to automatically detect Audit Leader column
        potential_columns = ["Audit Leader", "audit leader", "Audit_Leader", "audit_leader",
                             "AuditLeader", "auditLeader", "Auditor", "Lead Auditor"]

        for col in potential_columns:
            if col in df.columns:
                validated_columns["audit_leader_column"] = col
                logger.info(f"Automatically detected Audit Leader column: '{col}'")
                break
        else:
            validated_columns["audit_leader_column"] = None

            # Check and try to auto-detect audit entity column
        if audit_entity_column and audit_entity_column not in df.columns:
            logger.warning(f"Audit Entity column '{audit_entity_column}' not found in file.")

            # Try to automatically detect Audit Entity column
            potential_columns = ["Audit Entity", "audit entity", "Audit_Entity", "audit_entity",
                                 "AuditEntity", "auditEntity", "Entity", "AE",]

            for col in potential_columns:
                if col in df.columns:
                    validated_columns["audit_entity_column"] = col
                    logger.info(f"Automatically detected Audit Entity column: '{col}'")
                    break
            else:
                validated_columns["audit_entity_column"] = None

    return validated_columns


def analyze_file_with_batches(
        analyzer: EnhancedControlAnalyzer,
        file_path: str,
        id_column: str,
        desc_column: str,
        freq_column: Optional[str] = None,
        type_column: Optional[str] = None,
        risk_column: Optional[str] = None,
        audit_leader_column: Optional[str] = None,
        audit_entity_column: Optional[str] = None,
        output_file: Optional[str] = None,
        batch_size: int = 500,
        temp_dir: Optional[str] = None,
        skip_visualizations: bool = False
) -> List[Dict[str, Any]]:
    """
    Analyze controls from an Excel file in batches and generate a detailed report.

    Args:
        analyzer: The EnhancedControlAnalyzer instance
        file_path: Path to Excel file containing controls
        id_column: Column containing control IDs
        desc_column: Column containing control descriptions
        freq_column: Optional column containing frequency values
        type_column: Optional column containing control type values
        risk_column: Optional column containing risk descriptions
        audit_leader_column: Optional column containing audit leader information
        audit_entity_column: Optional column containing audit entity information
        output_file: Optional path for output Excel report
        batch_size: Number of controls to process in each batch
        temp_dir: Directory to store temporary batch results
        skip_visualizations: Whether to skip generating visualizations

    Returns:
        List of control analysis results
    """
    logger.info(f"Reading file: {file_path}")
    logger.info(f"Using columns: ID={id_column}, Description={desc_column}, "
                f"Frequency={freq_column}, Type={type_column}, Risk={risk_column}, "
                f"Audit Leader={audit_leader_column}, Audit Entity={audit_entity_column}")  # Update log
    logger.info(f"Using batch size: {batch_size}")

    # Create temp directory if specified
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = "temp_batch_results"
        os.makedirs(temp_dir, exist_ok=True)

    try:
        # Read the Excel file
        df = pd.read_excel(file_path, engine='openpyxl')

        # Validate columns
        columns = validate_and_get_columns(
            df, id_column, desc_column, freq_column,
            type_column, risk_column, audit_leader_column,
            audit_entity_column
        )

        # Update with validated column names
        id_column = columns["id_column"]
        desc_column = columns["desc_column"]
        freq_column = columns["freq_column"]
        type_column = columns["type_column"]
        risk_column = columns["risk_column"]
        audit_leader_column = columns["audit_leader_column"]
        audit_entity_column = columns["audit_entity_column"]

        # Initialize batch processing
        all_results = []
        total_controls = len(df)
        start_time = time.time()

        logger.info(f"Analyzing {total_controls} controls in batches of {batch_size}...")

        # Process in batches
        for batch_start in range(0, total_controls, batch_size):
            batch_end = min(batch_start + batch_size, total_controls)
            batch_num = batch_start // batch_size + 1
            total_batches = (total_controls + batch_size - 1) // batch_size

            batch_start_time = time.time()
            logger.info(f"\nProcessing batch {batch_num}/{total_batches}: "
                        f"controls {batch_start + 1}-{batch_end}")

            # Get the slice of DataFrame for this batch
            batch_df = df.iloc[batch_start:batch_end].copy()
            batch_results = []

            # Process each control in the batch
            for i, (idx, row) in enumerate(batch_df.iterrows()):
                if i % 25 == 0:
                    # Show more granular progress within batch
                    progress = (i + 1) / len(batch_df) * 100
                    logger.info(f"  Batch progress: {progress:.1f}% ({i + 1}/{len(batch_df)})")

                control_id = row[id_column]
                description = row[desc_column]

                # Get optional fields if available
                frequency = row.get(freq_column) if freq_column and pd.notna(row.get(freq_column)) else None
                control_type = row.get(type_column) if type_column and pd.notna(row.get(type_column)) else None
                risk_description = row.get(risk_column) if risk_column and pd.notna(row.get(risk_column)) else None
                audit_leader = row.get(audit_leader_column) if audit_leader_column and pd.notna(
                    row.get(audit_leader_column)) else None
                audit_entity = row.get(audit_entity_column) if audit_entity_column and pd.notna(
                    row.get(audit_entity_column)) else None

                try:
                    # Analyze this control
                    result = analyzer.analyze_control(
                        control_id, description, frequency, control_type, risk_description
                    )

                    # Add Audit Leader if available
                    if audit_leader:
                        result["Audit Leader"] = audit_leader
                    # Add Audit Entity if available
                    if audit_entity:
                        result["Audit Entity"] = audit_entity

                    batch_results.append(result)
                except Exception as e:
                    # Log error but continue processing
                    error_msg = f"Error analyzing control {control_id}: {str(e)}"
                    logger.error(error_msg)
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
                logger.info(f"  Batch results saved to {batch_filename}")
            except Exception as e:
                logger.warning(f"  Could not save batch file: {e}")

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
            logger.info(f"  Batch {batch_num}/{total_batches} completed in {batch_time:.1f} seconds")
            logger.info(f"  Processing speed: {controls_per_second:.2f} controls/second")
            logger.info(f"  Progress: {completed}/{total_controls} controls ({completed / total_controls * 100:.1f}%)")

            hours, remainder = divmod(estimated_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            logger.info(f"  Estimated time remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s")

            # Save checkpoint of all results so far
            checkpoint_file = os.path.join(temp_dir, "all_results_checkpoint.pkl")
            try:
                safe_pickle_dump(all_results, checkpoint_file)
                logger.info(f"  Checkpoint saved to {checkpoint_file}")
            except Exception as e:
                logger.warning(f"  Could not save checkpoint: {e}")

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
                logger.info(f"\nAnalysis complete. Results saved to {output_file}")
            except Exception as e:
                logger.error(f"Error generating final report: {e}")
                # Try to save results in pickle format
                emergency_output = os.path.splitext(output_file)[0] + "_emergency.pkl"
                safe_pickle_dump(all_results, emergency_output)
                logger.info(f"Emergency results saved to {emergency_output}")

        # Generate visualizations if requested
        if not skip_visualizations and output_file:
            try:
                logger.info("Generating visualizations...")
                vis_dir = os.path.splitext(output_file)[0] + "_visualizations"
                generate_core_visualizations(all_results, vis_dir)
                logger.info(f"Visualizations saved to {vis_dir}")
            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")

        # Calculate total time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"\nTotal processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logger.info(f"Average processing speed: {total_controls / total_time:.2f} controls/second")

        return all_results

    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}")
        traceback.print_exc()
        raise


def resume_from_checkpoint(
        analyzer: EnhancedControlAnalyzer,
        checkpoint_file: str,
        args: argparse.Namespace
) -> Tuple[Optional[List[Dict[str, Any]]], int]:
    """
    Resume analysis from a saved checkpoint.

    Args:
        analyzer: The EnhancedControlAnalyzer instance
        checkpoint_file: Path to the checkpoint file
        args: Command line arguments

    Returns:
        Tuple of (results, exit_code)
    """
    logger.info(f"Resuming from checkpoint: {checkpoint_file}")

    try:
        # Initialize variables to avoid "might be referenced before assignment" errors
        completed_results = []
        last_control_id = None

        # Load checkpoint data
        try:
            completed_results = safe_pickle_load(checkpoint_file)
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            # Try with absolute path
            abs_path = os.path.abspath(checkpoint_file)
            logger.info(f"Trying with absolute path: {abs_path}")
            completed_results = safe_pickle_load(abs_path)

        if not completed_results:
            logger.error("Error: Empty or invalid checkpoint file")
            return None, 1

        logger.info(f"Loaded {len(completed_results)} completed controls from checkpoint")

        # Get the last control ID
        if completed_results:
            last_control_id = completed_results[-1].get("control_id")

        if not last_control_id:
            logger.warning("Could not determine last processed control ID")
        else:
            logger.info(f"Last processed control ID: {last_control_id}")

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
            logger.warning("Could not find last processed control. Starting from beginning.")
            last_index = -1

        # Create DataFrame with remaining controls
        remaining_df = df.iloc[last_index + 1:].copy()

        if len(remaining_df) == 0:
            logger.info("No remaining controls to process. Analysis was already complete.")

            # Generate the final report
            if args.output_file:
                try:
                    logger.info("Generating final report...")
                    generate_final_report(analyzer, completed_results, args)
                    logger.info(f"Final report generated at {args.output_file}")
                except Exception as e:
                    logger.error(f"Error generating report: {e}")
                    # Save as pickle file
                    pkl_path = os.path.splitext(args.output_file)[0] + "_results.pkl"
                    safe_pickle_dump(completed_results, pkl_path)
                    logger.info(f"Results saved to {pkl_path}")

            return completed_results, 0

        logger.info(f"Resuming analysis with {len(remaining_df)} remaining controls")

        # Create a temporary file with the remaining controls
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name

        remaining_df.to_excel(temp_path, engine='openpyxl', index=False)

        # Process the remaining controls
        temp_output = os.path.splitext(args.output_file)[0] + "_remaining.xlsx"

        # Process the remaining controls based on the user's choice
        remaining_results = []
        if args.use_batches:
            remaining_results = analyze_file_with_batches(
                analyzer=analyzer,
                file_path=temp_path,
                id_column=args.id_column,
                desc_column=args.desc_column,
                freq_column=args.freq_column,
                type_column=args.type_column,
                risk_column=args.risk_column,
                audit_leader_column=args.audit_leader_column,
                audit_entity_column=args.audit_entity_column,
                output_file=temp_output,
                batch_size=args.batch_size,
                temp_dir=args.temp_dir,
                skip_visualizations=args.skip_visualizations
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
                logger.info(f"Final report generated at {args.output_file}")
            except Exception as e:
                logger.error(f"Error generating final report: {e}")
                # Save results as pickle in case of error
                pickle_path = os.path.splitext(args.output_file)[0] + "_all_results.pkl"
                safe_pickle_dump(all_results, pickle_path)
                logger.info(f"Results saved to {pickle_path}")

        # Generate visualizations if requested
        if not args.skip_visualizations and args.output_file:
            try:
                logger.info("Generating visualizations...")
                vis_dir = os.path.splitext(args.output_file)[0] + "_visualizations"
                generate_core_visualizations(all_results, vis_dir)
                logger.info(f"Visualizations saved to {vis_dir}")
            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")

        # Delete the temporary file
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Warning: Failed to delete temporary file: {e}")

        return all_results, 0

    except Exception as e:
        logger.error(f"Error resuming from checkpoint: {e}")
        traceback.print_exc()
        return None, 1


def generate_final_report(
        analyzer: EnhancedControlAnalyzer,
        results: List[Dict[str, Any]],
        args: argparse.Namespace
) -> None:
    """
    Generate final report without directly calling protected methods.

    Args:
        analyzer: The EnhancedControlAnalyzer instance
        results: List of analysis results
        args: Command line arguments
    """
    try:
        analyzer._generate_enhanced_report(
            results,
            args.output_file,
            args.freq_column is not None,
            args.type_column is not None,
            args.risk_column is not None
        )
    except Exception as e:
        logger.error(f"Warning: Error in report generation: {e}")
        # Try using a manual approach to generate Excel
        try:
            write_excel_report(results, args.output_file)
        except Exception as excel_error:
            logger.error(f"Failed to write Excel report: {excel_error}")
            raise


def write_excel_report(results: List[Dict[str, Any]], output_file: str) -> None:
    """
    Fallback method to write results to Excel in case of report generation error.

    Args:
        results: List of analysis results
        output_file: Path to save the Excel report
    """
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    # Convert results to DataFrame
    basic_results = []
    for r in results:
        result_dict = {
            "Control ID": r.get("control_id", "Unknown"),
            "Description": r.get("description", ""),
            "Control Quality Score (Official)": r.get("total_score", 0),
            "Quality Category": r.get("category", "Unknown"),
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


def process_with_batch_option(
        analyzer: EnhancedControlAnalyzer,
        args: argparse.Namespace
) -> int:
    """
    Process controls using batch processing if requested.

    Args:
        analyzer: The EnhancedControlAnalyzer instance
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        if args.use_batches:
            logger.info(f"Using batch processing with batch size of {args.batch_size}")
            logger.info(f"Using columns: ID={args.id_column}, Description={args.desc_column}, "
                        f"Frequency={args.freq_column}, Type={args.type_column}, "
                        f"Risk={args.risk_column}, Audit Leader={args.audit_leader_column}")

            analyze_file_with_batches(
                analyzer=analyzer,
                file_path=args.file,
                id_column=args.id_column,
                desc_column=args.desc_column,
                freq_column=args.freq_column,
                type_column=args.type_column,
                risk_column=args.risk_column,
                audit_leader_column=args.audit_leader_column,
                audit_entity_column=args.audit_entity_column,
                output_file=args.output_file,
                batch_size=args.batch_size,
                temp_dir=args.temp_dir,
                skip_visualizations=args.skip_visualizations
            )

        else:
            # Original non-batch processing
            results = analyzer.analyze_file(
                file_path=args.file,
                id_column=args.id_column,
                desc_column=args.desc_column,
                freq_column=args.freq_column,
                type_column=args.type_column,
                risk_column=args.risk_column,
                audit_leader_column=args.audit_leader_column,
                audit_entity_column=args.audit_entity_column,
                output_file=args.output_file
            )

            # Generate visualizations unless skipped
            if not args.skip_visualizations and results and args.output_file:
                logger.info("Generating visualizations...")
                vis_dir = os.path.splitext(args.output_file)[0] + "_visualizations"

                # Generate visualizations
                generate_core_visualizations(results, vis_dir)
                logger.info(f"Visualizations saved to {vis_dir}")

                # Open dashboard if requested
                if args.open_dashboard:
                    dashboard_path = os.path.join(vis_dir, "dashboard.html")
                    if os.path.exists(dashboard_path):
                        logger.info(f"Opening dashboard: {dashboard_path}")
                        import webbrowser
                        webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
                    else:
                        logger.warning("Dashboard not found. Skipping open request.")

        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        return 1


def update_argument_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add batch processing arguments to the argument parser.

    Args:
        parser: ArgumentParser to update

    Returns:
        Updated ArgumentParser
    """
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
    parser.add_argument('--audit-leader-column', help='Column containing Audit Leader information (overrides config)'),
    parser.add_argument('--audit-entity-column', help='Column containing Audit Entity information (overrides config)')  # Add this
    parser.add_argument('--debug', action='store_true', default=True,  # DEFAULT TO TRUE FOR VERBOSE VERSION
                        help='Print additional debug information (default: enabled)')
    parser.add_argument('--open-dashboard', action='store_true',
                        help='Automatically open the dashboard in default browser')
    parser.add_argument('--log-file', help='Path to log file')
    parser.add_argument('--verbose', action='store_true', default=True,  # DEFAULT TO TRUE FOR VERBOSE VERSION
                        help='Enable verbose logging (default: enabled)')

    return parser


def main() -> int:
    """
    Main function to run control analysis with batch processing support.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    version_info = f"Python {platform.python_version()} on {platform.system()} {platform.release()}"
    logger.info(f"Control Analyzer Integration Script (VERBOSE VERSION) {version_info}")

    parser = argparse.ArgumentParser(
        description='Enhanced Control Description Analyzer with batch processing support (VERBOSE VERSION)'
    )

    # Input file argument
    parser.add_argument('file', nargs='?', help='Excel file with control descriptions')

    # Column mapping arguments
    parser.add_argument('--id-column', default=None, help='Column containing control IDs')
    parser.add_argument('--desc-column', default=None, help='Column containing control descriptions')
    parser.add_argument('--freq-column', help='Column containing frequency values for validation')
    parser.add_argument('--type-column', help='Column containing control type values for validation')
    parser.add_argument('--risk-column', help='Column containing risk descriptions for alignment')

    # Output arguments
    parser.add_argument('--output-file', help='Output Excel file path')
    parser.add_argument('--config', help='Path to configuration file (YAML)')

    # Processing mode arguments
    parser.add_argument('--disable-enhanced', action='store_true',
                        help='Disable enhanced detection modules')

    # Add batch processing arguments
    parser = update_argument_parser(parser)

    args = parser.parse_args()

    # Set up logging - ALWAYS VERBOSE FOR THIS VERSION
    setup_logging(args.log_file, True)  # Force verbose=True

    # ALWAYS enable debug output for verbose version
    logger.debug(f"Arguments: {args}")

    # Set default output filename if not provided
    if args.file and not args.output_file:
        base_name = os.path.splitext(args.file)[0]
        args.output_file = f"{base_name}_analysis_results.xlsx"
        logger.debug(f"Setting default output file to {args.output_file}")

    if not args.file and not args.resume_from:
        parser.print_help()
        return 1

    try:
        # Create analyzer instance
        logger.info(f"Creating analyzer with config file: {args.config}")

        analyzer = EnhancedControlAnalyzer(args.config)

        logger.debug("Analyzer created successfully")
        if hasattr(analyzer, 'column_mappings'):
            logger.debug("Column mappings from config:")
            for key, value in analyzer.column_mappings.items():
                if not isinstance(value, dict):  # Skip complex nested objects
                    logger.debug(f"  {key}: {value}")

        # Apply command-line flags
        if args.disable_enhanced:
            analyzer.use_enhanced_detection = False
            logger.info("Enhanced detection modules disabled. Using base analysis only.")

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
        logger.info(f"Analyzing controls from {args.file}...")

        # Check if Excel file exists
        if not os.path.exists(args.file):
            logger.error(f"Error: Excel file '{args.file}' not found")
            return 1

        # Read Excel file to show columns - ALWAYS for verbose version
        try:
            df = pd.read_excel(args.file)
            logger.debug(f"Excel file columns: {df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")

        results = analyzer.analyze_file(
            file_path=args.file,
            id_column=args.id_column,
            desc_column=args.desc_column,
            freq_column=args.freq_column,
            type_column=args.type_column,
            risk_column=args.risk_column,
            audit_leader_column=args.audit_leader_column,
            audit_entity_column=args.audit_entity_column,
            output_file=args.output_file
        )

        # Generate visualizations if not skipped
        if not args.skip_visualizations and args.output_file:
            logger.info("Generating visualizations...")
            vis_dir = os.path.splitext(args.output_file)[0] + "_visualizations"

            # Generate visualizations
            generate_core_visualizations(results, vis_dir)
            logger.info(f"Visualizations saved to {vis_dir}")

            # Open dashboard if requested
            if args.open_dashboard:
                dashboard_path = os.path.join(vis_dir, "dashboard.html")
                if os.path.exists(dashboard_path):
                    logger.info(f"Opening dashboard: {dashboard_path}")
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
                else:
                    logger.warning("Dashboard not found. Skipping open request.")

        logger.info(f"Analysis complete. Results saved to {args.output_file}")
        return 0

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())