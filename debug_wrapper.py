#!/usr/bin/env python3
"""
Debug wrapper for Control Description Analyzer integration.py
This script will run the integration.py script with additional debugging output
"""

import os
import sys
import subprocess
import traceback


def run_with_debugging():
    """Run integration.py with additional debug output"""
    print("Debug Wrapper for Control Description Analyzer")
    print("=============================================")

    # Get command line arguments (skipping the script name)
    args = sys.argv[1:]
    if not args:
        print("Usage: python debug_wrapper.py [integration.py arguments]")
        print(
            "Example: python debug_wrapper.py who_test_controls.xlsx --config control_analyzer_config_final_with_columns.yaml")
        return 1

    # Print information about the environment
    print("\nEnvironment Information:")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")

    # Check if files exist
    print("\nChecking input files:")

    # Check Excel file
    excel_file = args[0] if args else None
    if excel_file:
        if os.path.exists(excel_file):
            print(f"✅ Excel file found: {excel_file}")
            # Try to read the Excel file to check it's valid
            try:
                import pandas as pd
                df = pd.read_excel(excel_file)
                print(f"   Excel file contains {len(df)} rows and {len(df.columns)} columns")
                print(f"   Columns: {', '.join(df.columns.tolist())}")
            except Exception as e:
                print(f"❌ Error reading Excel file: {e}")
        else:
            print(f"❌ Excel file not found: {excel_file}")

    # Check config file
    config_file = None
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            config_file = args[i + 1]
            break

    if config_file:
        if os.path.exists(config_file):
            print(f"✅ Config file found: {config_file}")
            # Try to read the YAML file
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"   Config contains {len(config.keys())} top-level keys")
                if 'columns' in config:
                    print(f"   Column mappings: {config['columns']}")
            except Exception as e:
                print(f"❌ Error reading config file: {e}")
        else:
            print(f"❌ Config file not found: {config_file}")

    # Check output file argument
    output_file = None
    for i, arg in enumerate(args):
        if arg == "--output-file" and i + 1 < len(args):
            output_file = args[i + 1]
            break

    if output_file:
        print(f"✅ Output will be written to: {output_file}")
    else:
        # Try to determine default output file
        if excel_file:
            default_output = f"{os.path.splitext(excel_file)[0]}_analysis_results.xlsx"
            print(f"ℹ️ No output file specified, will default to: {default_output}")

    # Run integration.py with the original arguments
    print("\nRunning integration.py with arguments:")
    cmd = ["python", "integration.py"] + args
    print(f"Command: {' '.join(cmd)}")
    print("\nOutput from integration.py:")
    print("-" * 50)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Print output in real-time
        for line in process.stdout:
            print(line, end='')

        # Get return code
        returncode = process.wait()

        # Print any errors
        stderr = process.stderr.read()
        if stderr:
            print("\nErrors:")
            print(stderr)

        print("-" * 50)
        print(f"Integration.py finished with exit code {returncode}")

        # Check if output file was created
        if output_file and os.path.exists(output_file):
            print(f"✅ Output file was created: {output_file} ({os.path.getsize(output_file)} bytes)")
        elif excel_file:
            default_output = f"{os.path.splitext(excel_file)[0]}_analysis_results.xlsx"
            if os.path.exists(default_output):
                print(f"✅ Default output file was created: {default_output} ({os.path.getsize(default_output)} bytes)")
            else:
                print("❌ No output file was created")

        return returncode
    except Exception as e:
        print(f"Error running integration.py: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_with_debugging())