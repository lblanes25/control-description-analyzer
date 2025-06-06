#!/usr/bin/env python3
"""
Control Analyzer to Tableau Hyper Integration

This script creates Tableau Hyper extracts from Control Description Analyzer results
for direct integration with Tableau Desktop/Server.

Dependencies:
- tableauhyperapi (Tableau Hyper API)
- pandas
- tableauserverclient (for publishing to Tableau Server)
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import Tableau Hyper API components
from tableauhyperapi import (
    HyperProcess, Connection, TableDefinition, TableName, SqlType,
    Telemetry, Inserter, CreateMode, NOT_NULLABLE, NULLABLE
)

# Import Tableau Server Client for publishing (optional)
try:
    import tableauserverclient as TSC
except ImportError:
    print("Warning: tableauserverclient not found. Publishing to Tableau Server will not be available.")
    TSC = None


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Convert Control Description Analyzer results to Tableau Hyper extracts'
    )
    parser.add_argument('--input-file', required=True, help='Path to Excel results file from Control Analyzer')
    parser.add_argument('--output-dir', default='.', help='Output directory for Hyper files')
    parser.add_argument('--publish', action='store_true', help='Publish to Tableau Server')
    parser.add_argument('--server', help='Tableau Server URL (if publishing)')
    parser.add_argument('--site', default='', help='Tableau Server site (if publishing)')
    parser.add_argument('--project', default='Default', help='Tableau Server project (if publishing)')
    parser.add_argument('--username', help='Tableau Server username (if publishing)')
    parser.add_argument('--password', help='Tableau Server password (if publishing)')
    parser.add_argument('--token-name', help='Tableau Server personal access token name (if publishing)')
    parser.add_argument('--token-value', help='Tableau Server personal access token value (if publishing)')
    parser.add_argument('--separate-sheets', action='store_true',
                        help='Create separate Hyper files for each sheet in the Excel file')

    return parser.parse_args()


def read_excel_sheets(excel_file):
    """
    Read Excel file with multiple sheets into a dictionary of DataFrames

    Args:
        excel_file: Path to Excel file

    Returns:
        Dictionary of DataFrames, one per sheet
    """
    try:
        # Read all sheets into a dictionary of DataFrames
        sheets = pd.read_excel(excel_file, sheet_name=None)
        print(f"Successfully read {len(sheets)} sheets from {excel_file}")
        return sheets
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)


def sql_type_from_pandas(dtype):
    """
    Map pandas data types to Tableau Hyper API SQL types

    Args:
        dtype: pandas data type

    Returns:
        Appropriate SqlType for the Hyper API
    """
    dtype_str = str(dtype)

    if 'int' in dtype_str:
        return SqlType.int()
    elif 'float' in dtype_str:
        return SqlType.double()
    elif 'bool' in dtype_str:
        return SqlType.bool()
    elif 'datetime' in dtype_str:
        return SqlType.timestamp()
    elif 'date' in dtype_str:
        return SqlType.date()
    else:
        # Default to text for object, string, and other types
        # With a generous max length to accommodate longer text fields
        return SqlType.text()


def create_table_definition(table_name, df):
    """
    Create Hyper table definition from pandas DataFrame

    Args:
        table_name: Name of the table to create
        df: pandas DataFrame

    Returns:
        TableDefinition object
    """
    columns = []

    # Create a column definition for each column in the DataFrame
    for col_name, dtype in df.dtypes.items():
        # Clean column name to avoid special characters
        clean_col_name = col_name.replace(" ", "_").replace("-", "_")

        # Get the SQL type for this column
        sql_type = sql_type_from_pandas(dtype)

        # Add column to the list
        if df[col_name].isna().any():
            # If there are any NaN values, make the column nullable
            columns.append(TableDefinition.Column(clean_col_name, sql_type, NULLABLE))
        else:
            # Otherwise, it's not nullable
            columns.append(TableDefinition.Column(clean_col_name, sql_type, NOT_NULLABLE))

    # Create and return the table definition
    return TableDefinition(
        table_name=TableName("Extract", table_name),
        columns=columns
    )


def insert_data(connection, table_def, df):
    """
    Insert DataFrame data into Hyper table

    Args:
        connection: Hyper connection
        table_def: Table definition
        df: pandas DataFrame with data to insert

    Returns:
        Number of rows inserted
    """
    row_count = 0

    # Clean column names to match the table definition
    df_columns = [col.replace(" ", "_").replace("-", "_") for col in df.columns]
    df.columns = df_columns

    # Insert the data
    with Inserter(connection, table_def) as inserter:
        for _, row in df.iterrows():
            # Convert row to list, handling NaN values
            row_data = []
            for val in row:
                if pd.isna(val):
                    row_data.append(None)
                else:
                    row_data.append(val)

            inserter.add_row(row_data)
            row_count += 1

        # Execute the insert
        inserter.execute()

    return row_count


def create_hyper_file(output_path, sheet_name, df):
    """
    Create a Hyper file from a DataFrame

    Args:
        output_path: Path where the Hyper file will be saved
        sheet_name: Name of the sheet/table
        df: pandas DataFrame with data

    Returns:
        Path to the created Hyper file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Remove file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Start the Hyper process
    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        print(f"Creating Hyper file: {output_path}")

        # Connect to the Hyper file
        with Connection(hyper.endpoint, output_path, CreateMode.CREATE_AND_REPLACE) as connection:
            # Create the schema
            connection.catalog.create_schema("Extract")

            # Create the table definition
            table_def = create_table_definition(sheet_name, df)

            # Create the table
            connection.catalog.create_table(table_def)

            # Insert the data
            rows_inserted = insert_data(connection, table_def, df)

            print(f"Inserted {rows_inserted} rows into {sheet_name} table")

    return output_path


def publish_to_tableau_server(hyper_file, args, datasource_name):
    """
    Publish Hyper file to Tableau Server

    Args:
        hyper_file: Path to Hyper file
        args: Command line arguments
        datasource_name: Name of the data source on the server

    Returns:
        URL of the published data source or None if publishing failed
    """
    if TSC is None:
        print("Error: tableauserverclient not installed. Cannot publish to Tableau Server.")
        return None

    try:
        # Set up authentication
        if args.token_name and args.token_value:
            # Use personal access token if provided
            tableau_auth = TSC.PersonalAccessTokenAuth(args.token_name, args.token_value, args.site)
        elif args.username and args.password:
            # Otherwise use username/password
            tableau_auth = TSC.TableauAuth(args.username, args.password, args.site)
        else:
            print("Error: Must provide either token or username/password for publishing")
            return None

        # Connect to the server
        server = TSC.Server(args.server)
        with server.auth.sign_in(tableau_auth):
            # Find the project
            projects = list(TSC.Pager(server.projects))
            project = next((p for p in projects if p.name.lower() == args.project.lower()), None)

            if project is None:
                print(f"Error: Project '{args.project}' not found")
                return None

            # Create the data source
            datasource = TSC.DatasourceItem(project.id)
            datasource.name = datasource_name

            # Publish the data source
            print(f"Publishing '{datasource_name}' to Tableau Server...")
            datasource = server.datasources.publish(
                datasource,
                hyper_file,
                TSC.Server.PublishMode.Overwrite
            )

            # Get the URL
            view_url = server.build_datasource_url(datasource)
            print(f"Published to: {view_url}")

            return view_url

    except Exception as e:
        print(f"Error publishing to Tableau Server: {e}")
        return None


def main():
    """Main function"""
    args = parse_arguments()

    # Read the Excel file
    print(f"Reading Excel file: {args.input_file}")
    sheets = read_excel_sheets(args.input_file)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process each sheet
    hyper_files = []

    if args.separate_sheets:
        # Create separate Hyper files for each sheet
        for sheet_name, df in sheets.items():
            output_filename = f"control_analyzer_{sheet_name.lower().replace(' ', '_')}_{timestamp}.hyper"
            output_path = os.path.join(args.output_dir, output_filename)

            hyper_file = create_hyper_file(output_path, sheet_name, df)
            hyper_files.append((hyper_file, sheet_name))
    else:
        # Create a single Hyper file with multiple tables
        output_filename = f"control_analyzer_all_{timestamp}.hyper"
        output_path = os.path.join(args.output_dir, output_filename)

        # Remove file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)

        # Start the Hyper process
        with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
            print(f"Creating combined Hyper file: {output_path}")

            # Connect to the Hyper file
            with Connection(hyper.endpoint, output_path, CreateMode.CREATE_AND_REPLACE) as connection:
                # Create the schema
                connection.catalog.create_schema("Extract")

                # Process each sheet
                for sheet_name, df in sheets.items():
                    # Create the table definition
                    table_def = create_table_definition(sheet_name, df)

                    # Create the table
                    connection.catalog.create_table(table_def)

                    # Insert the data
                    rows_inserted = insert_data(connection, table_def, df)

                    print(f"Inserted {rows_inserted} rows into {sheet_name} table")

        hyper_files.append((output_path, "All Results"))

    # Publish to Tableau Server if requested
    if args.publish:
        if not args.server:
            print("Error: Must provide server URL with --server when using --publish")
            return

        for hyper_file, sheet_name in hyper_files:
            datasource_name = f"Control Analyzer - {sheet_name}"
            publish_to_tableau_server(hyper_file, args, datasource_name)

    print("\nSummary:")
    for hyper_file, sheet_name in hyper_files:
        print(f"  {sheet_name}: {hyper_file}")
    print("\nProcess completed successfully!")


if __name__ == "__main__":
    main()