#!/usr/bin/env python3
"""excel_to_systems_yaml.py - Extract system names from Column B of report"""

#To run: python -m src.utils.excel_to_yaml list_of_systems.xlsx

import pandas as pd
import yaml
import re
from datetime import datetime
import sys
import os
import shutil


def sanitize_system_name(name):
    """Clean system names for YAML compatibility"""
    if pd.isna(name):
        return None

    name = str(name).strip()

    if not name:
        return None

    # Replace problematic YAML characters
    replacements = {
        ':': '-',  # Colons break YAML
        '|': '-',  # Pipe is YAML multiline
        '>': '-',  # YAML folding indicator
        '@': 'at',
        '*': 'star',  # YAML alias
        '&': 'and',  # YAML anchor
        '!': '',  # YAML tag
        '?': '',
        '#': 'num',  # Comments
        '%': 'pct',
        '{': '(',
        '}': ')',
        '[': '(',
        ']': ')',
        '\\': '/',
        '\n': ' ',
        '\r': ' ',
        '\t': ' ',
        '~': '-',  # YAML null
        '`': '',
        '^': '',
        '=': '-',
        '+': 'plus',
        '<': '-',
        ',': '',
        ';': '-',
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    # Handle quotes by removing them
    name = name.replace('"', '')
    name = name.replace("'", '')

    # Add prefix if starts with problematic character
    if name and name[0] in '-?:!@`%&*0123456789':
        name = 'sys_' + name

    # Clean up non-alphanumeric except spaces, hyphens, underscores
    name = re.sub(r'[^a-zA-Z0-9\s\-_]', '', name)

    # Replace multiple spaces/hyphens with single hyphen
    name = re.sub(r'[\s\-]+', '-', name)
    name = name.strip('-')

    # Convert to lowercase
    name = name.lower()

    # Handle YAML reserved words
    if name in {'yes', 'no', 'true', 'false', 'on', 'off', 'null'}:
        name = 'sys_' + name

    return name if name else None


def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_to_systems_yaml.py <excel_file>")
        sys.exit(1)

    excel_file = sys.argv[1]
    
    # Determine output path - always write to config folder
    # Handle running from different directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    config_dir = os.path.join(project_root, 'config')
    output_file = os.path.join(config_dir, 'systems_catalog.yaml')
    
    print(f"Reading {excel_file}...")
    print(f"Output will be written to: {output_file}")

    try:
        # Read Excel file - Column B (index 1) has the application names
        df = pd.read_excel(excel_file, usecols=[1])  # Only read column B

        # Get the column name (whatever it's called in the report)
        column_name = df.columns[0]
        print(f"Found column: '{column_name}' with {len(df)} entries")

        # Process and clean system names
        all_names = df[column_name].apply(sanitize_system_name)
        clean_names = all_names.dropna().unique().tolist()
        clean_names.sort()

        print(f"After cleaning: {len(clean_names)} unique systems")

        # Create simple YAML structure
        yaml_content = {
            '_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '_source': excel_file,
            '_count': len(clean_names),
            'systems': clean_names
        }
        
        # Create backup if file exists
        if os.path.exists(output_file):
            backup_dir = os.path.join(config_dir, 'backups')
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(backup_dir, f'systems_catalog.yaml.backup.{timestamp}')
            
            shutil.copy2(output_file, backup_file)
            print(f"\n✓ Backed up existing file to: {backup_file}")

        # Write YAML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Auto-generated systems list from Excel report\n")
            f.write(f"# Source: {excel_file}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            yaml.dump(yaml_content, f, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)

        print(f"\n✓ Generated {output_file}")
        print(f"  Total systems: {len(clean_names)}")

        # Show a few examples
        print("\nFirst 10 systems:")
        for name in clean_names[:10]:
            print(f"  - {name}")

        if len(clean_names) > 10:
            print(f"  ... and {len(clean_names) - 10} more")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()