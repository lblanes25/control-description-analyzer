#!/usr/bin/env python3
"""
Packaging script for Control Description Analyzer GUI
This script uses PyInstaller to create a standalone executable for the application
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Package Control Analyzer GUI as executable')
    parser.add_argument('--onefile', action='store_true', help='Create a single executable file')
    parser.add_argument('--debug', action='store_true', help='Include debug information')
    parser.add_argument('--icon', help='Path to icon file (.ico)')
    parser.add_argument('--name', default='ControlAnalyzerGUI', help='Name of the executable')
    parser.add_argument('--config', default='control_analyzer_config.yaml', help='Path to config file')

    return parser.parse_args()


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['pyinstaller', 'PyQt5', 'pandas', 'spacy', 'plotly', 'pyyaml']

    print("Checking dependencies...")
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ✗ {package}")

    if missing_packages:
        print("\nThe following packages are missing and need to be installed:")
        for package in missing_packages:
            print(f"  - {package}")

        print("\nInstall them with:")
        packages_str = ' '.join(missing_packages)
        print(f"pip install {packages_str}")

        return False

    return True


def check_spacy_model():
    """Check if required spaCy model is installed"""
    try:
        import spacy
        # Try to load the English model
        try:
            spacy.load('en_core_web_md')
            print("  ✓ spaCy model 'en_core_web_md'")
            return True
        except OSError:
            try:
                spacy.load('en_core_web_sm')
                print("  ✓ spaCy model 'en_core_web_sm' (minimal)")
                print("  ⚠️ For better results, consider installing 'en_core_web_md'")
                return True
            except OSError:
                print("  ✗ No spaCy model found")
                print("\nInstall a spaCy model with:")
                print("python -m spacy download en_core_web_md")
                return False
    except ImportError:
        # spaCy import already failed in check_dependencies
        return False


def create_build_directory(name):
    """Create and prepare build directory"""
    # Create build directory
    build_dir = Path('build') / name
    if build_dir.exists():
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    return build_dir


def copy_resources(build_dir, config_file):
    """Copy necessary resources to build directory"""
    # Create resources directory
    resources_dir = build_dir / 'resources'
    resources_dir.mkdir(exist_ok=True)

    # Copy config file if it exists
    if os.path.exists(config_file):
        shutil.copy(config_file, resources_dir / 'control_analyzer_config.yaml')
        print(f"Copied config file: {config_file}")
    else:
        print(f"Warning: Config file not found: {config_file}")

    # Copy any other necessary resources
    # For example, icon files, sample data, etc.

    return resources_dir


def create_pyinstaller_command(args, build_dir, resources_dir=None):
    """Create PyInstaller command based on arguments"""
    cmd = ['pyinstaller']

    # Basic options
    cmd.append('--clean')
    cmd.append('--noconfirm')

    # Debug options
    if args.debug:
        cmd.append('--debug=all')

    # One file or directory
    if args.onefile:
        cmd.append('--onefile')
    else:
        cmd.append('--onedir')

    # Name
    cmd.extend(['--name', args.name])

    # Icon
    if args.icon and os.path.exists(args.icon):
        cmd.extend(['--icon', args.icon])

    # Output directory
    cmd.extend(['--distpath', str(build_dir / 'dist')])
    cmd.extend(['--workpath', str(build_dir / 'work')])

    # Add data files
    if resources_dir:
        cmd.extend(['--add-data', f"{resources_dir}:resources"])

    # Entry point
    cmd.append('control_analyzer_gui.py')

    return cmd


def run_pyinstaller(cmd):
    """Run PyInstaller command"""
    print("\nRunning PyInstaller...")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running PyInstaller: {e}")
        return False


def main():
    """Main function"""
    print("Control Analyzer GUI Packaging Tool")
    print("===================================")

    # Parse arguments
    args = parse_arguments()

    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        return 1

    # Check spaCy model
    if not check_spacy_model():
        print("\nPlease install a spaCy model and try again.")
        return 1

    # Create build directory
    build_dir = create_build_directory(args.name)

    # Copy resources
    resources_dir = copy_resources(build_dir, args.config)

    # Create PyInstaller command
    cmd = create_pyinstaller_command(args, build_dir, resources_dir)

    # Run PyInstaller
    if not run_pyinstaller(cmd):
        return 1

    # Success
    print("\nPackaging completed successfully!")
    if args.onefile:
        exe_path = build_dir / 'dist' / f"{args.name}.exe"
    else:
        exe_path = build_dir / 'dist' / args.name / f"{args.name}.exe"

    print(f"Executable created at: {exe_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())