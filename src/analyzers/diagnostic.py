#!/usr/bin/env python3
"""
Enhanced diagnostic script for troubleshooting Control Analyzer issues
"""

import sys
import traceback
import importlib.util
import inspect
import os


def check_file_exists(filename):
    """Check if a file exists and print its details"""
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"✓ Found {filename} ({size} bytes)")
        return True
    else:
        print(f"✗ File not found: {filename}")
        return False


def inspect_class(cls):
    """Print details about a class structure"""
    print(f"\nClass: {cls.__name__}")
    print("Methods:")

    # Get all methods in class
    methods = [m for m, _ in inspect.getmembers(cls, inspect.isfunction)]
    for method in methods:
        print(f"  - {method}")

    # Show __init__ parameters
    try:
        init_sig = inspect.signature(cls.__init__)
        print(f"\nInit signature: {init_sig}")
    except Exception as e:
        print(f"Error getting init signature: {e}")


def main():
    print("Control Analyzer Deep Diagnostic")
    print("===============================")

    # Check key files
    print("\nChecking files...")
    check_file_exists("control_analyzer.py")
    check_file_exists("src/utils/config_adapter.py")
    check_file_exists("enhanced_who.py")
    check_file_exists("enhanced_what.py")
    check_file_exists("enhanced_when.py")
    check_file_exists("enhanced_why.py")
    check_file_exists("enhanced_escalation.py")

    # Try to import modules
    print("\nTrying imports...")
    try:
        # Use importlib to avoid propagating import errors
        spec = importlib.util.spec_from_file_location("config_adapter", "src/utils/config_adapter.py")
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        print("✓ Imported config_adapter")

        # Inspect ConfigAdapter
        config_manager_class = config_module.ConfigAdapter
        inspect_class(config_manager_class)

        # Try to create an instance
        print("\nTrying to create ConfigAdapter instance...")
        cm = config_manager_class()
        print("✓ Created ConfigAdapter instance")
    except Exception as e:
        print(f"✗ Error with config_adapter: {e}")
        traceback.print_exc()

    try:
        # Import control_analyzer
        spec = importlib.util.spec_from_file_location("control_analyzer", "control_analyzer.py")
        analyzer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analyzer_module)
        print("✓ Imported control_analyzer")

        # Inspect classes
        try:
            control_element_class = analyzer_module.ControlElement
            inspect_class(control_element_class)
        except Exception as e:
            print(f"✗ Error with ControlElement class: {e}")

        try:
            analyzer_class = analyzer_module.EnhancedControlAnalyzer
            inspect_class(analyzer_class)

            # Check if _get_column_name method exists
            if hasattr(analyzer_class, '_get_column_name') and callable(getattr(analyzer_class, '_get_column_name')):
                print("\n✓ _get_column_name method exists in EnhancedControlAnalyzer")
            else:
                print("\n✗ _get_column_name method MISSING from EnhancedControlAnalyzer")
        except Exception as e:
            print(f"✗ Error with EnhancedControlAnalyzer class: {e}")

        # Try to create analyzer instance
        print("\nTrying to create EnhancedControlAnalyzer instance...")
        try:
            analyzer = analyzer_module.EnhancedControlAnalyzer()
            print("✓ Created EnhancedControlAnalyzer instance")

            # Check key attributes
            print("\nChecking key attributes:")
            attributes = ['elements', 'config', 'nlp', 'vague_terms', 'vague_matcher',
                          'use_enhanced_detection', 'excellent_threshold', 'good_threshold']

            for attr in attributes:
                if hasattr(analyzer, attr):
                    print(f"✓ Has attribute: {attr}")
                else:
                    print(f"✗ Missing attribute: {attr}")

            # Check column mapping attributes
            if hasattr(analyzer, 'column_mappings'):
                print(f"✓ Has attribute: column_mappings = {analyzer.column_mappings}")
            else:
                print(f"✗ Missing attribute: column_mappings")

            if hasattr(analyzer, 'default_column_mappings'):
                print(f"✓ Has attribute: default_column_mappings = {analyzer.default_column_mappings}")
            else:
                print(f"✗ Missing attribute: default_column_mappings")

        except Exception as e:
            print(f"✗ Error creating EnhancedControlAnalyzer: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"✗ Error with control_analyzer: {e}")
        traceback.print_exc()

    print("\nDiagnostic complete")


if __name__ == "__main__":
    main()