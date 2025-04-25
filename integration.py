#!/usr/bin/env python3
# enhanced_integration.py - Integration script for Enhanced Control Description Analyzer

import argparse
import os
import sys
import yaml
from spacy.matcher import PhraseMatcher
from control_analyzer import EnhancedControlAnalyzer
from visualization import generate_core_visualizations


def load_config(config_file):
    """Load configuration from YAML file"""
    if not config_file:
        return {}

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}


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



def main():
    """Main function integrating enhanced analyzer and visualizations"""
    parser = argparse.ArgumentParser(description='Enhanced Control Description Analyzer with advanced NLP capabilities')
    parser.add_argument('file', nargs='?', help='Excel file with control descriptions')
    parser.add_argument('--id-column', help='Column name containing control IDs (e.g., Control_ID)')
    parser.add_argument('--desc-column', help='Column name containing control descriptions (e.g., Control_Description)')
    parser.add_argument('--freq-column', help='Column containing frequency values for validation')
    parser.add_argument('--type-column', help='Column containing control type values for validation')
    parser.add_argument('--risk-column', help='Column containing risk descriptions for alignment')
    parser.add_argument('--output-file', help='Output Excel file path')
    parser.add_argument('--config', help='Path to configuration file (YAML)')
    parser.add_argument('--disable-enhanced', action='store_true', help='Disable enhanced detection modules')
    parser.add_argument('--skip-visualizations', action='store_true', help='Skip generating visualizations')

    args = parser.parse_args()

    if not args.file:
        parser.print_help()
        return 1

    # Load configuration
    config = load_config(args.config) if args.config else {}

    # Create analyzer and apply configuration
    analyzer = EnhancedControlAnalyzer()

    if args.config:
        analyzer = apply_config_to_analyzer(analyzer, config)

    # Apply command-line flags
    if args.disable_enhanced:
        analyzer.use_enhanced_detection = False
        print("Enhanced detection modules disabled. Using base analysis only.")

    # Get column names from config if CLI args were not set
    column_config = config.get("columns", {})

    args.id_column = args.id_column or column_config.get("id", "Control_ID")
    args.desc_column = args.desc_column or column_config.get("description", "Control_Description")
    args.freq_column = args.freq_column or column_config.get("frequency", None)
    args.type_column = args.type_column or column_config.get("type", None)
    args.risk_column = args.risk_column or column_config.get("risk", None)

    # Set default output filename if not provided
    if not args.output_file:
        base_name = os.path.splitext(args.file)[0]
        args.output_file = f"{base_name}_enhanced_analysis.xlsx"

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

        # Generate visualizations unless skipped
        if not args.skip_visualizations:
            print("Generating visualizations...")
            vis_dir = os.path.splitext(args.output_file)[0] + "_visualizations"

            # Add metadata for visualization enhancement
            for result in results:
                # Extract audit leader if available in config
                result["metadata"] = {
                    "Audit Leader": config.get("audit_metadata", {}).get("leader", "Unknown")
                }

            # Generate visualizations
            generate_core_visualizations(results, vis_dir)
            print(f"Visualizations saved to {vis_dir}")

        print(f"Analysis complete. Results saved to {args.output_file}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())