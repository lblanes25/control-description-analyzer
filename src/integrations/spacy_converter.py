#!/usr/bin/env python3
"""
Convert to spaCy Format Script

This script takes a reviewed Excel template and converts it to spaCy training data format.

Usage:
    python convert_to_spacy_format.py reviewed_template.xlsx output_file.spacy [--train-split 0.8]

Input:
    - Excel file with columns: Control ID, Description, WHO Spans, WHAT Spans, etc.

Output:
    - spaCy training data file(s) (.spacy format)
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import re
import random
import json
from pathlib import Path
import spacy
from spacy.tokens import DocBin


def excel_format_to_spans(formatted_str):
    """
    Convert Excel-formatted spans back to list of tuples.

    Args:
        formatted_str (str): Formatted string from Excel (keyword|start|end; ...)

    Returns:
        list: List of (keyword, start, end) tuples
    """
    if pd.isna(formatted_str) or formatted_str.strip() == "":
        return []

    spans = []
    for span_str in formatted_str.split(";"):
        span_str = span_str.strip()
        if not span_str:
            continue

        parts = span_str.split("|")
        if len(parts) >= 3:
            keyword = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
                spans.append((keyword, start, end))
            except ValueError:
                # Skip if start/end aren't valid integers
                continue

    return spans


def convert_to_spacy_format(input_file, output_file, train_split=0.8):
    """
    Convert reviewed Excel template to spaCy training data.

    Args:
        input_file (str): Path to reviewed Excel template
        output_file (str): Path to output spaCy file
        train_split (float): Split ratio for train/dev if creating both
    """
    # Read the input Excel file
    df = pd.read_excel(input_file)

    # Check for required columns
    required_cols = ["Control ID", "Description"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input file")

    # Elements to process
    elements = ["WHO", "WHAT", "WHEN", "WHY", "ESCALATION"]

    # Check if spans columns exist
    spans_cols = [f"{element} Spans" for element in elements]
    available_elements = [element for element, col in zip(elements, spans_cols) if col in df.columns]

    if not available_elements:
        raise ValueError("No element spans columns found in input file")

    print(f"Found spans columns for elements: {available_elements}")

    # Create spaCy NLP object for tokenization
    nlp = spacy.blank("en")

    # Create DocBin objects for train and dev sets
    train_db = DocBin()
    dev_db = DocBin()

    # Create entity mapping
    entity_mapping = {}
    for element in available_elements:
        entity_mapping[element] = element

    # Create a list to save examples in JSON format (for reference)
    examples = []

    # Process each row
    total_examples = 0
    total_entities = 0
    element_counts = {element: 0 for element in available_elements}

    for i, row in df.iterrows():
        control_id = row["Control ID"]
        description = row["Description"]

        if pd.isna(description) or description.strip() == "":
            continue

        # Get all spans for this control
        all_spans = []
        for element in available_elements:
            spans_col = f"{element} Spans"
            if spans_col in df.columns:
                spans = excel_format_to_spans(row[spans_col])

                # Add to element counts
                element_counts[element] += len(spans)

                # Add to total entities count
                total_entities += len(spans)

                # Convert to spaCy entity format and add to list
                entity_label = entity_mapping[element]
                for keyword, start, end in spans:
                    all_spans.append((start, end, entity_label))

        # Sort spans by start position
        all_spans.sort()

        # Check for overlapping spans
        non_overlapping_spans = []
        last_end = -1
        for start, end, label in all_spans:
            if start >= last_end:  # No overlap
                non_overlapping_spans.append((start, end, label))
                last_end = end
            else:
                print(f"Warning: Skipping overlapping span {start}-{end} ({label}) in Control ID: {control_id}")

        # Create spaCy Doc
        doc = nlp.make_doc(description)

        # Add entities to the Doc
        ents = []
        for start, end, label in non_overlapping_spans:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
            else:
                print(f"Warning: Could not create span for {start}-{end} ({label}) in Control ID: {control_id}")

        # Set entities for the Doc
        doc.ents = ents

        # Add to DocBin (split between train and dev)
        if random.random() < train_split:
            train_db.add(doc)
        else:
            dev_db.add(doc)

        # Add to examples list
        example = {
            "control_id": control_id,
            "text": description,
            "entities": [{"start": start, "end": end, "label": label} for start, end, label in non_overlapping_spans]
        }
        examples.append(example)
        total_examples += 1

    # Save DocBin to file
    output_path = Path(output_file)

    # If output has extension, create train and dev files
    if output_path.suffix:
        base_path = output_path.with_suffix('')
        train_path = f"{base_path}_train{output_path.suffix}"
        dev_path = f"{base_path}_dev{output_path.suffix}"

        train_db.to_disk(train_path)
        dev_db.to_disk(dev_path)

        print(f"Saved {train_db.count} training examples to {train_path}")
        print(f"Saved {dev_db.count} development examples to {dev_path}")
    else:
        # If no extension, use .spacy
        train_path = f"{output_path}_train.spacy"
        dev_path = f"{output_path}_dev.spacy"

        train_db.to_disk(train_path)
        dev_db.to_disk(dev_path)

        print(f"Saved {train_db.count} training examples to {train_path}")
        print(f"Saved {dev_db.count} development examples to {dev_path}")

    # Save examples as JSON for reference
    json_path = f"{output_path}_examples.json"
    with open(json_path, 'w') as f:
        json.dump(examples, f, indent=2)

    print(f"Saved {len(examples)} examples as JSON to {json_path}")

    # Print statistics
    print("\nStatistics:")
    print(f"Total examples: {total_examples}")
    print(f"Total entities: {total_entities}")
    print("Entities by type:")
    for element, count in element_counts.items():
        print(f"  {element}: {count}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert reviewed Excel template to spaCy training data")
    parser.add_argument("input_file", help="Input Excel template with reviewed spans")
    parser.add_argument("output_file", help="Output spaCy file (without extension for train/dev split)")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Ratio of examples to use for training (default: 0.8)")
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    convert_to_spacy_format(args.input_file, args.output_file, args.train_split)
    return 0


if __name__ == "__main__":
    sys.exit(main())