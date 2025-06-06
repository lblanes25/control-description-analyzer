#!/usr/bin/env python3
"""
Visualize Entities Script

This script visualizes the entities in the training data, using either
the spaCy format file or the JSON examples file.

Usage:
    python visualize_entities.py input_file.spacy [--output visualizations.html]
    python visualize_entities.py input_file_examples.json [--output visualizations.html]

Input:
    - spaCy format file (.spacy) or JSON examples file

Output:
    - HTML file with visualizations of the entities
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
import spacy
from spacy import displacy
from spacy.tokens import DocBin


def visualize_entities(input_file, output_file=None, num_examples=50):
    """
    Visualize entities in the training data.

    Args:
        input_file (str): Path to spaCy format file or JSON examples file
        output_file (str, optional): Path to output HTML file
        num_examples (int): Number of examples to visualize
    """
    # Determine input file type
    input_path = Path(input_file)
    file_extension = input_path.suffix.lower()

    examples = []

    if file_extension == '.spacy':
        # Load spaCy format file
        nlp = spacy.blank("en")
        doc_bin = DocBin().from_disk(input_file)
        docs = list(doc_bin.get_docs(nlp.vocab))

        if len(docs) > num_examples:
            docs = random.sample(docs, num_examples)

        examples = docs

    elif file_extension == '.json':
        # Load JSON examples file
        with open(input_file, 'r') as f:
            json_examples = json.load(f)

        # Create spaCy docs from JSON examples
        nlp = spacy.blank("en")

        if len(json_examples) > num_examples:
            json_examples = random.sample(json_examples, num_examples)

        for example in json_examples:
            text = example["text"]
            entities = example.get("entities", [])

            doc = nlp.make_doc(text)
            ents = []
            for entity in entities:
                span = doc.char_span(
                    entity["start"],
                    entity["end"],
                    label=entity["label"]
                )
                if span is not None:
                    ents.append(span)

            doc.ents = ents
            examples.append(doc)

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    if not examples:
        print("No examples found in input file")
        return

    # Set up colors for different entity types
    colors = {
        "WHO": "#7aecec",
        "WHAT": "#bfeeb7",
        "WHEN": "#feca74",
        "WHY": "#ff9561",
        "ESCALATION": "#aa9cfc"
    }

    # Prepare examples with control IDs if available
    examples_to_render = []
    for i, doc in enumerate(examples):
        # Check if we have control ID from JSON
        if file_extension == '.json' and i < len(json_examples):
            control_id = json_examples[i].get("control_id", f"Example {i + 1}")
            doc.user_data = {"title": f"Control ID: {control_id}"}
        else:
            doc.user_data = {"title": f"Example {i + 1}"}
        examples_to_render.append(doc)

    # Visualize entities
    html = displacy.render(
        examples_to_render,
        style="ent",
        options={"colors": colors},
        page=True,
        jupyter=False
    )

    # Add Control IDs to the rendered HTML
    for i, doc in enumerate(examples_to_render):
        title = doc.user_data.get("title", "")
        html = html.replace(f'<div class="entities" style="',
                            f'<div class="control-id">{title}</div><div class="entities" style="',
                            1)  # Replace only the first occurrence for each example

    # Add some custom CSS for better readability
    html = html.replace("</head>", """
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.5; }
        .entities { line-height: 2.5; direction: ltr }
        .example { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
        h2 { color: #333; margin-top: 30px; }
        .entity { border-radius: 0.25em; padding: 0.1em 0.35em; margin: 0 0.25em; line-height: 1; }
        .legend { margin: 20px 0; padding: 15px; background: #f8f8f8; border-radius: 5px; }
        .legend .entity { display: inline-block; margin-right: 15px; }
        .control-id { font-weight: bold; color: #555; margin-bottom: 10px; }
    </style>
    <script>
        // Add a function to show entity counts
        document.addEventListener('DOMContentLoaded', function() {
            // Count entities by type
            let entityCounts = {};
            document.querySelectorAll('.entity').forEach(function(entity) {
                const label = entity.getAttribute('data-entity');
                entityCounts[label] = (entityCounts[label] || 0) + 1;
            });

            // Create legend
            let legend = document.createElement('div');
            legend.className = 'legend';
            legend.innerHTML = '<h3>Entity Counts</h3>';

            for (const [label, count] of Object.entries(entityCounts)) {
                let entitySpan = document.createElement('span');
                entitySpan.className = 'entity';
                entitySpan.setAttribute('data-entity', label);
                entitySpan.textContent = label + ': ' + count;
                legend.appendChild(entitySpan);
            }

            // Add legend at the top
            document.body.insertBefore(legend, document.body.firstChild);
        });
    </script>