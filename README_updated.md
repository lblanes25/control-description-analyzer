
# Enhanced Control Description Analyzer

A sophisticated tool for analyzing control descriptions using advanced Natural Language Processing (NLP) techniques. This tool helps identify completeness, clarity, and effectiveness of internal control descriptions by analyzing seven key elements.

## Features

- **Advanced NLP Analysis**: Uses cutting-edge natural language processing to understand control descriptions beyond simple keyword matching
- **Seven Key Elements**: Analyzes WHO, WHAT, WHEN, WHY, EVIDENCE, STORAGE, and ESCALATION components
- **Enhanced Detection Modules**: Specialized modules for each element that understand context, grammar, and intent
- **Multi-Control Detection**: Identifies when a description contains multiple controls that should be separated
- **Risk Alignment**: Evaluates how well control descriptions align with their mapped risks
- **Vague Term Detection**: Identifies ambiguous language and suggests specific alternatives
- **Comprehensive Visualizations**: Interactive charts showing analysis results and patterns
- **Configurable**: Supports customization via YAML configuration files

## Modular Design

The analyzer is modularized for maintainability and scalability:
- `control_analyzer.py`: Core logic and orchestration
- `config_manager.py`: Handles YAML config loading and keyword customization
- `enhanced_who.py`, `enhanced_why.py`, etc.: Specialized NLP modules for each element
- `visualization.py`: Chart generation (radar, bar, missing elements, etc.)
- `integration.py`: Command-line entry point

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/enhanced-control-analyzer.git
   cd enhanced-control-analyzer
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install the required spaCy language model:
   ```
   python -m spacy download en_core_web_md
   ```

## Usage

### Basic Usage

```bash
python integration.py your_controls.xlsx
```

This will analyze the Excel file containing control descriptions and generate both an analysis report and visualizations.

### Advanced Options

```bash
python integration.py your_controls.xlsx   --id-column Control_ID   --desc-column Control_Description   --freq-column Frequency   --type-column Control_Type   --risk-column Risk_Description   --output-file analysis_results.xlsx   --config your_config.yaml
```

### Command-line Arguments

- `file`: Excel file containing control descriptions (required)
- `--id-column`: Column containing control IDs (default: 'A' or 'Control_ID')
- `--desc-column`: Column containing control descriptions (default: 'B' or 'Control_Description')
- `--freq-column`: Column containing frequency values for validation
- `--type-column`: Column containing control type values for validation
- `--risk-column`: Column containing risk descriptions for alignment
- `--output-file`: Custom output file path
- `--config`: Path to YAML configuration file
- `--disable-enhanced`: Disable enhanced detection modules (use base analysis only)
- `--skip-visualizations`: Skip generating visualizations

## Configuration File

You can customize the analyzer behavior using a YAML configuration file. Here's an example:

```yaml
# Element weights and keywords
elements:
  WHO:
    weight: 30
    keywords:
      - controller
      - supervisor
    append: true
  WHEN:
    weight: 20
    keywords:
      - daily basis
      - every two weeks
    append: true
  WHAT:
    weight: 30
    keywords:
      - verify
      - review
      - validate
    append: false

# Vague terms to detect
vague_terms:
  - unclear
  - not defined
  - as appropriate
  - as needed
append_vague_terms: true

# Column mapping (optional if headers match defaults)
columns:
  id: Control_ID
  description: Control_Description
  frequency: Frequency
  type: Control_Type
  risk: Risk_Description

# Feature toggles
use_enhanced_detection: true
```

## Understanding the Output

The analysis generates an Excel workbook with multiple sheets:

1. **Analysis Results**: Main results with scores for each element and overall rating
2. **Keyword Matches**: Detected keywords for each element 
3. **Enhancement Feedback**: Specific suggestions for improving each control
4. **Multi-Control Candidates**: Potential standalone controls that should be separated
5. **Executive Summary**: Overall statistics and findings
6. **Methodology**: Explanation of the analysis approach
7. **Example Controls**: Examples of good and poor control descriptions

Additionally, interactive visualizations are generated in a separate folder, including:

- Score distribution
- Element radar charts
- Missing elements analysis
- Vague term frequency
- Audit leader performance (if metadata provided)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
