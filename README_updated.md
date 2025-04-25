# Enhanced Control Description Analyzer

A sophisticated tool for analyzing control descriptions using advanced Natural Language Processing (NLP) techniques. This tool helps identify completeness, clarity, and effectiveness of internal control descriptions by analyzing key control elements and providing detailed feedback.

## Features

- **Advanced NLP Analysis**: Uses state-of-the-art natural language processing to understand control descriptions beyond simple keyword matching
- **Key Element Detection**: Analyzes all critical components of good control descriptions:
  - **WHO**: Identifies who performs the control with role-specific detection
  - **WHAT**: Analyzes the control actions and verb strength
  - **WHEN**: Detects timing patterns and frequency information
  - **WHY**: Evaluates purpose statements and risk alignment
  - **ESCALATION**: Identifies exception handling and escalation procedures
- **Smart Detection Capabilities**:
  - Distinguishes between human and system performers
  - Identifies passive voice and suggests improvements
  - Detects vague terms and provides specific alternatives
  - Recognizes multi-control descriptions that should be separated
  - Validates consistency with declared frequency and control type
- **Comprehensive Visualizations**: Generates interactive charts showing analysis results and patterns
- **Detailed Reporting**: Provides actionable feedback with specific improvement suggestions
- **Customizable**: Supports extensive configuration via YAML files

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/enhanced-control-analyzer.git
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
python integration.py your_controls.xlsx \
  --id-column Control_ID \
  --desc-column Control_Description \
  --freq-column Frequency \
  --type-column Control_Type \
  --risk-column Risk_Description \
  --output-file analysis_results.xlsx \
  --config your_config.yaml
```

### Command-line Arguments

- `file`: Excel file containing control descriptions (required)
- `--id-column`: Column containing control IDs (default: 'Control_ID')
- `--desc-column`: Column containing control descriptions (default: 'Control_Description')
- `--freq-column`: Column containing frequency values for validation
- `--type-column`: Column containing control type values for validation
- `--risk-column`: Column containing risk descriptions for alignment
- `--output-file`: Custom output file path
- `--config`: Path to YAML configuration file
- `--disable-enhanced`: Disable enhanced detection modules (use base analysis only)
- `--skip-visualizations`: Skip generating visualizations

## Architecture

The analyzer is built with a modular design for maintainability and extensibility:

- `control_analyzer.py`: Core analyzer with comprehensive scoring logic
- `config_manager.py`: Handles YAML configuration loading and management
- `integration.py`: Command-line entry point with file handling
- `visualization.py`: Chart generation for visual analysis
- Enhanced modules:
  - `enhanced_who.py`: Specialized WHO detection with role classification
  - `enhanced_what.py`: Action analysis with verb strength assessment
  - `enhanced_when.py`: Timing detection with vague term identification
  - `enhanced_why.py`: Purpose analysis and risk alignment evaluation
  - `enhanced_escalation.py`: Exception handling and escalation detection

## Output

The analysis generates multiple artifacts:

### Excel Report

The Excel report includes multiple sheets:

1. **Analysis Results**: Main results with scores for each element and overall rating
2. **Keyword Matches**: Detected keywords for each element 
3. **Enhancement Feedback**: Specific suggestions for improving each control
4. **Multi-Control Candidates**: Potential standalone controls that should be separated
5. **Executive Summary**: Overall statistics and findings
6. **Methodology**: Explanation of the analysis approach
7. **Example Controls**: Examples of good and poor control descriptions

### Visualizations

Interactive visualizations are generated in a separate folder, including:

- Score distribution by category
- Element radar charts showing strengths and weaknesses
- Missing elements analysis
- Vague term frequency charts
- Audit leader performance comparisons (if metadata provided)

## Configuration

You can customize the analyzer behavior using a YAML configuration file. Here's an example structure:

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
  
# Vague terms to detect
vague_terms:
  - unclear
  - not defined
  - as appropriate
  - as needed
append_vague_terms: true

# Column mapping
columns:
  id: Control_ID
  description: Control_Description
  frequency: Frequency
  type: Control_Type
  risk: Risk_Description

# Feature toggles
use_enhanced_detection: true
```

## Understanding Control Elements

The analyzer evaluates each control description based on five key elements:

1. **WHO**: Identifies who performs the control
   - Good Example: "The Finance Manager reviews..."
   - Poor Example: "A review is performed..."

2. **WHAT**: Describes the actions being performed
   - Good Example: "...reconciles the accounts and verifies all differences are resolved..."
   - Poor Example: "...checks the accounts..."

3. **WHEN**: Specifies when the control is performed
   - Good Example: "...on a monthly basis by the 5th business day..."
   - Poor Example: "...periodically..."

4. **WHY**: Explains the purpose or risk being mitigated
   - Good Example: "...to ensure completeness and accuracy of financial reporting..."
   - Poor Example: [Missing]

5. **ESCALATION**: Details how exceptions are handled
   - Good Example: "...discrepancies over $1,000 are escalated to the Controller..."
   - Poor Example: [Missing]

## Example Controls

### Excellent Control

"The Accounting Manager reviews the monthly reconciliation between the subledger and general ledger by the 5th business day of the following month. The reviewer examines supporting documentation, verifies that all reconciling items have been properly identified and resolved, and ensures compliance with accounting policies. The review is evidenced by electronic sign-off in the financial system. Any discrepancies exceeding $10,000 are escalated to the Controller and documented in the issue tracking system."

### Needs Improvement Control

"Management reviews financial statements periodically and addresses any issues as appropriate."

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.