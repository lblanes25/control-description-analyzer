# Control Analyzer

A Python-based tool for analyzing control descriptions to ensure completeness and quality by identifying essential elements (WHO, WHAT, WHEN, WHY, ESCALATION) and detecting multi-control scenarios.

## Project Structure

```
control-analyzer/
├── src/                              # Core application code
│   ├── core/                         # Core business logic
│   │   ├── analyzer.py              # Main ControlAnalyzer class - orchestrates the analysis pipeline
│   │   └── elements.py              # ControlElement definitions and scoring logic
│   ├── analyzers/                    # Element-specific detection modules
│   │   ├── who.py                   # Detects WHO performs the control (roles, departments)
│   │   ├── what.py                  # Identifies WHAT actions are performed
│   │   ├── when.py                  # Determines WHEN controls are executed (frequency, timing)
│   │   ├── why.py                   # Extracts WHY the control exists (purpose, objectives)
│   │   ├── escalation.py           # Finds escalation procedures and exception handling
│   │   └── multi_control.py        # Detects when multiple controls are combined
│   ├── utils/                        # Utility and helper modules
│   │   ├── config_manager.py        # YAML configuration loading and validation
│   │   ├── nlp_utils.py            # NLP utilities for text processing
│   │   └── excel_utils.py          # Excel file I/O and formatting helpers
│   ├── integrations/                 # External system integrations
│   │   ├── tableau.py               # Tableau Hyper file generation
│   │   └── spacy_converter.py      # Convert data for SpaCy model training
│   └── gui/                          # Graphical user interface
│       └── main_window.py           # Main GUI application window
├── config/                           # Configuration files
│   ├── control_analyzer.yaml        # Main configuration (thresholds, patterns, scoring)
│   └── logging.yaml                 # Logging configuration
├── data/                            # Data directory
│   ├── input/                       # Place input Excel files here
│   ├── output/                      # Analysis results and reports
│   └── models/                      # Trained NLP models
├── scripts/                         # Standalone utility scripts
│   ├── generate_review_template.py  # Generate auditor review templates
│   ├── train_model.py              # Train custom SpaCy models
│   └── validate_controls.py        # Batch validation of control sets
├── tests/                           # Test suite
│   ├── unit/                        # Unit tests for individual components
│   ├── integration/                 # End-to-end integration tests
│   └── fixtures/                    # Test data and sample controls
├── docs/                            # Documentation
│   ├── user_guide.md               # End-user documentation
│   ├── architecture.md             # System design and architecture
│   └── api/                        # API reference documentation
├── examples/                        # Example usage and tutorials
│   ├── basic_analysis.py           # Simple analysis example
│   └── custom_integration.py       # Extending the analyzer
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt             # Development dependencies
├── setup.py                         # Package installation script
├── README.md                        # This file
├── LICENSE                          # License information
├── .gitignore                       # Git ignore patterns
└── pyproject.toml                   # Modern Python project configuration
```

## Quick Start

1. **Installation**
   ```bash
   pip install -r requirements.txt
   python setup.py install
   ```

2. **Basic Usage**
   ```python
   from src.core.analyzer import ControlAnalyzer
   
   analyzer = ControlAnalyzer('config/control_analyzer.yaml')
   results = analyzer.analyze_file('data/input/controls.xlsx')
   ```

3. **GUI Application**
   ```bash
   python -m src.gui.main_window
   ```

## Key Components

### Core Modules
- **analyzer.py**: Main analysis orchestrator that coordinates all element analyzers
- **elements.py**: Defines control elements and scoring methodology

### Analyzers
Each analyzer module focuses on detecting specific control elements:
- **who.py**: Identifies responsible parties using role patterns and NLP
- **what.py**: Extracts action verbs and processes being controlled
- **when.py**: Detects timing, frequency, and scheduling information
- **why.py**: Finds purpose statements and control objectives
- **escalation.py**: Locates exception handling and escalation procedures
- **multi_control.py**: Identifies when multiple controls are improperly combined

### Configuration
The `control_analyzer.yaml` file contains:
- Detection patterns and keywords
- Scoring thresholds
- Column mappings for Excel files
- NLP model settings

## Development

### Setting Up Development Environment
```bash
pip install -r requirements-dev.txt
pre-commit install  # Install git hooks
```

### Running Tests
```bash
pytest tests/                    # Run all tests
pytest tests/unit/              # Run unit tests only
pytest -v --cov=src             # Run with coverage
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function signatures
- Document all public methods and classes
- Run `black` for code formatting
- Run `flake8` for linting

## Data Flow

1. **Input**: Excel file with control descriptions
2. **Processing**: 
   - Load configuration
   - Initialize NLP models
   - Analyze each control for elements
   - Calculate completeness scores
3. **Output**: 
   - Excel report with detailed analysis
   - Optional visualizations
   - Tableau-ready data files

## Extending the Analyzer

To add a new element detector:

1. Create a new module in `src/analyzers/`
2. Implement the detection function following the pattern:
   ```python
   def detect_element(text: str, nlp_doc: Doc, config: dict) -> dict:
       # Your detection logic
       return {
           'found': bool,
           'details': [],
           'score': float
       }
   ```
3. Register it in the main analyzer
4. Add configuration entries in YAML

## License

[Your License Here]

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.