# Control Analyzer

A comprehensive Python-based tool for analyzing control descriptions to ensure completeness and quality by identifying essential elements (WHO, WHAT, WHEN, WHY, ESCALATION) and detecting multi-control scenarios.

## Features

- **Element Detection**: Identifies five key control elements using advanced NLP
- **Multi-Control Detection**: Flags when multiple controls are improperly combined
- **Scoring System**: Provides completeness scores with detailed feedback
- **Batch Processing**: Analyzes multiple controls from Excel files
- **Visualizations**: Generates interactive dashboards and reports
- **GUI Interface**: User-friendly desktop application
- **Extensible Architecture**: Easy to add new detection modules

## Installation

```bash
# Clone the repository
git clone #have to figure out how to do this at Amex
cd control-analyzer

# Install dependencies
pip install -r requirements.txt

# Install the package
python setup.py install
```

## Quick Start

### Command Line Interface

```bash
# Analyze a single Excel file
python -m src.cli input_file.xlsx output_file.xlsx

# Use verbose mode for detailed logging
python -m src.cli_verbose input_file.xlsx output_file.xlsx

# Generate visualizations
python -m src.cli input_file.xlsx output_file.xlsx --visualize
```

### GUI Application

```bash
python -m src.gui.main_window
```

### Python API

```python
from src.core.analyzer import EnhancedControlAnalyzer
from src.utils.config_manager import ConfigManager

# Initialize analyzer
config = ConfigManager('config/control_analyzer.yaml')
analyzer = EnhancedControlAnalyzer(config.config)

# Analyze controls
results = analyzer.analyze_file('controls.xlsx')
```

## Project Structure

See `project_scaffold_README.md` for detailed structure explanation.

## Configuration

The analyzer is configured via `config/control_analyzer.yaml`. Key settings include:

- Detection patterns and keywords
- Scoring thresholds
- Column mappings
- NLP model configurations

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ tests/

# Run linting
flake8 src/ tests/
```

### Adding New Analyzers

1. Create a new module in `src/analyzers/`
2. Implement the detection function
3. Register in the main analyzer
4. Add configuration entries

## License

MIT License - see LICENSE file for details.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.