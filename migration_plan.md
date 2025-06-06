# Control Analyzer - File Migration Plan

This document maps all existing files to their new locations in the standardized project structure.

## Updated Project Structure with CLI Entry Points

```
control-analyzer/
├── src/
│   ├── __init__.py
│   ├── cli.py                       # Main CLI interface (from integration.py)
│   ├── cli_verbose.py               # Verbose CLI mode (from integration_verbose.py)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── analyzer.py              # from control_analyzer.py
│   │   └── elements.py              # Extract from control_analyzer.py
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── who.py                   # from enhanced_who.py
│   │   ├── what.py                  # from enhanced_what.py
│   │   ├── when.py                  # from enhanced_when.py
│   │   ├── why.py                   # from enhanced_why.py
│   │   ├── escalation.py           # from enhanced_escalation.py
│   │   ├── multi_control.py        # from enhanced_multi_control.py
│   │   └── diagnostic.py           # from enhanced_diagnostic.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config_manager.py        # from config_manager.py
│   │   ├── visualization.py        # from visualization.py
│   │   ├── debug_wrapper.py        # from debug_wrapper.py
│   │   └── diagnostic_simple.py    # from simple_diagnostic.py
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── tableau.py               # from tableau-hyper-integration.py
│   │   ├── tableau_workbook.py     # from tableau_workbook_generator.py
│   │   └── spacy_converter.py      # from convert_to_spacy_format.py
│   └── gui/
│       ├── __init__.py
│       └── main_window.py           # from control_analyzer_gui.py
├── config/
│   └── control_analyzer.yaml        # from control_analyzer_config_final_with_columns.yaml
├── data/
│   ├── input/
│   │   ├── Control_Test_Data2.xlsx
│   │   ├── Test_Controls_for_Analyzer.csv
│   │   ├── generated_controls_for_testing.xlsx
│   │   ├── generated_controls_for_testing_edge_cases.xlsx
│   │   ├── generated_controls_for_testing_realistic.xlsx
│   │   ├── test_controls_enhanced_modules.xlsx
│   │   ├── test_controls_multi_control.xlsx
│   │   ├── what_test_controls.xlsx
│   │   ├── when_test_controls.xlsx
│   │   ├── who_test_controls.xlsx
│   │   └── why_test_controls.xlsx
│   ├── output/
│   │   ├── EdgeCaseResults.xlsx
│   │   ├── test_controls_enhanced_modules_analysis_results.xlsx
│   │   ├── test_controls_enhanced_modules_analysis_results1.xlsx
│   │   ├── test_controls_enhanced_modules_analysis_results2.xlsx
│   │   ├── test_controls_enhanced_modules_analysis_results_backup.xlsx
│   │   ├── when_test_controls_analysis_results.xlsx
│   │   └── visualizations/          # Merge both visualization folders
│   └── models/
├── scripts/
│   ├── generate_review_template.py  # from generate_review_template.py
│   ├── auditor_review_template.py   # from auditor_review_template.py
│   ├── train_model.py              # from train_spacy_model.py
│   ├── validate_spans.py           # from validate_spans.py
│   ├── visualize_entities.py       # from visualize_entities.py
│   └── package.py                  # from package.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_multi_control.py   # from test_enhanced_multi_control.py
│   │   ├── test_elements.py        # from test_elements_suite.py
│   │   └── test_column_mapping.py  # from column_mapping_test.py
│   └── integration/
│       └── test_regression.py       # from test_regression_controls.py
├── docs/
│   ├── user_guide.md               # from Control Description Analyzer - User Guide.txt
│   ├── architecture.md             # from control_analyzer_architecture.md
│   └── system_architecture.md      # from System_Architecture.md
└── ...
```

## Migration Steps

1. **Create new directory structure**
   ```bash
   mkdir -p src/{core,analyzers,utils,integrations,gui}
   mkdir -p {config,scripts,tests/{unit,integration},docs,data/{input,output/visualizations,models}}
   ```

2. **Move and rename files** (in order):
   - Core files first
   - Then analyzers
   - Utilities
   - Scripts
   - Tests
   - Data files

3. **Update imports** in all Python files to reflect new structure:
   ```python
   # Old
   from enhanced_who import enhanced_who_detection_v2
   
   # New
   from src.analyzers.who import enhanced_who_detection_v2
   ```

4. **Create __init__.py files** in each package directory

5. **Update configuration paths** in YAML and code

6. **Merge README files** into single comprehensive README.md

7. **Convert text documentation** to Markdown format

## Files to Archive/Remove

- `Test Text.txt` - Appears to be temporary test file
- `.idea/` folder - IDE-specific files (add to .gitignore)
- `__pycache__/` folders - Python cache (add to .gitignore)
- `gitignore` - Rename to `.gitignore`

## New Files to Create

- `setup.py` - Package installation script
- `pyproject.toml` - Modern Python project configuration
- `requirements-dev.txt` - Development dependencies
- `LICENSE` - License file
- `CONTRIBUTING.md` - Contribution guidelines
- All `__init__.py` files for packages

This migration will result in a clean, professional structure that's easy to navigate and extend.