# Control Description Analyzer

A sophisticated **natural language processing (NLP) tool** that analyzes control descriptions for completeness using the 5W framework (WHO, WHAT, WHEN, WHERE, WHY) plus ESCALATION detection. Built with **spaCy NLP** for enhanced context-aware detection beyond simple keyword matching.

## Project Overview

### Purpose
The Control Description Analyzer automates the quality assessment of control descriptions by identifying and scoring critical elements using a **weighted scoring methodology**:

#### Core Elements (100% of base score):
- **WHO** (30%): Person, role, or system performing the control
- **WHAT** (35%): Specific action being performed  
- **WHEN** (35%): Timing or frequency of the control

#### Conditional Elements:
- **WHERE** (conditional scoring): Location/system information with relevance multipliers based on control type

#### Feedback-Only Elements (no scoring impact):
- **WHY**: Purpose or objective of the control (provides feedback but doesn't affect score)
- **ESCALATION**: Exception handling and escalation procedures (provides feedback but doesn't affect score)

### Target Users
- **Audit teams** conducting quality reviews of control portfolios
- **Control owners** seeking to improve control descriptions
- **Compliance professionals** validating control documentation
- **Management** requiring portfolio-wide control analysis and reporting

### Technology
- **spaCy NLP Engine**: Advanced natural language processing for context-aware detection
- **Enhanced Detection Modules**: Specialized analyzers for each element with semantic understanding
- **Configurable Architecture**: All logic controlled via YAML configuration files
- **Batch Processing**: Efficient handling of large control portfolios (500+ controls)

### Key Benefits
- **Automated Scoring**: Objective assessment with detailed feedback and improvement suggestions
- **Vague Term Detection**: Identifies and suggests replacements for ambiguous language
- **Multi-Control Analysis**: Detects when descriptions contain multiple controls
- **Validation Capabilities**: Cross-validates frequency and control type declarations
- **Rich Visualizations**: Interactive dashboards and executive reporting

## ⚙️ Configuration Management

### Config-Driven Architecture
**ALL analysis logic is controlled via `config/control_analyzer.yaml`** - no code changes required for customization.

### Customizable Parameters

#### Scoring Configuration
```yaml
scoring:
  # Core element weights (WHO + WHAT + WHEN = 100%)
  core_elements:
    WHO: 30     # Accountability and ownership
    WHAT: 35    # Control action and verb strength
    WHEN: 35    # Timing and frequency
    
  # Conditional element scoring
  conditional_elements:
    WHERE:
      system_controls: 1.10      # 10% bonus for system controls
      location_dependent: 1.05   # 5% bonus for location-dependent controls
      other: 1.00               # No bonus for other controls
      
  # Feedback-only elements (no scoring impact)
  feedback_elements:
    WHY: true         # Purpose analysis for feedback only
    ESCALATION: true  # Exception handling for feedback only

  # Category thresholds
  category_thresholds:
    meets_expectations: 75    # 75+ points = Meets Expectations
    requires_attention: 50    # 50-74 points = Requires Attention
    # <50 points = Needs Improvement
```

#### Control Type Relevance (WHERE Scoring)
```yaml
where_element:
  control_type_relevance:
    Manual:
      systems: 0.8        # Less dependent on specific systems
      locations: 1.2      # More dependent on physical locations
      organizational: 1.3  # Heavily dependent on organizational context
    Automated:
      systems: 1.4        # Highly dependent on specific systems
      locations: 0.8      # Less dependent on physical locations
      organizational: 1.0  # Standard organizational dependency
    Hybrid:
      systems: 1.1        # Moderate system dependency
      locations: 1.0      # Standard location dependency
      organizational: 1.1  # Slightly higher organizational dependency
```

#### Vague Term Detection
```yaml
vague_terms:
  - appropriate    # Flagged as vague
  - periodically
  - as needed
  # Custom penalties applied
```

#### Column Mappings for Excel Files
```yaml
column_mapping:
  primary_columns:
    - Control Description    # Primary column name to look for
    - Control_Description    # Alternative column name
    - Control Statement      # Another alternative
    
  control_id_columns:
    - Control ID
    - Control_ID
    - Control #
    
  control_type_columns:      # Maps to control nature (Preventive/Detective/Corrective)
    - Control Type
    - Control_Type
    - Type
    
  control_automation_columns: # Maps to automation level (Manual/Hybrid/Automated)
    - Control Automation
    - Automation Type
    - Manual/Automated
    
  risk_columns:
    - Key Risk Description
    - Risk Description
    - Risk
    
  owner_columns:             # For audit leader information
    - Audit Leader
    - Control Owner
    - Owner
    
  entity_columns:            # For organizational grouping
    - Audit Entity
    - Business Process
    - Department
```

### How to Modify Configuration

#### Common Changes

**Add new WHO keywords:**
```yaml
elements:
  WHO:
    keywords:
      - "your new role"
      - "custom system name"
```

**Adjust scoring thresholds:**
```yaml
scoring:
  category_thresholds:
    meets_expectations: 80  # Raise bar for "Meets Expectations"
    requires_attention: 60  # Raise bar for "Requires Attention"
```

**Modify control type relevance for WHERE scoring:**
```yaml
where_element:
  control_type_relevance:
    Manual:
      systems: 0.9        # Adjust system relevance for manual controls
      locations: 1.3      # Increase location importance
      organizational: 1.4  # Increase organizational context importance
```

**Change column mappings (add your column names to the lists):**
```yaml
column_mapping:
  primary_columns:
    - Control Description
    - Your Custom Description Column  # Add your column name
  control_id_columns:
    - Control ID
    - Your Control ID Column         # Add your column name
```

### Config Validation
The `ConfigAdapter` class handles all configuration access and provides sensible defaults if values are missing, ensuring robust operation.

## Enhanced Detection Architecture

### Specialized Modules
Each element has a dedicated enhanced detection module with sophisticated NLP capabilities:

#### Core Modules
- **`src/analyzers/who.py`**: Person/role/system detection with entity recognition
- **`src/analyzers/what.py`**: Action detection with compound verb analysis  
- **`src/analyzers/when.py`**: Timing detection with vague term identification
- **`src/analyzers/where.py`**: Location/system detection with control type relevance
- **`src/analyzers/where_service.py`**: Shared WHERE detection service used by WHAT and WHERE analyzers
- **`src/analyzers/why.py`**: Purpose detection with inference transparency  
- **`src/analyzers/escalation.py`**: Exception handling detection
- **`src/analyzers/multi_control.py`**: Multi-control identification
- **`src/analyzers/diagnostic.py`**: Comprehensive diagnostic analysis

#### Advanced Features
- **Context-Aware Scoring**: Semantic similarity and dependency parsing
- **Relationship Analysis**: Cross-element validation and consistency checking
- **Inference Labeling**: Clear distinction between explicit and inferred content
- **Compound Detection**: Recognition of complex phrases like "review and approve"
- **Voice Analysis**: Active vs. passive voice detection and recommendations

### Integration Architecture
- **`src/core/analyzer.py`**: Main orchestration engine
- **`src/utils/config_adapter.py`**: Configuration management layer
- **`src/utils/visualization.py`**: Chart and dashboard generation

## Getting Started

### Prerequisites
```bash
# Install required packages
pip install pandas spacy openpyxl pyyaml plotly

# Download spaCy language model
python -m spacy download en_core_web_md
```

**Note**: The tool uses `en_core_web_md` for better accuracy, with automatic fallback to `en_core_web_sm` if unavailable.

### Basic Usage
```bash
# Analyze controls from Excel file (specify config file)
python -m src.cli controls.xlsx --config config/control_analyzer.yaml --output-file results.xlsx

# With custom column names
python -m src.cli controls.xlsx \
  --config config/control_analyzer.yaml \
  --id-column "Control Number" \
  --desc-column "Description" \
  --output-file results.xlsx

# Generate visualizations and open dashboard
python -m src.cli controls.xlsx \
  --config config/control_analyzer.yaml \
  --output-file results.xlsx \
  --open-dashboard

# For large files, use batch processing
python -m src.cli controls.xlsx \
  --config config/control_analyzer.yaml \
  --output-file results.xlsx \
  --use-batches --batch-size 500
```

### Required Input Columns (Configurable)
- **Control ID**: Unique identifier for each control
- **Control Description**: The control text to analyze

### Optional Input Columns  
- **Control Frequency**: For timing validation (e.g., "Monthly", "Quarterly")
- **Control Type**: For control nature validation (e.g., "Preventive", "Detective", "Corrective")
- **Control Automation**: For automation level and WHERE scoring (e.g., "Manual", "Hybrid", "Automated")
- **Risk Description**: For WHY element alignment analysis
- **Audit Leader**: For portfolio reporting and filtering
- **Audit Entity**: For organizational grouping

### Output
- **Excel Workbook**: Multi-sheet report with detailed analysis
- **Interactive Visualizations**: HTML dashboards and charts
- **Element-by-Element Results**: Scores, missing elements, suggestions

## Output and Visualizations

### Excel Report Sheets
1. **Summary**: Executive overview with score distributions
2. **Detailed Results**: Complete analysis for each control
3. **Missing Elements**: Controls lacking specific elements
4. **Vague Terms**: Flagged ambiguous language with suggestions
5. **Multi-Controls**: Descriptions containing multiple controls
6. **Validation Results**: Frequency and type validation outcomes

### Interactive Dashboards
Generated via `src/utils/visualization.py`:
- **Score Distribution**: Portfolio performance overview
- **Missing Elements Frequency**: Most common gaps
- **Audit Leader Breakdown**: Performance by responsible party
- **Element Radar Charts**: Multi-dimensional analysis
- **Worst Performing Controls**: Priority improvement targets

### Filtering Capabilities
Results can be filtered by:
- Audit leader/entity
- Score categories (Excellent/Good/Poor)  
- Missing elements
- Control types
- Score ranges

## 🔧 Maintenance and Development

### Dependencies
```bash
# Core functionality
pandas>=1.3.0        # Data manipulation
spacy>=3.4.0         # NLP processing  
openpyxl>=3.0.0      # Excel file handling
PyYAML>=5.4.0        # Configuration files

# Visualization
plotly>=5.0.0        # Interactive charts

# Optional GUI
PyQt5>=5.15.0        # Desktop interface (if using GUI)
```

### SpaCy Models
- **Primary**: `en_core_web_md` (150MB, better accuracy for complex text)
- **Fallback**: `en_core_web_sm` (13MB, lightweight alternative)

### Code Structure
```
src/
├── cli.py                    # Main command-line interface
├── cli_verbose.py           # Alternative CLI with detailed output
├── core/
│   └── analyzer.py          # Core analysis orchestration
├── analyzers/               # Element-specific detection modules
│   ├── who.py              # WHO detection with NER
│   ├── what.py             # WHAT detection with verb analysis
│   ├── when.py             # WHEN detection with timing patterns
│   ├── where.py            # WHERE detection as standalone element
│   ├── where_service.py    # Shared WHERE detection service
│   ├── why.py              # WHY detection with purpose inference
│   ├── escalation.py       # Exception handling detection
│   ├── multi_control.py    # Multi-control identification
│   ├── diagnostic.py       # Comprehensive diagnostics
│   └── control_classifier.py # Control classification utilities
├── utils/                   # Supporting utilities
│   ├── config_adapter.py   # Configuration management
│   ├── config_manager.py   # Legacy config manager
│   ├── debug_wrapper.py    # Debug utilities
│   ├── diagnostic_simple.py # Simple diagnostic tools
│   └── visualization.py    # Chart and dashboard generation
├── integrations/            # External system integrations
│   ├── spacy_converter.py  # spaCy model conversion
│   ├── tableau.py          # Tableau integration
│   └── tableau_workbook.py # Tableau workbook generation
└── gui/                     # Optional desktop interface
    └── main_window.py      # PyQt5 GUI (if available)

config/
└── control_analyzer.yaml   # Main configuration file

tests/                       # Test coverage
├── unit/                   # Unit tests for individual modules
└── integration/           # End-to-end testing
```

### GUI Components  
- **Desktop Interface**: Optional PyQt5-based GUI in `src/gui/`
- **Web Interface**: HTML visualizations generated by analysis
- **Command Line**: Primary interface via `src/cli.py`

## Testing and Quality Assurance

### Sample Control Descriptions

**Meets Expectations (75+ points):**
```
"The Finance Manager reviews and reconciles monthly bank statements 
within 5 business days of month-end to ensure accuracy and identify 
any unauthorized transactions. Discrepancies are investigated and 
resolved within 2 business days, with findings reported to the CFO."
```
- WHO: Finance Manager (clear role)
- WHAT: reviews and reconciles (strong action verbs)
- WHEN: within 5 business days of month-end (specific timing)
- WHERE: bank statements (system/location context)

**Requires Attention (50-74 points):**
```
"Staff review monthly reconciliations to ensure accuracy. 
Any exceptions are escalated to management for resolution."
```
- WHO: Staff (somewhat vague)
- WHAT: review (moderate action)
- WHEN: monthly (basic timing)
- ESCALATION: escalated to management (good)

**Needs Improvement (<50 points):**
```
"Reconciliations are performed as needed."
```
- WHO: Missing (who performs?)
- WHAT: performed (weak action)
- WHEN: as needed (vague timing)

### Validation Features
- **Frequency Validation**: Cross-checks declared vs. detected timing
- **Control Type Validation**: Validates preventive/detective/corrective alignment  
- **Risk Alignment**: Analyzes WHY element consistency with risk descriptions
- **Multi-Control Detection**: Flags descriptions containing multiple controls

### Testing Framework
```bash
# Run unit tests
python -m pytest tests/unit/

# Run integration tests  
python -m pytest tests/integration/

# Test with sample data
python -m src.cli tests/fixtures/sample_controls.xlsx \
  --config config/control_analyzer.yaml \
  --output-file test_results.xlsx
```

## 👥 Team Handoff Information

### Key Contact Information
- **Original Development Team**: [Internal Development Team]
- **Last Stable Version**: December 2024
- **Documentation Date**: December 2024

### Known Issues
- **Large Files**: Files with 1000+ controls may require batch processing (`--use-batches` flag)
- **Memory Usage**: spaCy models require ~500MB RAM; use `en_core_web_sm` for resource-constrained environments
- **Excel Compatibility**: Requires `.xlsx` format; `.xls` files must be converted

### Future Enhancements
- **Machine Learning**: Training custom models on organization-specific control patterns
- **API Integration**: REST API for system-to-system integration
- **Real-time Analysis**: Web-based interface for immediate feedback
- **Bulk Import**: Support for additional file formats (CSV, JSON)

### Support Resources

**Configuration Questions:**
- Review YAML comments and examples in `config/control_analyzer.yaml`
- Configuration changes require no code modifications
- Default values provided for all optional settings

**Code Modifications:**
- Each enhanced module (`src/analyzers/*.py`) is self-contained
- Modify detection logic by updating configuration files first
- New keywords/patterns can be added via YAML configuration

**Performance Issues:**
- Use batch processing for large files: `--use-batches --batch-size 250`
- Consider using lighter spaCy model: Configure `fallback_model: en_core_web_sm`
- Monitor memory usage during processing

**Common Troubleshooting:**

| Issue | Solution |
|-------|----------|
| "spaCy model not found" | Run `python -m spacy download en_core_web_md` |
| "Column not found" | Check column names with `--debug` flag |
| "Out of memory" | Use `--use-batches` with smaller `--batch-size` |
| "No visualizations" | Check `plotly` installation: `pip install plotly` |

### Success Criteria

A new team member should be able to:
1. **Understand the tool's purpose** within 5 minutes of reading this README
2. **Successfully run their first analysis** within 15 minutes using sample data
3. **Modify common configurations** (keywords, thresholds) without code changes
4. **Troubleshoot basic issues** using documentation alone
5. **Scale to production usage** with batch processing and configuration tuning

### Quick Start Checklist
- [ ] Install dependencies: `pip install pandas spacy openpyxl pyyaml plotly`
- [ ] Download spaCy model: `python -m spacy download en_core_web_md`
- [ ] Test with sample: `python -m src.cli sample.xlsx --config config/control_analyzer.yaml --output-file test.xlsx`
- [ ] Review configuration: `config/control_analyzer.yaml`  
- [ ] Generate visualizations: Add `--open-dashboard` flag
- [ ] For large files: Add `--use-batches` flag
- [ ] Ensure your Excel columns match the configuration or add them to `column_mapping` section

---

**Ready to analyze your control descriptions?** Start with the basic usage examples above and gradually customize the configuration to match your organization's specific needs.