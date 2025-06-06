# Configuration Migration Guide

This guide helps you migrate from the old configuration format to the new enhanced configuration structure.

## Overview of Changes

The new configuration structure separates data from behavioral logic, providing:
- Clearer organization following the 5W framework
- Extracted constants for all thresholds
- Dedicated detector classes for each element
- Separated penalty calculations
- Better maintainability and testability

## Key Differences

### 1. NLP Configuration
**Old:**
```yaml
spacy:
  preferred_model: en_core_web_md
  fallback_model: en_core_web_sm
```

**New:**
```yaml
nlp_config:
  preferred_model: en_core_web_md
  fallback_model: en_core_web_sm
```

### 2. Constants and Thresholds
**Old:** Scattered throughout configuration
**New:** Centralized in `constants` section and detector configurations

### 3. Element Configuration
**Old:** Simple keywords list per element
**New:** Structured by detector type with categorized keywords

## Migration Steps

### Option 1: Use the ConfigAdapter (Recommended for Gradual Migration)

1. Replace `ConfigManager` imports with `ConfigAdapter`:
```python
# Old
from src.utils.config_manager import ConfigManager

# New
from src.utils.config_adapter import ConfigAdapter
```

2. Update initialization:
```python
# Old
config_manager = ConfigManager(config_file)

# New
config_adapter = ConfigAdapter(config_file)
```

The ConfigAdapter automatically detects the configuration format and provides a unified interface.

### Option 2: Full Migration to New Structure

1. **Update configuration file:**
   - Rename your config file to `control_analyzer_updated.yaml`
   - Or convert your existing config using the provided template

2. **Update imports:**
```python
from src.utils.config_manager_updated import EnhancedConfigManager, ConfigConstants
```

3. **Update analyzer initialization:**
```python
# In src/core/analyzer.py
def __init__(self, config_file: Optional[str] = None):
    # Use ConfigAdapter instead of ConfigManager
    self.config_adapter = ConfigAdapter(config_file)
    self.config = self.config_adapter.get_config()
    
    # Rest of initialization remains similar
```

## Using New Features

### 1. Detector Classes
```python
# Get a specific detector
person_detector = config_adapter.get_detector('person_role')
if person_detector:
    confidence = person_detector.calculate_confidence(text, matches)
```

### 2. Penalties Manager
```python
# Get penalties manager
penalties = config_adapter.get_penalties_manager()
if penalties:
    vague_penalty = penalties.calculate_vague_term_penalty('severe')
```

### 3. Constants
```python
# Access configuration constants
if config_adapter.is_enhanced_format():
    threshold = ConfigConstants.CONFIDENCE_HIGH
else:
    threshold = config_adapter.get_confidence_threshold('high')
```

## Code Changes Required

### 1. In `src/core/analyzer.py`:

Replace:
```python
from src.utils.config_manager import ConfigManager
```

With:
```python
from src.utils.config_adapter import ConfigAdapter
```

Update initialization:
```python
# Line 487
self.config_adapter = ConfigAdapter(config_file)
self.config = self.config_adapter.get_config()

# Line 565
yaml_columns = self.config_adapter.get_column_defaults()

# Line 586
spacy_config = self.config_adapter.get_nlp_config()
```

### 2. In element detection modules:

Update to use detector classes when available:
```python
def enhanced_who_detection_v2(text, nlp, config):
    # Try to use new detector if available
    if hasattr(config, 'get_detector'):
        person_detector = config.get_detector('person_role')
        system_detector = config.get_detector('system')
        
        if person_detector and system_detector:
            # Use new detection logic
            ...
    
    # Fall back to existing logic
    ...
```

## Backward Compatibility

The ConfigAdapter ensures backward compatibility:
- Automatically detects configuration format
- Provides unified interface for both formats
- Maps old configuration keys to new structure
- Logs warnings for features only available in new format

## Testing the Migration

1. **Test with old configuration:**
```bash
python -m src.cli input.xlsx output.xlsx --config config/control_analyzer.yaml
```

2. **Test with new configuration:**
```bash
python -m src.cli input.xlsx output.xlsx --config config/control_analyzer_updated.yaml
```

Both should produce similar results, with the new configuration providing enhanced detection capabilities.

## Benefits of Migration

1. **Better Organization:** Configuration follows 5W framework
2. **Type Safety:** Detector classes provide type hints
3. **Testability:** Behavioral logic can be unit tested
4. **Maintainability:** Clear separation of concerns
5. **Extensibility:** Easy to add new detectors
6. **Performance:** Optimized confidence calculations

## Troubleshooting

If you encounter issues:

1. **Check configuration format:** Look for 'nlp_config' and 'constants' keys
2. **Verify imports:** Ensure using ConfigAdapter
3. **Check logs:** ConfigAdapter logs which format it detected
4. **Test incrementally:** Use ConfigAdapter first, then migrate fully

## Next Steps

1. Start with ConfigAdapter for immediate compatibility
2. Test thoroughly with existing data
3. Gradually adopt new detector classes
4. Eventually migrate to full enhanced configuration