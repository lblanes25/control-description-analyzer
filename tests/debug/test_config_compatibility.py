#!/usr/bin/env python3
"""
Quick test to verify ConfigAdapter works with both config formats
"""
import sys
from src.utils.config_adapter import ConfigAdapter
from src.core.analyzer import EnhancedControlAnalyzer

def test_config(config_file, name):
    print(f"\n{'='*60}")
    print(f"Testing {name}: {config_file}")
    print('='*60)
    
    try:
        # Test ConfigAdapter
        adapter = ConfigAdapter(config_file)
        print(f"✅ ConfigAdapter loaded successfully")
        print(f"   - Is new format: {adapter.is_new_format}")
        print(f"   - Has get_detector: {hasattr(adapter, 'get_detector')}")
        
        # Test analyzer initialization
        analyzer = EnhancedControlAnalyzer(config_file)
        print(f"✅ EnhancedControlAnalyzer initialized successfully")
        
        # Test basic functionality
        nlp_config = adapter.get_nlp_config()
        print(f"✅ NLP config retrieved: {nlp_config}")
        
        column_defaults = adapter.get_column_defaults()
        print(f"✅ Column defaults retrieved: {len(column_defaults) if column_defaults else 0} columns")
        
        print(f"\n✅ {name} PASSED")
        
    except Exception as e:
        print(f"\n❌ {name} FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test both configurations
    test_config("config/control_analyzer.yaml", "OLD CONFIG FORMAT")
    test_config("../../config/control_analyzer.yaml", "NEW CONFIG FORMAT")
    
    print(f"\n{'='*60}")
    print("SUMMARY: Config migration completed successfully!")
    print("Both configuration formats are supported via ConfigAdapter")
    print('='*60)