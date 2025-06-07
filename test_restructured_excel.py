#!/usr/bin/env python3
"""Test the restructured Excel output"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Initialize analyzer
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config', 'control_analyzer_updated.yaml')
analyzer = EnhancedControlAnalyzer(config_path)

print("🧪 Testing Restructured Excel Output")
print()

# Test controls with different characteristics
test_controls = [
    {
        "id": "EXCELLENT001",
        "description": "The Finance Manager reviews and reconciles monthly bank statements within 5 business days to ensure accuracy. Discrepancies exceeding $10,000 are escalated to the CFO."
    },
    {
        "id": "GOOD002", 
        "description": "The Operations Team monitors the monthly reconciliations within defined timelines to ensure compliance with internal policies."
    },
    {
        "id": "NEEDS_IMPROVEMENT003",
        "description": "Staff reviews reports periodically and addresses issues as appropriate."
    }
]

# Analyze the controls
print("Analyzing test controls...")
results = []
for control in test_controls:
    result = analyzer.analyze_control(control['id'], control['description'])
    results.append(result)
    print(f"  ✓ {control['id']} - Elements Found: {result.get('elements_found_count')}, Category: {result.get('simple_category')}")

print()

# Generate Excel report
output_file = "test_restructured_output.xlsx"
print(f"Generating Excel report: {output_file}")

try:
    success = analyzer._generate_enhanced_report(
        results, 
        output_file,
        include_frequency=False,
        include_control_type=False, 
        include_risk_alignment=False
    )
    
    if success:
        print("✅ Excel report generated successfully!")
        print()
        print("Expected structure:")
        print("📄 Tab 1: 'Analysis Results' (Main tab)")
        print("   - Control ID, Description, Elements Found, Simple Category")
        print("   - Missing Elements, WHO/WHEN/WHAT/WHY/ESCALATION Keywords, Vague Terms")
        print("🔒 Tab 2: 'Element Scores & Metadata' (Hidden)")
        print("   - Control ID, Total Score, Category, Individual Element Scores")
        print("   - Multi-control indicators, validation metadata")
        print("📝 Tab 3: 'Enhancement Feedback' (Existing)")
        print("   - Improvement suggestions for each element")
        print()
        print(f"📁 File saved: {output_file}")
        
        # Check if file exists
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"📊 File size: {file_size:,} bytes")
        
    else:
        print("❌ Failed to generate Excel report")
        
except Exception as e:
    print(f"❌ Error generating report: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n🎯 Test complete!")
print("Open the Excel file to verify the new structure:")
print("- Clean 'Analysis Results' tab with user-friendly columns")
print("- Hidden 'Element Scores & Metadata' tab with detailed scoring")
print("- Enhanced subthreshold feedback in 'Enhancement Feedback' tab")