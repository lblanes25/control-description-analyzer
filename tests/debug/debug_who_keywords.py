#!/usr/bin/env python3
"""Debug WHO keywords specifically"""

import sys
sys.path.append('.')
from src.core.analyzer import EnhancedControlAnalyzer
import os

# Initialize analyzer
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config', 'control_analyzer.yaml')
analyzer = EnhancedControlAnalyzer(config_path)

print("🔍 Debugging WHO Keywords Configuration")
print()

# Check the WHO element configuration
who_element = analyzer.elements.get("WHO")
if who_element:
    print("WHO Element Keywords:")
    print(f"  Total keywords: {len(who_element.keywords)}")
    print(f"  First 20 keywords: {who_element.keywords[:20]}")
    print()
    
    # Check if "compliance officer" is in the keywords
    compliance_keywords = [k for k in who_element.keywords if "compliance" in k.lower()]
    print(f"Compliance-related keywords: {compliance_keywords}")
    print()
    
    # Check if "officer" is in the keywords
    officer_keywords = [k for k in who_element.keywords if "officer" in k.lower()]
    print(f"Officer-related keywords: {officer_keywords}")
    print()

# Check the configuration directly
config = analyzer.config
who_config = config.get('who_element', {})
person_roles = who_config.get('person_roles', {})

print("Person Roles Configuration:")
for category, roles in person_roles.items():
    print(f"  {category}: {len(roles)} roles")
    if category == 'audit_compliance':
        print(f"    Audit/Compliance roles: {roles}")
    elif 'compliance' in str(roles).lower():
        compliance_roles = [r for r in roles if 'compliance' in r.lower()]
        if compliance_roles:
            print(f"    Compliance roles in {category}: {compliance_roles}")

print()

# Test the specific phrase
test_text = "The Compliance Officer tests the user entitlements periodically"
print(f"Testing: '{test_text}'")

# Check if we can find the keyword manually
import re
for keyword in who_element.keywords:
    pattern = r'\b' + re.escape(keyword) + r'\b'
    if re.search(pattern, test_text, re.IGNORECASE):
        print(f"  Found keyword match: '{keyword}'")

print("\nDone.")