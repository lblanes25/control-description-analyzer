#!/usr/bin/env python3
"""Test Finance Manager detection"""

import sys
sys.path.append('.')

from src.analyzers.who import enhanced_who_detection_v2
import spacy

# Load spacy model
try:
    nlp = spacy.load('en_core_web_md')
    print("✅ Using en_core_web_md")
except OSError:
    try:
        nlp = spacy.load('en_core_web_sm')
        print("✅ Using en_core_web_sm")
    except OSError:
        print("❌ No spaCy model found")
        sys.exit(1)

# Test the Finance Manager case
test_text = "The Finance Manager reviews and reconciles monthly bank statements within 5 business days of month-end to ensure accuracy and identify any unauthorized transactions. Discrepancies are investigated and resolved within 2 business days, with findings reported to the CFO."

print(f"\n🧪 Testing Finance Manager detection:")
print(f"Text: {test_text}")
print("\n" + "="*60)

result = enhanced_who_detection_v2(test_text, nlp)

print("\n" + "="*60)
print(f"🎯 RESULT:")
print(f"Primary: {result.get('primary', {}).get('text', 'None')}")
print(f"Confidence: {result.get('confidence', 0)}")
print(f"Detection methods: {result.get('detection_methods', [])}")

if result.get('secondary'):
    print(f"Secondary: {[s.get('text', 'Unknown') for s in result['secondary']]}")