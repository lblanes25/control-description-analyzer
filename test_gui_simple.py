#!/usr/bin/env python3
"""Test simple GUI-like WHO detection"""

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

# Test simple cases that GUI might encounter
test_cases = [
    "Staff reviews monthly reports",
    "The system validates transactions",
    "Manager approves all entries",
    "Reconciliation is performed",
    "The reconciliation is performed by staff"
]

for i, test_text in enumerate(test_cases, 1):
    print(f"\n🧪 Test Case {i}: {test_text}")
    try:
        result = enhanced_who_detection_v2(test_text, nlp)
        print(f"  ✅ Primary: {result.get('primary', {}).get('text', 'None')}")
        print(f"  ✅ Confidence: {result.get('confidence', 0)}")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()