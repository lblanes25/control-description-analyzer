#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

from src.analyzers.control_classifier import ControlTypeClassifier

# Create classifier with test config
test_config = {
    'classification': {
        'control_participating_verbs': [
            'calculates', 'validates', 'approves', 'alerts', 'flags',
            'reconciles', 'generates', 'processes', 'identifies', 'matches',
            'automatically'
        ],
        'documentation_verbs': [
            'saves', 'stores', 'documents', 'records', 'enters',
            'uploads', 'maintains', 'tracks', 'files'
        ],
        'system_names': [
            'sap', 'oracle', 'peoplesoft', 'sharepoint', 'teams', 'application', 'system'
        ],
        'system_context_weight': 2,
        'location_context_weight': 1
    }
}
classifier = ControlTypeClassifier(test_config)

# Test the new failing cases
print("=== Test 3: Documentation verb (should NOT upgrade) ===")
result3 = classifier.classify_control("Analyst stores documents in system", "manual")
print(f"Upgraded: {result3['upgraded']} (should be False)")
print(f"Final type: {result3['final_type']}")
print(f"System score: {result3['system_score']}")
print(f"Reasoning: {result3['reasoning']}")

# Debug the context detection
print("\n=== Debug Context Detection ===")
text = "analyst stores documents in system"
words = text.split()
try:
    system_idx = words.index("system")
    context_words = words[max(0, system_idx-5):system_idx+6]
    context_text = ' '.join(context_words)
    print(f"Context around 'system': {context_text}")
    
    has_control_verb = any(verb in context_text for verb in classifier.control_participating_verbs)
    has_doc_verb = any(verb in context_text for verb in classifier.documentation_verbs)
    
    print(f"has_control_verb: {has_control_verb}")
    print(f"has_doc_verb: {has_doc_verb}")
    print(f"Should award points: {has_control_verb or (not has_doc_verb)}")
except ValueError:
    print("System not found in words")

# Test the hybrid prominence case  
print("\n=== Test 4: Location prominence (actual test case) ===")
result4 = classifier.classify_control("Manager physically inspects vault using system checklist", "hybrid")
print(f"Final type: {result4['final_type']} (should be location_dependent)")
print(f"System score: {result4['system_score']}")
print(f"Location score: {result4['location_score']}")
print(f"Reasoning: {result4['reasoning']}")

# Debug the scoring breakdown
print("\n=== Debug Location vs System Scoring ===")
text = "Manager physically inspects vault using system checklist"
sys_score = classifier._calculate_system_context_score(text)
loc_score = classifier._calculate_location_context_score(text)
print(f"System score: {sys_score}")
print(f"Location score: {loc_score}")
print(f"Should be location_dependent: {loc_score > sys_score}")

# Debug which patterns are matching
print(f"\n=== Debug Pattern Matching ===")
import re
text_lower = text.lower()
print(f"Text: {text_lower}")

print("Location patterns:")
for pattern in classifier.location_context_patterns:
    if re.search(pattern, text_lower):
        print(f"  MATCH: {pattern}")
    else:
        print(f"  no match: {pattern}")

print("System context - checking 'system' name:")
# Check system scoring logic
words = text_lower.split()
try:
    system_idx = words.index("system")
    context_words = words[max(0, system_idx-5):system_idx+6]
    context_text = ' '.join(context_words)
    print(f"  Context around 'system': {context_text}")
    
    has_control_verb = any(verb in context_text for verb in classifier.control_participating_verbs)
    has_doc_verb = any(verb in context_text for verb in classifier.documentation_verbs)
    
    print(f"  has_control_verb: {has_control_verb}")
    print(f"  has_doc_verb: {has_doc_verb}")
    print(f"  Awards points: {has_control_verb or (not has_doc_verb)}")
except ValueError:
    print("  System not found")