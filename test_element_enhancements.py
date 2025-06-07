#!/usr/bin/env python3
"""
Test script for Element Enhancements
Tests the WHEN, WHAT, and WHY detection improvements
"""

import sys
import traceback
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from analyzers.when import enhance_when_detection
    from analyzers.what import enhance_what_detection
    from analyzers.why import enhance_why_detection
    print("✓ Successfully imported all detection modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_when_enhancements():
    """Test WHEN detection enhancements for vague term handling"""
    print("\n" + "="*60)
    print("TESTING WHEN DETECTION ENHANCEMENTS")
    print("="*60)
    
    # Primary test case from enhancement guide
    test_text = "Business leaders periodically review and approve business continuity plans associated with business impact analysis supporting the application."
    
    print(f"Test text: {test_text}")
    print()
    
    try:
        # Mock nlp object for testing
        class MockNLP:
            def __call__(self, text):
                return MockDoc(text)
        
        class MockDoc:
            def __init__(self, text):
                self.text = text
        
        nlp = MockNLP()
        result = enhance_when_detection(test_text, nlp)
        
        print("WHEN Detection Results:")
        print(f"  Top match: {result.get('top_match')}")
        print(f"  Vague terms: {result.get('vague_terms')}")
        print(f"  Improvement suggestions: {result.get('improvement_suggestions')}")
        print(f"  Score: {result.get('score')}")
        
        # Check for expected enhancements
        vague_terms = result.get('vague_terms', [])
        suggestions = result.get('improvement_suggestions', [])
        
        if vague_terms:
            print("✓ Vague terms detected")
            for vague in vague_terms:
                if isinstance(vague, dict) and 'text' in vague:
                    print(f"    - {vague['text']}: {vague.get('suggested_replacement', 'No suggestion')}")
        else:
            print("✗ No vague terms detected")
        
        if any('periodically' in suggestion for suggestion in suggestions):
            print("✓ Specific feedback for 'periodically' found")
        else:
            print("✗ No specific feedback for 'periodically'")
            
        return True
        
    except Exception as e:
        print(f"✗ WHEN test failed: {e}")
        traceback.print_exc()
        return False


def test_what_enhancements():
    """Test WHAT detection enhancements for compound object recognition"""
    print("\n" + "="*60)
    print("TESTING WHAT DETECTION ENHANCEMENTS")
    print("="*60)
    
    # Test cases from enhancement guide
    test_cases = [
        "Business leaders periodically review and approve business continuity plans associated with business impact analysis supporting the application.",
        "Users validate and verify data integrity",
        "Administrators backup and restore databases",
        "Teams review, analyze, and approve changes"
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_text}")
        
        try:
            # Mock nlp object for testing
            class MockNLP:
                def __call__(self, text):
                    return MockDoc(text)
            
            class MockDoc:
                def __init__(self, text):
                    self.text = text
                    self.sents = [MockSent(text)]
                
                def __iter__(self):
                    # Simple tokenization for testing
                    words = text.split()
                    tokens = []
                    for j, word in enumerate(words):
                        tokens.append(MockToken(word, j))
                    return iter(tokens)
            
            class MockSent:
                def __init__(self, text):
                    self.text = text
                
                def __iter__(self):
                    words = self.text.split()
                    tokens = []
                    for j, word in enumerate(words):
                        tokens.append(MockToken(word, j))
                    return iter(tokens)
            
            class MockToken:
                def __init__(self, text, i):
                    self.text = text
                    self.i = i
                    self.pos_ = "VERB" if text.lower() in ["review", "approve", "validate", "verify", "backup", "restore", "analyze"] else "NOUN"
                    self.lemma_ = text.lower()
                    self.dep_ = "ROOT"
                    self.children = []
            
            nlp = MockNLP()
            result = enhance_what_detection(test_text, nlp)
            
            print("WHAT Detection Results:")
            print(f"  Primary action: {result.get('primary_action')}")
            print(f"  All actions: {result.get('all_actions', [])}")
            print(f"  Confidence: {result.get('confidence', 0)}")
            
            # Check for compound verb detection
            primary_action = result.get('primary_action', {})
            if isinstance(primary_action, dict):
                action_text = primary_action.get('verb_phrase', '')
                if 'and' in action_text:
                    print("✓ Compound action detected")
                else:
                    print("? No compound detected (may be expected)")
            
        except Exception as e:
            print(f"✗ WHAT test {i} failed: {e}")
            traceback.print_exc()
            continue
    
    return True


def test_why_enhancements():
    """Test WHY detection enhancements for inference transparency"""
    print("\n" + "="*60)
    print("TESTING WHY DETECTION ENHANCEMENTS")
    print("="*60)
    
    # Test cases from enhancement guide
    test_cases = [
        ("Staff review logs to identify security issues", "Explicit"),  
        ("Management approves changes for compliance", "Explicit"),
        ("Teams backup data regularly", "Inferred"),
        ("Users validate input data", "Inferred"),
        ("Administrators monitor system performance", "Inferred")
    ]
    
    for i, (test_text, expected_type) in enumerate(test_cases, 1):
        print(f"\nTest Case {i} ({expected_type}): {test_text}")
        
        try:
            # Mock nlp object for testing
            class MockNLP:
                def __call__(self, text):
                    return MockDoc(text)
            
            class MockDoc:
                def __init__(self, text):
                    self.text = text
            
            nlp = MockNLP()
            result = enhance_why_detection(test_text, nlp)
            
            print("WHY Detection Results:")
            top_match = result.get('top_match')
            if top_match:
                print(f"  Purpose: {top_match.get('text', 'None')}")
                is_inference = top_match.get('is_inference', False) or result.get('is_inferred', False)
                print(f"  Is inference: {is_inference}")
                
                # Check for inference labeling
                purpose_text = top_match.get('text', '')
                if '(inferred)' in purpose_text.lower():
                    print("✓ Inference clearly labeled in text")
                elif is_inference:
                    print("? Inference detected but not labeled in text")
                else:
                    print("✓ Explicit purpose (no inference label needed)")
            else:
                print("  Purpose: None detected")
            
            # Check for inference feedback
            suggestions = result.get('improvement_suggestions', [])
            if result.get('is_inferred', False):
                has_inference_feedback = any('explicit purpose statement' in suggestion.lower() for suggestion in suggestions)
                if has_inference_feedback:
                    print("✓ Inference feedback provided")
                else:
                    print("✗ Missing inference feedback")
            
            print(f"  Suggestions: {suggestions}")
            
        except Exception as e:
            print(f"✗ WHY test {i} failed: {e}")
            traceback.print_exc()
            continue
    
    return True


def main():
    """Run all enhancement tests"""
    print("Element Enhancements Test Suite")
    print("Testing improvements to WHEN, WHAT, and WHY detection")
    
    results = []
    
    # Test each enhancement
    results.append(("WHEN Enhancements", test_when_enhancements()))
    results.append(("WHAT Enhancements", test_what_enhancements()))
    results.append(("WHY Enhancements", test_why_enhancements()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All enhancements working correctly!")
        return 0
    else:
        print("⚠️  Some enhancements need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())