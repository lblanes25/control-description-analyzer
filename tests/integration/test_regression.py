
import os
from control_analyzer import EnhancedControlAnalyzer

# Define test control inputs
test_controls = [
    {
        "control_id": "GOLD-001",
        "description": "The Accounting Manager reviews monthly bank reconciliations prepared by the Senior Accountant to ensure completeness and accuracy. Reconciliations are completed by the 5th business day. Unresolved items over $1,000 are escalated to the Controller.",
        "expected_score": 100,
        "expected_category": "Excellent",
        "expected_missing": []
    },
    {
        "control_id": "GOOD-001",
        "description": "The Accounting Manager reviews reconciliations each month.",
        "expected_score_range": (45, 64),
        "expected_category": "Good"
    },
    {
        "control_id": "WEAK-001",
        "description": "Management reviews periodically.",
        "expected_score_range": (0, 44),
        "expected_category": "Needs Improvement"
    }
]

# Initialize analyzer with config (use your actual config path)
analyzer = EnhancedControlAnalyzer(config_file="control_analyzer_config_final_with_columns.yaml")

def run_tests():
    print("Running regression tests...")
    for test in test_controls:
        result = analyzer.analyze_control(
            control_id=test["control_id"],
            description=test["description"]
        )

        print(f"Testing: {test['control_id']}")
        print(f"Score: {result['total_score']}, Category: {result['category']}")
        print(f"Missing elements: {result['missing_elements']}")

        try:
            if "expected_score" in test:
                assert round(result["total_score"], 1) == test["expected_score"], f"Score mismatch: {result['total_score']}"
            elif "expected_score_range" in test:
                low, high = test["expected_score_range"]
                assert low <= result["total_score"] <= high, f"Score out of expected range: {result['total_score']}"

            assert result["category"] == test["expected_category"], f"Category mismatch: {result['category']}"

            if "expected_missing" in test:
                assert result["missing_elements"] == test["expected_missing"], f"Unexpected missing elements: {result['missing_elements']}"

            print(f"PASS: {test['control_id']}")
        except AssertionError as e:
            print(f"FAIL: {test['control_id']} - {str(e)}")

        print("-------------------")
