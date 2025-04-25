
from control_analyzer import EnhancedControlAnalyzer

# Controls to test each element in isolation
test_cases = [
    # WHO
    {
        "control_id": "WHO-01",
        "description": "The Controller reviews the monthly report.",
        "expect": {"WHO": 20, "missing": []}
    },
    {
        "control_id": "WHO-02",
        "description": "The control is reviewed.",
        "expect": {"WHO": 0, "missing": ["WHO"]}
    },

    # WHEN
    {
        "control_id": "WHEN-01",
        "description": "The reconciliation is performed monthly by the 5th business day.",
        "expect": {"WHEN": 15, "missing": []}
    },
    {
        "control_id": "WHEN-02",
        "description": "The reconciliation is performed periodically.",
        "expect": {"WHEN": 0, "missing": ["WHEN"]}
    },

    # WHAT
    {
        "control_id": "WHAT-01",
        "description": "The Accounting Manager reviews and validates journal entries.",
        "expect": {"WHAT": 20, "missing": []}
    },
    {
        "control_id": "WHAT-02",
        "description": "The entries are reviewed.",
        "expect": {"WHAT": 5, "missing": []}
    },

    # WHY
    {
        "control_id": "WHY-01",
        "description": "This control ensures accuracy and completeness of financial reporting.",
        "expect": {"WHY": 10, "missing": []}
    },
    {
        "control_id": "WHY-02",
        "description": "The reconciliation is performed monthly.",
        "expect": {"WHY": 0, "missing": ["WHY"]}
    },

    # ESCALATION
    {
        "control_id": "ESCALATION-01",
        "description": "Exceptions are escalated to the Controller for resolution.",
        "expect": {"ESCALATION": 3, "missing": []}
    },
    {
        "control_id": "ESCALATION-02",
        "description": "Exceptions are addressed.",
        "expect": {"ESCALATION": 0, "missing": ["ESCALATION"]}
    }
]

analyzer = EnhancedControlAnalyzer(config_file="control_analyzer_config_final_with_columns.yaml")

def run_tests():
    for test in test_cases:
        result = analyzer.analyze_control(
            control_id=test["control_id"],
            description=test["description"]
        )

        print(f"Testing: {test['control_id']}")
        for element, expected_score in test["expect"].items():
            if element == "missing":
                for missing_element in expected_score:
                    assert missing_element in result["missing_elements"], f"{test['control_id']} - Expected missing {missing_element}"
            else:
                actual_score = result["weighted_scores"][element]
                assert actual_score >= expected_score, f"{test['control_id']} - {element} score too low: {actual_score}"
        print("PASS")

if __name__ == "__main__":
    run_tests()
