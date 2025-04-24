import re
import spacy
from typing import Dict, List, Any, Optional


def enhance_escalation_detection(text, nlp):
    """Enhanced ESCALATION detection with improved context handling and process awareness"""
    if not text or text.strip() == '':
        return {
            "detected": False,
            "score": 0,
            "type": None,
            "phrases": [],
            "suggestions": []
        }

    # Process text with spaCy
    doc = nlp(text)

    # Initialize results
    escalation_results = {
        "detected": False,
        "score": 0,
        "type": None,
        "phrases": [],
        "suggestions": []
    }

    # Pattern categories with relative scores
    patterns = {
        "explicit": {
            "score": 1.0,
            "patterns": [
                # Role-specific patterns
                r"escalate[ds]?\sto\s(?:the\s)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
                r"notif(?:y|ies|ied)\s(?:the\s)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
                r"report[eds]?\sto\s(?:the\s)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
            ]
        },
        "process_with_governance": {
            "score": 0.7,
            "patterns": [
                r"(?:through|via|using|per|following)\s(?:the\s)?([a-zA-Z\s]+)\sprocess\s(?:.*?\s)?(?:approv|authoriz|review)",
                r"escalate[ds]?\s(?:via|through|to)\s(?:the\s)?([a-zA-Z\s]+)\sprocess\s(?:.*?)?(?:level|tier|approv)",
                r"(?:using|through|following)\s(?:the\s)?([a-zA-Z\s]+)\sprocedure\s(?:.*?)(?:approv|authoriz|review)"
            ]
        },
        "process_generic": {
            "score": 0.4,
            "patterns": [
                r"(?:through|via|using|per|following)\s(?:the\s)?([a-zA-Z\s]+)\sprocess",
                r"escalate[ds]?\s(?:via|through|to)\s(?:the\s)?([a-zA-Z\s]+)\sprocess",
                r"handle[ds]?\sby\s(?:the\s)?([a-zA-Z\s]+)\sprocess"
            ]
        }
    }

    # Search for each pattern type
    for pattern_type, pattern_info in patterns.items():
        for pattern in pattern_info["patterns"]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                escalation_results["detected"] = True

                # If we haven't assigned a type yet, or if this is a higher-scoring type
                if (escalation_results["type"] is None or
                        patterns[escalation_results["type"]]["score"] < pattern_info["score"]):
                    escalation_results["type"] = pattern_type

                # Store matched phrase
                escalation_results["phrases"].append({
                    "text": match.group(0),
                    "span": [match.start(), match.end()],
                    "pattern_type": pattern_type
                })

    # If we detected something, calculate final score
    if escalation_results["detected"]:
        # Base score is the score of the highest-quality match
        base_score = patterns[escalation_results["type"]]["score"]

        # Adjust based on number of elements (diminishing returns)
        elements_bonus = min(0.2, len(escalation_results["phrases"]) * 0.1)

        # Calculate final score (0-1 range)
        escalation_results["score"] = min(1.0, base_score + elements_bonus)

        # Generate suggestions based on detected type
        if escalation_results["type"] == "process_generic":
            escalation_results["suggestions"].append(
                "Consider specifying approval levels or accountable roles in the process reference"
            )
    else:
        # Detect if control mentions exceptions but has no escalation
        if re.search(r"exception|issue|error|discrepanc|problem|failure", text, re.IGNORECASE):
            escalation_results["suggestions"].append(
                "Control mentions exceptions but doesn't specify how they are handled or escalated"
            )

    return escalation_results