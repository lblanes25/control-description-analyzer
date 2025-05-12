import re
from typing import Dict, List, Any, Optional, Tuple, Union


def enhance_escalation_detection(text: str, nlp, existing_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Enhanced ESCALATION detection with improved pattern recognition and context-aware scoring.

    This function identifies escalation paths, exception handling procedures, and reporting
    mechanisms in control descriptions using multiple detection techniques:
    1. Dependency parsing for active voice patterns
    2. Passive voice pattern recognition
    3. Reporting and notification detection
    4. Exception handling identification
    5. Implicit escalation inference

    Args:
        text: The control description text to analyze
        nlp: spaCy NLP model
        existing_keywords: Optional list of custom escalation keywords

    Returns:
        Dictionary containing detailed escalation analysis:
        - detected: Boolean indicating if escalation was detected
        - score: Confidence score (0-3 scale)
        - type: Detection method used
        - phrases: List of identified escalation phrases with metadata
        - suggestions: Improvement suggestions if issues identified
    """
    if not text or text.strip() == '':
        return {
            "detected": False,
            "score": 0,
            "type": None,
            "phrases": [],
            "suggestions": []
        }

    doc = nlp(text)

    # Combine default keywords with custom keywords if provided
    escalation_verbs = set([
        "escalate", "notify", "inform", "report", "raise", "alert", "communicate",
        "elevate", "forward", "route", "submit", "send", "transmit", "escalated",
        "notified", "alerted", "reported"
    ])

    if existing_keywords:
        escalation_verbs.update(existing_keywords)

    # Target entities that escalations are directed to
    escalation_targets = {
        "management", "supervisor", "manager", "leadership", "committee",
        "executive", "director", "cfo", "board", "team", "audit", "compliance",
        "governance", "authority", "senior", "higher", "risk", "department",
        "department head", "lead", "chief", "officer", "owner", "assurance"
    }

    # Issue/exception terms that are often escalated
    issue_terms = {
        "exception", "error", "issue", "discrepancy", "finding", "anomaly",
        "incident", "problem", "concern", "breach", "violation", "deviation",
        "noncompliance", "inconsistency", "irregularity", "variance"
    }

    escalation_phrases = []
    suggestions = []
    matched_types = []

    # ============ ACTIVE VOICE DETECTION ============
    # Using dependency parsing to find active voice escalation patterns
    for token in doc:
        # Check for escalation verbs
        if token.lemma_.lower() in escalation_verbs and token.pos_ in {"VERB", "AUX"}:
            # Track if we found a target in this verb's dependencies
            target_found = False

            # Look for direct targets (e.g., "escalate to manager")
            for child in token.children:
                if child.dep_ in {"dobj", "pobj", "attr", "obl"} and child.text.lower() in escalation_targets:
                    escalation_phrases.append({
                        "text": f"{token.text} {child.text}",
                        "span": [token.idx, child.idx + len(child.text)],
                        "pattern_type": "active_with_target",
                        "target": child.text,
                        "score": 0.9  # High confidence when target is explicit
                    })
                    matched_types.append("active_with_target")
                    target_found = True
                    break

                # Check for prepositional phrase targets (e.g., "escalate to the manager")
                elif child.dep_ == "prep" and child.text.lower() in ["to", "with"]:
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            # Check if target entity or preceded by relevant adjective
                            is_target = (grandchild.text.lower() in escalation_targets or
                                         any(gc.text.lower() in ["senior", "higher", "upper", "appropriate", "proper"]
                                             for gc in grandchild.children))

                            if is_target:
                                # Get the complete target phrase
                                target_span = doc[grandchild.i:grandchild.i + 1]
                                for span in doc.noun_chunks:
                                    if grandchild.i >= span.start and grandchild.i < span.end:
                                        target_span = span
                                        break

                                escalation_phrases.append({
                                    "text": f"{token.text} {child.text} {target_span.text}",
                                    "span": [token.idx, target_span.end_char],
                                    "pattern_type": "active_with_prep_target",
                                    "target": target_span.text,
                                    "score": 0.95  # Very high confidence
                                })
                                matched_types.append("active_with_prep_target")
                                target_found = True
                                break

            # If no direct target but the verb itself is a strong indicator
            if not target_found and token.lemma_.lower() in ["escalate", "escalated", "elevate", "elevated"]:
                escalation_phrases.append({
                    "text": token.text,
                    "span": [token.idx, token.idx + len(token.text)],
                    "pattern_type": "escalation_verb_only",
                    "score": 0.5  # Medium confidence as strong verb with no target
                })
                matched_types.append("escalation_verb_only")
                suggestions.append(
                    "Escalation verb detected without a clear recipient — specify where issues are escalated to")

            # Handle reporting verbs differently (e.g., "report")
            elif not target_found and token.lemma_.lower() in ["report", "notify", "inform", "alert"]:
                reporting_context = get_surrounding_context(text, token.idx, token.idx + len(token.text), 30)
                if any(issue in reporting_context.lower() for issue in issue_terms):
                    escalation_phrases.append({
                        "text": token.text,
                        "span": [token.idx, token.idx + len(token.text)],
                        "pattern_type": "reporting_verb_with_issue",
                        "score": 0.6  # Medium-high confidence as reporting verb with issue context
                    })
                    matched_types.append("reporting_verb_with_issue")
                else:
                    escalation_phrases.append({
                        "text": token.text,
                        "span": [token.idx, token.idx + len(token.text)],
                        "pattern_type": "reporting_verb_only",
                        "score": 0.4  # Medium confidence
                    })
                    matched_types.append("reporting_verb_only")
                    suggestions.append("Reporting verb detected without clear recipient — specify who is notified")

    # ============ PASSIVE VOICE DETECTION ============
    # Key patterns that significantly improve passive voice detection
    passive_patterns = [
        # Core passive patterns with various targets
        (
        r'(?:is|are|will be|must be|should be|shall be)\s+(?:then\s+)?(?:escalated|reported|forwarded|routed|submitted|sent|elevated|notified|communicated|alerted)(?:\s+to\s+(?:the\s+)?([A-Za-z][A-Za-z\s]+))?',
        0.8),

        # These all use capturing groups to extract the target
        (
        r'(?:exceptions|issues|discrepancies|errors|problems|incidents|findings|concerns|violations|breaches|anomalies|variances|deviations)\s+(?:are|will be|must be|should be|shall be)\s+(?:escalated|reported|forwarded|routed|submitted|sent|elevated|notified|communicated)(?:\s+to\s+(?:the\s+)?([A-Za-z][A-Za-z\s]+))?',
        0.85),

        # Financial services specific patterns
        (
        r'(?:exceptions|issues)\s+(?:exceeding|above|greater than|over|more than)\s+(?:[\$€£]\d[\d,.]*k?|threshold|limit|[.\d]+%|[.\d]+\s+percent)\s+(?:are|will be|must be|should be|shall be)\s+(?:escalated|reported|notified)(?:\s+to\s+(?:the\s+)?([A-Za-z][A-Za-z\s]+))?',
        0.9),

        # Documentation patterns
        (
        r'(?:exceptions|issues|discrepancies|errors|problems)\s+(?:are|will be|must be|should be|shall be)\s+(?:documented|recorded|logged|tracked|noted)\s+(?:and|before|prior to|then)\s+(?:escalated|reported|forwarded|elevated)(?:\s+to\s+(?:the\s+)?([A-Za-z][A-Za-z\s]+))?',
        0.85),
    ]

    # Process passive patterns
    for pattern, base_score in passive_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            match_text = match.group(0)

            # Extract target if available from the first capturing group
            target = None
            if len(match.groups()) > 0 and match.group(1):
                target = match.group(1)

            # Adjust score based on context
            adjusted_score = base_score

            # Boost score if target is present
            if target:
                adjusted_score += 0.1

                # Additional boost if target looks like a management role
                if any(role in target.lower() for role in ["manager", "director", "supervisor",
                                                           "executive", "board", "committee",
                                                           "senior", "lead", "chief"]):
                    adjusted_score += 0.05

            # Context-specific adjustments
            context = get_surrounding_context(text, match.start(), match.end(), 40)

            # Boost for issue terms proximity
            if any(issue in context.lower() for issue in issue_terms):
                adjusted_score += 0.05

            # Boost for threshold-based escalation
            if any(term in context.lower() for term in ["threshold", "exceed", "limit", "above", "greater than"]):
                adjusted_score += 0.1

            # Patterns with higher confidence
            pattern_type = "passive_escalation"
            if "exceeding" in match_text.lower() or "threshold" in match_text.lower():
                pattern_type = "threshold_escalation"

            # Cap at 1.0
            adjusted_score = min(1.0, adjusted_score)

            escalation_phrases.append({
                "text": match_text,
                "span": [match.start(), match.end()],
                "pattern_type": pattern_type,
                "target": target,
                "score": adjusted_score
            })
            matched_types.append(pattern_type)

    # ============ EXCEPTION HANDLING DETECTION ============
    # Patterns for identifying exception handling with implied escalation
    exception_patterns = [
        # Documentation and tracking patterns that may imply escalation
        (
        r'(?:exceptions|issues|errors|discrepancies|findings)\s+are\s+(?:documented|logged|tracked|recorded)\s+in\s+(?:the|a|an)\s+([A-Za-z][A-Za-z\s]+)',
        0.4),
        (
        r'(?:exception|issue|error|discrepancy|finding)\s+(?:log|register|tracker|report|documentation|record)\s+is\s+(?:maintained|reviewed|updated)',
        0.4),

        # Resolution and follow-up patterns
        (
        r'(?:exceptions|issues|errors|discrepancies|findings)\s+are\s+(?:resolved|addressed|remediated|fixed|corrected)\s+(?:by|with)\s+(?:the\s+)?([A-Za-z][A-Za-z\s]+)',
        0.5),
        (r'follow(?:-|\s+)up\s+(?:is performed|occurs|takes place|is conducted|is documented)', 0.5),
    ]

    for pattern, base_score in exception_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Extract target system/role if available
            target = None
            if len(match.groups()) > 0 and match.group(1):
                target = match.group(1)

            context = get_surrounding_context(text, match.start(), match.end(), 30)

            # Check for explicit escalation terms nearby for boosting
            context_score_adjustment = 0
            if any(verb in context.lower() for verb in escalation_verbs):
                context_score_adjustment += 0.2

            adjusted_score = min(0.7, base_score + context_score_adjustment)

            escalation_phrases.append({
                "text": match.group(0),
                "span": [match.start(), match.end()],
                "pattern_type": "exception_handling",
                "target": target,
                "score": adjusted_score
            })
            matched_types.append("exception_handling")

    # ============ GOVERNANCE REFERENCES ============
    # Identifying references to escalation governance - more subtle indications of escalation procedures
    governance_patterns = [
        (r'(?:escalation|reporting)\s+(?:process|procedure|protocol|matrix|framework|requirements|policy|guidelines)',
         0.6),
        (
        r'(?:according to|as per|following|based on|in accordance with|per)\s+(?:the\s+)?(?:escalation|reporting)\s+(?:process|procedure|protocol|matrix|framework|requirements|policy)',
        0.6),
    ]

    for pattern, score in governance_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            escalation_phrases.append({
                "text": match.group(0),
                "span": [match.start(), match.end()],
                "pattern_type": "governance_reference",
                "score": score
            })
            matched_types.append("governance_reference")

    # ============ SCORING AND RESULT ASSEMBLY ============
    # Determine if escalation was detected
    detected = len(escalation_phrases) > 0

    # Calculate final score with more nuanced approach
    if detected:
        # Get the highest individual phrase score
        max_phrase_score = max(phrase["score"] for phrase in escalation_phrases)

        # Count pattern types for diversity of evidence
        unique_pattern_types = set(phrase["pattern_type"] for phrase in escalation_phrases)

        # Base score with progressive scaling based on evidence
        if "active_with_prep_target" in matched_types or "threshold_escalation" in matched_types:
            # Very high confidence patterns
            base_score = max_phrase_score + 0.5
        elif "passive_escalation" in matched_types and any(
                target for phrase in escalation_phrases if phrase.get("target")):
            # Passive with explicit target has high confidence
            base_score = max_phrase_score + 0.3
        elif len(unique_pattern_types) >= 2:
            # Multiple pattern types suggest strong evidence
            base_score = max_phrase_score + 0.2
        else:
            # Single pattern type needs less boost
            base_score = max_phrase_score + 0.1

        # Scale by phrase count with diminishing returns
        num_phrases = len(escalation_phrases)
        phrase_count_factor = min(0.4, 0.1 * (1 + num_phrases / 3))
        final_score = base_score + phrase_count_factor

        # Cap at 3.0 as per requirements
        final_score = min(3.0, final_score)
    else:
        final_score = 0.0

    # Generate suggestions for missing escalation
    if not detected and any(term in text.lower() for term in issue_terms):
        suggestions.append(
            "Control mentions exceptions or issues but doesn't specify how they are handled or escalated")
    elif detected and final_score < 1.0:
        suggestions.append("Escalation detected but could be strengthened by specifying who receives the escalation")

    return {
        "detected": detected,
        "score": final_score,
        "type": "comprehensive",
        "phrases": escalation_phrases,
        "suggestions": suggestions
    }


def get_surrounding_context(text: str, start: int, end: int, window_size: int = 40) -> str:
    """
    Get the surrounding context of a match for better contextual analysis.

    Args:
        text: The full text
        start: Start position of match
        end: End position of match
        window_size: Size of context window in each direction

    Returns:
        String containing context before and after the match
    """
    return text[max(0, start - window_size):min(len(text), end + window_size)]