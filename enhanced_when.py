#!/usr/bin/env python3
"""
Enhanced WHEN Element Detection Module

This module analyzes control descriptions to identify timing elements using a
combination of keyword matching, regex patterns, and NLP techniques.
The implementation prioritizes effectiveness over efficiency, using a layered
detection strategy optimized for the most contextually relevant timing patterns.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Union


def enhance_when_detection(text: str, nlp, control_type: Optional[str] = None,
                           automation_level: Optional[str] = None,
                           existing_keywords: Optional[Union[Dict[str, List[str]], List[str]]] = None,
                           frequency_metadata: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhanced WHEN detection with improved handling of complex timing patterns,
    multi-frequency detection, and strict handling of vague terms.

    Args:
        text: Control description text
        nlp: Loaded spaCy model
        control_type: Optional control type (preventive, detective, corrective)
        automation_level: Optional automation level (manual, automated, hybrid)
        existing_keywords: Optional dictionary or list of timing keywords to supplement defaults
        frequency_metadata: Optional declared frequency from metadata for validation

    Returns:
        Dictionary with detection results, scores, and improvement suggestions
    """
    if not text or text.strip() == '':
        return build_empty_result("No text provided")

    # Create config from existing_keywords
    config = {}
    if existing_keywords:
        if isinstance(existing_keywords, dict):
            config['frequency_terms'] = existing_keywords
        elif isinstance(existing_keywords, list):
            # Convert list to dict with "other" category
            config['frequency_terms'] = {
                "other": [kw for kw in existing_keywords if kw.lower() != "may"]
            }

    # Preprocess text
    text_lower, doc = preprocess_when_text(text, nlp)

    # Short-circuit for procedure-only references
    if is_only_procedure_reference(text_lower, config):
        return build_procedure_only_result()

    # Check for primary vague terms (vague terms at start of control)
    primary_vague_terms = detect_primary_vague_terms(text_lower, config)
    if primary_vague_terms:
        return build_vague_term_result(primary_vague_terms[0])

    # Initialize candidates and flags
    when_candidates = []
    specific_timing_found = False
    detected_frequencies = []
    vague_terms_found = []

    # Detect all vague terms for reporting
    vague_terms_found = detect_vague_terms(text_lower, config)

    # 1. COMPLEX PATTERNS (highest precision, most specific)
    complex_candidates = detect_complex_patterns(text_lower, config)
    if complex_candidates:
        when_candidates.extend(complex_candidates)
        specific_timing_found = True

    # 2. EVENT TRIGGERS (especially important for corrective controls)
    event_candidates = detect_event_triggers(text_lower, config)
    if event_candidates:
        when_candidates.extend(event_candidates)
        specific_timing_found = True

    # 3. EXPLICIT FREQUENCIES
    explicit_candidates = detect_explicit_frequencies(text_lower, doc, config)
    if explicit_candidates:
        when_candidates.extend(explicit_candidates)
        specific_timing_found = True
        # Extract normalized frequencies
        for candidate in explicit_candidates:
            if 'frequency' in candidate and candidate['frequency'] not in detected_frequencies:
                detected_frequencies.append(candidate['frequency'])

    # 4. IMPLICIT TEMPORALS (last resort)
    if not specific_timing_found:
        implicit_candidates = detect_implicit_temporals(doc, text_lower, config)
        if implicit_candidates:
            when_candidates.extend(implicit_candidates)
            specific_timing_found = True

    # Add vague terms to candidates (for consistent return format)
    for vague in vague_terms_found:
        # Check if this is primary or secondary vague term
        # Primary if no specific timing or it comes before specific timing
        if not specific_timing_found:
            is_primary = True
        else:
            # Check if specific timing comes before this vague term
            specific_timing_before = False
            for specific in when_candidates:
                if not specific.get("is_vague", True) and specific.get("span", [0, 0])[0] < vague["span"][0]:
                    specific_timing_before = True
                    break
            is_primary = not specific_timing_before

        when_candidates.append({
            "text": vague["text"],
            "method": "vague_timing",
            "score": 0.1,  # Very low score for vague terms
            "span": vague["span"],
            "is_primary": is_primary,
            "is_vague": True,
            "context": get_context_window(text, vague["span"][0], vague["span"][1])
        })

    # Rank candidates to select top match
    top_match, final_score = rank_when_candidates(when_candidates)

    # Check for primary vague terms (vague terms that are the main timing indicator)
    primary_vague_term = any(c.get("is_primary", False) and c.get("is_vague", False) for c in when_candidates)

    # Calculate additional context-specific score adjustments using both control type and automation level
    if (control_type or automation_level) and top_match:
        final_score = apply_context_aware_scoring(final_score, top_match, control_type, automation_level)

    # Validate against metadata frequency if provided
    validation_result = build_frequency_validation(top_match, frequency_metadata, detected_frequencies, config)

    # Generate improvement suggestions
    improvement_suggestions = generate_when_suggestions(top_match, vague_terms_found,
                                                        final_score, specific_timing_found,
                                                        detected_frequencies, validation_result,
                                                        control_type, automation_level)

    # Prepare final result
    result = {
        "candidates": when_candidates,
        "top_match": top_match,
        "score": final_score,
        "extracted_keywords": [c["text"] for c in when_candidates],
        "multi_frequency_detected": len(detected_frequencies) > 1,
        "frequencies": detected_frequencies,
        "validation": validation_result,
        "vague_terms": vague_terms_found,
        "improvement_suggestions": improvement_suggestions,
        "specific_timing_found": specific_timing_found,
        "primary_vague_term": primary_vague_term
    }

    return result


def preprocess_when_text(text: str, nlp) -> Tuple[str, Any]:
    """
    Preprocess the text for WHEN detection

    Args:
        text: Raw text to preprocess
        nlp: spaCy NLP model

    Returns:
        Tuple of (lowercased text, spaCy doc)
    """
    text_lower = text.lower()
    doc = nlp(text_lower)
    return text_lower, doc


def is_only_procedure_reference(text_lower: str, config: Dict) -> bool:
    """
    Check if text only references a procedure/policy without specific timing

    Args:
        text_lower: Lowercased text
        config: Configuration dictionary

    Returns:
        True if only procedure reference, False otherwise
    """
    # Get configured procedure patterns or use default
    procedure_patterns = get_config_value(config, "procedure_patterns", [
        r'defined in (procedure|policy|document|standard)',
        r'outlined in (procedure|policy|document|standard)',
        r'described in (procedure|policy|document|standard)',
        r'according to (procedure|policy|document|standard)',
        r'per (procedure|policy|document|standard)',
        r'as per (procedure|policy|document|standard)',
        r'based on (procedure|policy|document|standard)',
        r'in accordance with (procedure|policy|document|standard)'
    ])

    # Get default frequency terms for checking
    frequency_terms = get_config_value(config, "frequency_terms", {
        "daily": ["daily", "each day", "every day"],
        "weekly": ["weekly", "each week", "every week"],
        "monthly": ["monthly", "each month", "every month"],
        "quarterly": ["quarterly", "each quarter", "every quarter"],
        "annually": ["annually", "yearly", "each year", "every year"]
    })

    # Flatten frequency terms
    flat_freq_terms = []
    for terms in frequency_terms.values():
        flat_freq_terms.extend(terms)

    # Check if text has procedure reference but no frequency terms
    has_procedure_ref = any(re.search(pattern, text_lower) for pattern in procedure_patterns)
    has_frequency = any(term in text_lower for term in flat_freq_terms +
                        ["within", "every", "each", "upon", "when", "after", "before"])

    return has_procedure_ref and not has_frequency


def detect_primary_vague_terms(text_lower: str, config: Dict) -> List[Dict[str, Any]]:
    """
    Detect vague timing terms at the start of the control description

    Args:
        text_lower: Lowercased text
        config: Configuration dictionary

    Returns:
        List of detected primary vague terms
    """
    # Get vague terms from config or use default
    vague_timing_terms = get_config_value(config, "vague_timing_terms", [
        "periodically", "regularly", "as needed", "when necessary", "as appropriate",
        "as required", "on a regular basis", "timely", "promptly", "from time to time",
        "when appropriate", "where appropriate", "if needed", "if necessary",
        "occasionally", "sometimes", "at times", "now and then", "intermittently",
        "at intervals", "frequently", "infrequently", "as applicable", "as deemed necessary",
        "when needed", "if appropriate", "where necessary", "as necessary"
    ])

    # Add special "may" handling
    if re.search(r'may\s+(?:vary|differ|change|be|not|need)', text_lower) or " may " in text_lower:
        # Exclude cases where "may" is clearly used as the month name
        if not re.search(r'\b(?:in|of|during|by|for|before|after)\s+may\b', text_lower) and not re.search(
                r'may\s+(?:\d{1,2}|\d{4})', text_lower):
            return [{
                "text": "may",
                "span": [text_lower.find("may"), text_lower.find("may") + 3],
                "is_primary": True,
                "suggested_replacement": "specific frequency (daily, weekly, monthly)"
            }]

    # Check for vague terms at start of description
    starts_with_vague = any(text_lower.strip().startswith(term) for term in vague_timing_terms)

    # Also check for other vague term patterns that shouldn't be missed
    vague_patterns = get_config_value(config, "vague_patterns", [
        r'^exceptions are (?:addressed|resolved|reviewed|handled) (when|as|if) (needed|appropriate|required)',
        r'^(as|when|if) (appropriate|needed|required)',
        r'^on an (as needed|when needed|if needed) basis',
        r'performed (periodically|occasionally|regularly|as necessary)'
    ])

    for pattern in vague_patterns:
        match = re.search(pattern, text_lower)
        if match:
            starts_with_vague = True
            # Return the matching vague term
            return [{
                "text": match.group(0),
                "span": [match.start(), match.end()],
                "is_primary": True,
                "suggested_replacement": "specific frequency (daily, weekly, monthly)"
            }]

    # If text starts with a vague term, find which one
    if starts_with_vague:
        for term in vague_timing_terms:
            if text_lower.strip().startswith(term):
                return [{
                    "text": term,
                    "span": [0, len(term)],
                    "is_primary": True,
                    "suggested_replacement": suggest_specific_alternative(term)
                }]

    return []


def detect_vague_terms(text_lower: str, config: Dict) -> List[Dict[str, Any]]:
    """
    Detect all vague timing terms in text

    Args:
        text_lower: Lowercased text
        config: Configuration dictionary

    Returns:
        List of detected vague terms
    """
    # Get vague terms from config or use default
    vague_timing_terms = get_config_value(config, "vague_timing_terms", [
        "periodically", "regularly", "as needed", "when necessary", "as appropriate",
        "as required", "on a regular basis", "timely", "promptly", "from time to time",
        "when appropriate", "where appropriate", "if needed", "if necessary",
        "occasionally", "sometimes", "at times", "now and then", "intermittently",
        "at intervals", "frequently", "infrequently", "as applicable", "as deemed necessary",
        "when needed", "if appropriate", "where necessary", "as necessary",
        "may vary", "may change", "may differ", "may be", "may not"
    ])

    vague_terms_found = []

    # Detect vague timing terms
    for term in vague_timing_terms:
        term_regex = r'\b' + re.escape(term) + r'\b'
        for match in re.finditer(term_regex, text_lower):
            start, end = match.span()
            vague_terms_found.append({
                "text": match.group(),
                "span": [start, end],
                "suggested_replacement": suggest_specific_alternative(term)
            })

    # Special check for "may"
    if re.search(r'may\s+(?:vary|differ|change|be|not|need)', text_lower) or " may " in text_lower:
        # Exclude cases where "may" is the month name
        if not re.search(r'\b(?:in|of|during|by|for|before|after)\s+may\b', text_lower) and not re.search(
                r'may\s+(?:\d{1,2}|\d{4})', text_lower):
            # Find the position of "may"
            may_pos = text_lower.find("may")
            if may_pos >= 0:
                vague_terms_found.append({
                    "text": "may",
                    "span": [may_pos, may_pos + 3],
                    "suggested_replacement": "specific frequency (daily, weekly, monthly)"
                })

    return vague_terms_found


def detect_complex_patterns(text_lower: str, config: Dict) -> List[Dict[str, Any]]:
    """
    Detect complex timing patterns using regex

    Args:
        text_lower: Lowercased text
        config: Configuration dictionary

    Returns:
        List of detected complex timing patterns
    """
    # Get vague terms first for exclusion
    vague_terms = detect_vague_terms(text_lower, config)

    # Get complex patterns from config or use default
    default_complex_patterns = [
        # Basic timeframe patterns
        (r'within\s+(\d+)\s+(day|week|month|business day|working day)s?', "timeframe", 0.95),
        (r'after\s+([\w\s]+?)\s+(is|are|has been|have been)', "sequential", 0.85),
        (r'prior\s+to\s+([\w\s]+)', "precondition", 0.85),
        (r'following\s+(completion|finalization|approval|review)\s+of', "sequential", 0.85),
        (r'upon\s+([\w\s]+)', "trigger", 0.85),
        (r'by\s+the\s+(\d+)(?:st|nd|rd|th)\s+(day|week|month)', "deadline", 0.95),

        # Enhanced staff transition patterns
        (
        r'(?:when|upon|after|at the time)\s+(?:an?\s+)?(?:employee|staff\s+member|individual|person)\s+(?:leaves|resigns|departs|terminates|separates)',
        "staff_transition", 0.9),
        (
        r'(?:when|upon|after|at the time of)\s+(?:an?\s+)?(?:employee|staff\s+member|individual|person)\'?s?\s+(?:departure|resignation|termination|separation)',
        "staff_transition", 0.9),
        (
        r'(?:when|upon|after)\s+(?:an?\s+)?(?:new\s+)?(?:employee|staff\s+member|individual|person)\s+(?:joins|starts|begins|commences)',
        "staff_transition", 0.9),

        # Enhanced system change patterns
        (
        r'(?:when|upon|after|before)\s+(?:system|software|application|platform)\s+(?:changes?|updates?|upgrades?|modifications?)',
        "system_change", 0.9),
        (
        r'(?:when|upon|after|before)\s+(?:a\s+)?(?:change|update|upgrade|modification)\s+(?:to|of|in)\s+(?:the\s+)?(?:system|software|application|platform)',
        "system_change", 0.9),

        # Enhanced business cycle patterns
        (
        r'(?:at|during|before|after)\s+(?:the end of|the beginning of|the start of|the close of)\s+(?:each|every|the)\s+(?:quarter|month|year|period)',
        "business_cycle", 0.9),

        # Financial cycle patterns
        (r'before\s+(?:each|every)\s+(?:financial|month[- ]end|quarter[- ]end|year[- ]end)\s+close',
         "financial_cycle", 0.95),
        (r'after\s+(?:each|every|the)\s+(?:financial|month[- ]end|quarter[- ]end|year[- ]end)\s+close',
         "financial_cycle", 0.95),
        (r'during\s+(?:the|each|every)\s+(?:financial|month[- ]end|quarter[- ]end|year[- ]end)\s+close',
         "financial_cycle", 0.95),
        (r'at\s+(?:the|each|every)\s+(?:financial|month[- ]end|quarter[- ]end|year[- ]end)\s+close',
         "financial_cycle", 0.95),
        (r'(?:before|after|during|at)\s+(?:the|each|every)\s+close\s+(?:process|period|cycle)', "financial_cycle", 0.9),

        # Day of week/month patterns
        (r'(?i)every\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)', "weekly_schedule", 0.95),
        (r'(?i)each\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)', "weekly_schedule", 0.95),
        (r'(?i)on\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?', "weekly_schedule", 0.95),

        # Day of month patterns
        (r'(?:on|by)\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)(?:\s+day)?(?:\s+of\s+(?:each|every|the)\s+month)?',
         "day_of_month", 0.95),
        (r'(?:on|by)\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)(?:\s+of\s+(?:each|every|the)\s+month)',
         "day_of_month", 0.95),

        # Time period patterns
        (r'(?:at|during|before|after)\s+(?:fiscal|calendar)\s+(?:year|quarter|month)[\s-]end', "fiscal_period", 0.9),
        (r'(?:at|during|before|after)\s+(?:month|quarter|year)[\s-]end\s+close', "close_period", 0.9),
        (r'(?:at|during)\s+(?:each|every|the)\s+closing\s+(?:cycle|period|process)', "close_period", 0.9),
        (r'(?:at|during)\s+(?:year|quarter|month)[\s-]end', "period_end", 0.9),
        (r'(?:fiscal|calendar)\s+(?:year|quarter|month)[\s-]end', "fiscal_period", 0.85),
        (r'(?:close|closing)\s+(?:cycle|period|process)', "close_period", 0.85),
    ]

    # Get complex patterns from config or use default
    complex_patterns = get_config_value(config, "complex_patterns", default_complex_patterns)

    # Track detected patterns
    complex_candidates = []

    # Find matches for complex patterns
    for pattern, pattern_type, score in complex_patterns:
        for match in re.finditer(pattern, text_lower):
            start, end = match.span()

            # Skip if this is part of a vague term
            if any(vague["span"][0] <= start and vague["span"][1] >= end for vague in vague_terms):
                continue

            # Get context for validation
            surrounding_text = get_context_window(text_lower, start, end)
            matched_text = match.group()

            # Special validations for certain pattern types
            if pattern_type == "sequential" and "following" in pattern:
                # Skip non-timing "following" usages
                if any(non_timing in matched_text for non_timing in
                       ["following lines", "following business", "following areas"]):
                    continue

            # For cycle-based patterns, validate they are actual timing indicators
            if "cycle" in pattern_type or "period" in pattern_type:
                if not is_valid_cycle_reference(matched_text, surrounding_text):
                    continue

            # For staff transition and system change patterns,
            # ensure they're associated with action verbs related to controls
            if pattern_type in ["staff_transition", "system_change"]:
                # Look for control-related action verbs in the surrounding context
                control_verbs = ["review", "approve", "check", "verify", "validate",
                                 "examine", "update", "assess", "ensure", "confirm",
                                 "monitor", "reconcile", "disable", "remove"]

                has_control_verb = any(verb in surrounding_text for verb in control_verbs)

                # Boost score if clearly tied to a control action
                if has_control_verb:
                    score = min(1.0, score + 0.05)

            complex_candidates.append({
                "text": matched_text,
                "method": f"complex_pattern_{pattern_type}",
                "score": score,
                "span": [start, end],
                "pattern_type": pattern_type,
                "is_primary": True,  # Complex patterns are usually significant
                "is_vague": False,
                "context": surrounding_text
            })

    return complex_candidates


def detect_event_triggers(text_lower: str, config: Dict) -> List[Dict[str, Any]]:
    """
    Detect event-based triggers in text

    Args:
        text_lower: Lowercased text
        config: Configuration dictionary

    Returns:
        List of detected event triggers
    """
    # Get vague terms first for exclusion
    vague_terms = detect_vague_terms(text_lower, config)

    # Get event trigger patterns from config or use default
    default_event_patterns = [
        # Business cycle patterns
        (
        r'(?:during|after|before|at)\s+(?:the|each|every)\s+(?:audit|review|assessment|reporting)\s+(?:cycle|period|process)',
        "business_cycle", 0.8),
        (
        r'(?:upon|after|following)\s+(?:completion|finalization)\s+of\s+(?:the|each|every)\s+(?:audit|review|assessment|reporting)',
        "business_cycle", 0.8),
        (r'(?:as\s+part\s+of|during)\s+(?:the|each|every)\s+(?:audit|review|assessment|reporting)\s+(?:cycle|process)',
         "business_cycle", 0.75),

        # Enhanced notification patterns
        (r'(?:upon|after|when|following)\s+(?:receipt|notification|alert|message)\s+(?:of|from|about)',
         "notification_trigger", 0.95),
        (r'(?:upon|after|when|following)\s+(?:being|getting)\s+(?:notified|alerted|informed)',
         "notification_trigger", 0.95),
        (r'(?:upon|after|when|following)\s+(?:system|application|tool|platform)\s+(?:alert|notification|message)',
         "system_notification", 0.95),

        # Enhanced detection patterns
        (r'(?:upon|after|when|following)\s+(?:identification|detection|discovery|finding)\s+of\s+(?:an?|the)',
         "detection_trigger", 0.95),
        (r'(?:immediately|promptly)\s+(?:upon|after|following)\s+(?:detection|identification|discovery)',
         "detection_trigger", 0.9),
        (r'(?:within\s+\d+\s+(?:day|hour|minute|week)s?)\s+of\s+(?:detection|identification|discovery)',
         "detection_trigger", 0.9),

        # Enhanced request patterns
        (
        r'(?:upon|when|following|after)\s+(?:receipt|receiving)\s+of\s+(?:a|the)\s+(?:request|ticket|inquiry|question)',
        "request_trigger", 0.95),
        (r'(?:when|if)\s+(?:a|an)\s+(?:request|ticket|inquiry|approval)\s+is\s+(?:submitted|received|sent|initiated)',
         "request_trigger", 0.95),

        # System event triggers
        (r'(?:after|when|upon)\s+(?:system|application|platform|database)\s+(?:update|upgrade|change|modification)',
         "system_event", 0.85),
        (r'(?:before|prior\s+to)\s+(?:system|application|platform|database)\s+(?:update|upgrade|change|modification)',
         "system_event", 0.85),

        # Enhanced threshold patterns
        (
        r'(?:when|if)\s+(?:the|a)\s+(?:amount|value|total|balance|sum|level|number|count)\s+(?:exceeds|reaches|falls\s+below|goes\s+(?:above|below))',
        "threshold_trigger", 0.95),
        (
        r'(?:when|if)\s+(?:the|a)\s+(?:threshold|limit|cap|ceiling|floor)\s+(?:is|has\s+been)\s+(?:reached|exceeded|breached|hit|met)',
        "threshold_trigger", 0.95),

        # Exception handling
        (
        r'when\s+(?:exceptions|anomalies|discrepancies|errors|issues|problems)\s+(?:are|have been)\s+(?:identified|detected|found|discovered)',
        "exception_handling", 0.9),
        (
        r'if\s+(?:exceptions|anomalies|discrepancies|errors|issues|problems)\s+(?:are|have been)\s+(?:identified|detected|found|discovered)',
        "exception_handling", 0.9),
        (
        r'(?:should|when|if)\s+(?:an?|the)\s+(?:exception|anomaly|discrepancy|error|issue|problem)\s+(?:occur|arise|be\s+(?:found|discovered|detected))',
        "exception_handling", 0.9),
    ]

    # Get event patterns from config or use default
    event_patterns = get_config_value(config, "event_patterns", default_event_patterns)

    # Track detected events
    event_candidates = []

    # Find matches for event patterns
    for pattern, pattern_type, score in event_patterns:
        for match in re.finditer(pattern, text_lower):
            start, end = match.span()

            # Skip if this is part of a vague term
            if any(vague["span"][0] <= start and vague["span"][1] >= end for vague in vague_terms):
                continue

            # Get context for validation
            surrounding_text = get_context_window(text_lower, start, end)
            matched_text = match.group()

            # Check if the event trigger is associated with a control action
            control_verbs = ["review", "approve", "check", "verify", "validate",
                             "examine", "update", "assess", "ensure", "confirm",
                             "monitor", "reconcile", "disable", "remove"]

            has_control_verb = any(verb in surrounding_text for verb in control_verbs)

            # Slightly increase score if the event trigger is clearly tied to a control action
            if has_control_verb:
                score = min(1.0, score + 0.05)

            # For threshold-based triggers, check for numeric values which make the trigger more specific
            if pattern_type == "threshold_trigger":
                has_numeric = bool(re.search(r'\d+', surrounding_text))
                if has_numeric:
                    score = min(1.0, score + 0.05)  # Boost score for specific thresholds

            event_candidates.append({
                "text": matched_text,
                "method": f"event_trigger_{pattern_type}",
                "score": score,
                "span": [start, end],
                "pattern_type": pattern_type,
                "is_primary": True,  # Event triggers are usually significant
                "is_vague": False,
                "context": surrounding_text
            })

    return event_candidates


def detect_explicit_frequencies(text_lower: str, doc, config: Dict) -> List[Dict[str, Any]]:
    """
    Detect explicit frequency terms in text

    Args:
        text_lower: Lowercased text
        doc: spaCy document
        config: Configuration dictionary

    Returns:
        List of detected explicit frequencies
    """
    # Get frequency terms from config or use default
    default_frequency_patterns = {
        "daily": ["daily", "each day", "every day", "on a daily basis", "day", "daily basis"],
        "weekly": ["weekly", "each week", "every week", "on a weekly basis", "week", "weekly basis"],
        "monthly": ["monthly", "each month", "every month", "on a monthly basis", "month", "monthly basis"],
        "quarterly": ["quarterly", "each quarter", "every quarter", "on a quarterly basis", "quarter"],
        "annually": ["annually", "yearly", "each year", "every year", "annual", "on an annual basis", "year"]
    }

    frequency_patterns = get_config_value(config, "frequency_terms", default_frequency_patterns)

    # IMPORTANT: Filter out "may" from any frequency patterns to avoid confusion with the auxiliary verb
    for freq, patterns in frequency_patterns.items():
        frequency_patterns[freq] = [p for p in patterns if p.lower() != "may"]

    # Get exclusion contexts
    context_exclusions = get_config_value(config, "context_exclusions", {
        "following": ["lines", "business", "areas", "items", "steps", "fields", "sections"],
        "time": ["real-time", "one time", "first time", "last time"],
        "period": ["reporting period", "accounting period", "time period"]
    })

    # Flatten the patterns for easier searching
    all_patterns = []
    pattern_to_frequency = {}
    for freq, patterns in frequency_patterns.items():
        for pattern in patterns:
            all_patterns.append(pattern)
            pattern_to_frequency[pattern] = freq

    # Track detected explicit frequencies
    explicit_candidates = []

    # Find vague terms first to exclude them
    vague_terms = detect_vague_terms(text_lower, config)

    # Match frequency patterns
    for pattern in all_patterns:
        pattern_regex = r'\b' + re.escape(pattern) + r'\b'
        for match in re.finditer(pattern_regex, text_lower):
            start, end = match.span()
            detected_freq = pattern_to_frequency.get(pattern, "unknown")

            # Skip if this is part of a vague term (e.g., "on a regular basis")
            if any(vague["span"][0] <= start and vague["span"][1] >= end for vague in vague_terms):
                continue

            # Check for exclusion contexts
            should_exclude = False
            if pattern in context_exclusions:
                exclusion_context = context_exclusions[pattern]
                context_after = text_lower[end:min(len(text_lower), end + 30)]

                # Check if any exclusion words appear right after the pattern
                if any(re.search(r'\b' + re.escape(exclude_word) + r'\b', context_after)
                       for exclude_word in exclusion_context):
                    should_exclude = True

            # Skip if it's an excluded context
            if should_exclude:
                continue

            # Determine if this is the main control frequency or a supporting activity
            surrounding_text = get_context_window(text_lower, start, end)
            is_primary = any(
                word in surrounding_text for word in ["review", "verify", "check", "ensure", "validate"])

            explicit_candidates.append({
                "text": match.group(),
                "method": "explicit_frequency",
                "score": 0.85,  # Slight lower score than complex patterns
                "span": [start, end],
                "frequency": detected_freq,
                "is_primary": is_primary,
                "is_vague": False,
                "context": surrounding_text
            })

    return explicit_candidates


def detect_implicit_temporals(doc, text_lower: str, config: Dict) -> List[Dict[str, Any]]:
    """
    Detect implicit temporal modifiers using spaCy's dependency parsing
    and enhanced pattern recognition

    Args:
        doc: spaCy document
        text_lower: Lowercased text
        config: Configuration dictionary

    Returns:
        List of detected implicit temporal modifiers
    """
    # Get temporal modifiers from config or use default
    temporal_indicators = get_config_value(config, "temporal_indicators",
                                           ["time", "moment", "instance", "point", "period",
                                            "interval", "date", "cycle", "phase", "stage"])

    # Get context exclusions
    context_exclusions = get_config_value(config, "context_exclusions", {
        "time": ["real-time", "one time", "first time", "last time"],
        "period": ["reporting period", "accounting period", "time period"]
    })

    # Flatten exclusions for easier checking
    flat_exclusions = []
    for ex_list in context_exclusions.values():
        flat_exclusions.extend(ex_list)

    # Get full text for context extraction
    full_text = doc.text

    # Track detected implicit temporals
    implicit_candidates = []

    # METHOD 1: Look for temporal modifiers through dependency parsing
    for token in doc:
        if token.dep_ in ["npadvmod", "advmod", "dobj", "pobj"] and token.head.pos_ == "VERB":
            # Check if it's likely a time-related modifier
            if any(time_word in token.text.lower() for time_word in temporal_indicators):
                # Skip if this appears to be an exclusion
                context_window = get_context_window(full_text, token.idx, token.idx + len(token.text))

                if any(excluded in context_window.lower() for excluded in flat_exclusions):
                    continue

                implicit_candidates.append({
                    "text": token.text,
                    "method": "implicit_temporal_modifier",
                    "score": 0.6,
                    "span": [token.idx, token.idx + len(token.text)],
                    "is_primary": False,
                    "is_vague": False,
                    "context": context_window
                })

    # METHOD 2: Enhanced patterns for implicit time references
    implicit_patterns = [
        # Activity completion patterns
        (
        r'(?:once|after)\s+(?:the|this)\s+(?:activity|task|process|action|step|review|approval)\s+(?:is|has been)\s+(?:completed|finished|done)',
        "completion_trigger", 0.7),

        # Condition fulfillment patterns
        (
        r'(?:once|when|after)\s+(?:the|these|all|both)\s+(?:condition|criteria|requirement)s?\s+(?:is|are|have been)\s+(?:met|satisfied|fulfilled)',
        "condition_trigger", 0.7),

        # Status change patterns
        (r'(?:when|once)\s+(?:the|a)\s+(?:status|state)\s+(?:changes|is changed|becomes|reaches)',
         "status_trigger", 0.7),

        # Initial setup patterns
        (r'(?:during|at|when)\s+(?:initial|first|new)\s+(?:setup|configuration|implementation|installation)',
         "initial_setup", 0.7)
    ]

    # Find matches for implicit temporal patterns
    for pattern, pattern_type, score in implicit_patterns:
        for match in re.finditer(pattern, text_lower):
            start, end = match.span()

            # Get context for validation
            surrounding_text = get_context_window(text_lower, start, end)
            matched_text = match.group()

            # Check if associated with a control verb
            control_verbs = ["review", "approve", "check", "verify", "validate",
                             "examine", "update", "assess", "ensure", "confirm",
                             "monitor", "reconcile"]

            has_control_verb = any(verb in surrounding_text for verb in control_verbs)

            # Adjust score based on control verb presence
            if has_control_verb:
                score = min(1.0, score + 0.1)

            implicit_candidates.append({
                "text": matched_text,
                "method": f"implicit_{pattern_type}",
                "score": score,
                "span": [start, end],
                "pattern_type": pattern_type,
                "is_primary": has_control_verb,  # Primary if associated with control verb
                "is_vague": False,
                "context": surrounding_text
            })

    return implicit_candidates


def rank_when_candidates(when_candidates: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Rank candidates and select the best match with improved prioritization logic

    Args:
        when_candidates: List of candidates

    Returns:
        Tuple of (top match, score)
    """
    if not when_candidates:
        return None, 0

    # First filter out vague terms for primary selection
    specific_candidates = [c for c in when_candidates if not c.get("is_vague", True)]

    # If explicit candidates exist, use them
    if specific_candidates:
        # Score adjustment for certain methods to better prioritize certain timing patterns
        for candidate in specific_candidates:
            method = candidate.get("method", "")

            # Prioritize detection triggers for corrective controls
            if "detection_trigger" in method or "exception_handling" in method:
                candidate["score"] = min(1.0, candidate.get("score", 0) * 1.1)

            # Prioritize threshold triggers as they're usually important
            if "threshold_trigger" in method:
                candidate["score"] = min(1.0, candidate.get("score", 0) * 1.08)

            # Give slightly higher weight to staff transition patterns
            if "staff_transition" in method:
                candidate["score"] = min(1.0, candidate.get("score", 0) * 1.05)

            # Explicit frequency patterns are good but slightly behind complex patterns
            if method == "explicit_frequency":
                # No adjustment needed - already well-scored
                pass

            # Give higher priority to patterns with time specificity
            if "timeframe" in method or "deadline" in method or "day_of_month" in method:
                candidate["score"] = min(1.0, candidate.get("score", 0) * 1.05)

        # Sort by score, then by position (earlier in text)
        specific_candidates.sort(key=lambda x: (-x.get("score", 0), x.get("span", [0, 0])[0]))
        top_match = specific_candidates[0]

        # Use the enhanced score from the candidate that includes method-specific boosts
        base_score = top_match.get("score", 0)

        # For event-based timing, ensure score is at least 0.65
        # This addresses the issue where event-based timing patterns score lower than they should
        if (("event" in top_match.get("method", "") or
             "trigger" in top_match.get("method", "") or
             "exception" in top_match.get("method", "")) and
                base_score < 0.65):
            base_score = 0.65
    else:
        # No specific timing - use highest scoring vague term
        vague_candidates = [c for c in when_candidates if c.get("is_vague", False)]
        if not vague_candidates:
            return None, 0  # No candidates at all

        vague_candidates.sort(key=lambda x: (-x.get("score", 0), x.get("span", [0, 0])[0]))
        top_match = vague_candidates[0]

        # Base score for vague terms depends on context
        # Instead of always giving zero, we check if other elements suggest
        # the timing is implied elsewhere in the control description
        vague_text = top_match.get("text", "").lower()

        # Get context from the candidate if available
        context = top_match.get("context", "").lower()

        # "As needed" with clear condition might deserve a minimal score
        if vague_text in ["as needed", "when needed", "if needed"] and "based on" in context:
            base_score = 0.1  # Small but not zero
        # Condition-based vague terms might deserve a minimal score
        elif vague_text in ["when necessary", "as required", "if required"] and "if" in context:
            base_score = 0.1  # Small but not zero
        else:
            base_score = 0  # Zero score for vague term as primary

    return top_match, base_score


def apply_context_aware_scoring(score: float, top_match: Dict[str, Any],
                                control_type: Optional[Union[str, List]] = None,
                                automation_level: Optional[Union[str, List]] = None) -> float:
    """
    Apply context-aware scoring based on control type and automation level

    Args:
        score: Base score
        top_match: Top candidate match
        control_type: Control type (preventive, detective, corrective)
        automation_level: Automation level (manual, automated, hybrid)

    Returns:
        Adjusted score
    """
    if not score or not top_match:
        return score

    # Control type adjustments
    if control_type:
        # Handle control_type that might be a list
        control_type_lower = ""
        if isinstance(control_type, list):
            if control_type and len(control_type) > 0:
                control_type_lower = str(control_type[0]).lower()
        else:
            control_type_lower = str(control_type).lower()

        # Detective controls need very clear timing
        if control_type_lower == "detective" and score > 0:
            # Check for precision
            if top_match.get("score", 0) < 0.8 or top_match.get("method", "").startswith("implicit"):
                score *= 0.8  # Reduce score if timing isn't very specific

        # Preventive controls might have implicit timing
        elif control_type_lower == "preventive" and score > 0:
            # Preventive controls can work with less explicit timing
            score = min(1.0, score * 1.2)  # Slightly boost score

        # Corrective controls typically need event-based triggers
        elif control_type_lower == "corrective" and score > 0:
            if "event" in top_match.get("method", "") or "trigger" in top_match.get("method", ""):
                score = min(1.0, score * 1.15)  # Boost for appropriate timing triggers
            else:
                score *= 0.9  # Slight penalty if not event-triggered

    # Automation level adjustments
    if automation_level:
        # Handle automation_level that might be a list
        automation_lower = ""
        if isinstance(automation_level, list):
            if automation_level and len(automation_level) > 0:
                automation_lower = str(automation_level[0]).lower()
        else:
            automation_lower = str(automation_level).lower()

        # Automated controls need very specific timing
        if automation_lower == "automated" and score > 0:
            # Precise timing (strong for scheduling/job automation)
            if "explicit_frequency" in top_match.get("method", ""):
                score = min(1.0, score * 1.15)  # Stronger boost for clear frequency
            elif "complex_pattern" in top_match.get("method", ""):
                score = min(1.0, score * 1.1)  # Boost for precise timing
            elif "implicit" in top_match.get("method", ""):
                score *= 0.7  # Bigger penalty for vague timing in automated controls

        # Hybrid controls need balance
        elif automation_lower == "hybrid" and score > 0:
            if "explicit_frequency" in top_match.get("method", ""):
                score = min(1.0, score * 1.05)  # Small boost

        # Manual controls have more flexibility in timing
        elif automation_lower == "manual" and score > 0:
            # Less penalty for implicit timing in manual controls
            if "implicit" in top_match.get("method", ""):
                score *= 0.9  # Smaller penalty than for automated

    return score


def build_frequency_validation(top_match: Optional[Dict[str, Any]], frequency_metadata: Optional[str],
                               detected_frequencies: List[str], config: Dict) -> Dict[str, Any]:
    """
    Validate detected frequency against metadata

    Args:
        top_match: Top candidate match
        frequency_metadata: Declared frequency metadata
        detected_frequencies: List of detected normalized frequencies
        config: Configuration dictionary

    Returns:
        Validation result dictionary
    """
    if not frequency_metadata:
        return {"is_valid": True, "message": "No frequency metadata provided for validation"}

    # Normalize metadata
    normalized_metadata = frequency_metadata.lower().strip()

    # Get frequency terms from config
    frequency_terms = get_config_value(config, "frequency_terms", {
        "daily": ["daily", "each day", "every day"],
        "weekly": ["weekly", "each week", "every week"],
        "monthly": ["monthly", "each month", "every month"],
        "quarterly": ["quarterly", "each quarter", "every quarter"],
        "annually": ["annually", "yearly", "each year", "every year"]
    })

    # Check if any detected frequency matches metadata
    metadata_match = False
    matching_frequency = None

    for freq in detected_frequencies:
        # Direct match
        if freq == normalized_metadata:
            metadata_match = True
            matching_frequency = freq
            break

        # Term-based match
        if freq in frequency_terms:
            if any(pattern.lower() in normalized_metadata for pattern in frequency_terms.get(freq, [])):
                metadata_match = True
                matching_frequency = freq
                break

    if metadata_match:
        return {
            "is_valid": True,
            "message": f"Frequency in description matches metadata ({normalized_metadata})",
            "matched_frequency": matching_frequency
        }
    else:
        return {
            "is_valid": False,
            "message": f"Frequency in description ({', '.join(detected_frequencies) if detected_frequencies else 'none'}) " +
                       f"does not match metadata ({normalized_metadata})",
            "detected_frequencies": detected_frequencies
        }


def generate_when_suggestions(top_match: Optional[Dict[str, Any]], vague_terms: List[Dict[str, Any]],
                              score: float, specific_timing_found: bool,
                              detected_frequencies: List[str], validation_result: Dict[str, Any],
                              control_type: Optional[Union[str, List]] = None,
                              automation_level: Optional[Union[str, List]] = None) -> List[str]:
    """
    Generate improvement suggestions based on analysis

    Args:
        top_match: Top candidate match
        vague_terms: List of detected vague terms
        score: Calculated score
        specific_timing_found: Whether specific timing was found
        detected_frequencies: List of detected frequencies
        validation_result: Frequency validation result
        control_type: Control type (preventive, detective, corrective)
        automation_level: Automation level (manual, automated, hybrid)

    Returns:
        List of improvement suggestions
    """
    suggestions = []

    # Check for multiple frequencies
    if len(detected_frequencies) > 1:
        suggestions.append(
            "Multiple frequencies detected. Consider whether this is describing a process rather than a single control."
        )

    # Check for missing timing
    if not specific_timing_found:
        suggestions.append(
            "No specific timing information detected. Add specific frequency (daily, weekly, monthly) or timing (within X days)."
        )

    # Check for vague terms
    for vague_term in vague_terms:
        suggestions.append(
            f"Replace vague timing term '{vague_term['text']}' with {vague_term['suggested_replacement']}."
        )

    # Check validation result
    if validation_result and not validation_result.get("is_valid", True):
        # Extract frequency metadata from the message if available
        message = validation_result.get("message", "")
        metadata_parts = message.split("metadata (")
        if len(metadata_parts) > 1:
            frequency_metadata = metadata_parts[1].split(")")[0]
            if frequency_metadata:
                suggestions.append(
                    f"Align the frequency in the description with the declared frequency ({frequency_metadata})"
                )

    # Additional suggestions based on score, control type, and automation level
    if score > 0 and score < 0.5 and specific_timing_found:
        suggestions.append(
            "Enhance timing clarity by specifying exactly when the control occurs."
        )

    # Control type specific suggestions
    if control_type:
        # Handle control_type that might be a list
        control_type_lower = ""
        if isinstance(control_type, list):
            if control_type and len(control_type) > 0:
                control_type_lower = str(control_type[0]).lower()
        else:
            control_type_lower = str(control_type).lower()

        method = top_match.get("method", "") if top_match else ""

        if control_type_lower == "detective" and not specific_timing_found:
            suggestions.append(
                "Detective controls require precise timing. Add specific frequency or schedule."
            )
        elif control_type_lower == "detective" and "implicit" in method:
            suggestions.append(
                "Detective controls need explicit timing. Replace implicit timing with specific schedule."
            )
        elif control_type_lower == "corrective" and not any(
                term in method for term in ["event", "trigger", "exception"]):
            suggestions.append(
                "Consider adding event-based triggers for this corrective control (e.g., 'when errors are detected')."
            )

    # Automation level specific suggestions
    if automation_level:
        # Handle automation_level that might be a list
        automation_lower = ""
        if isinstance(automation_level, list):
            if automation_level and len(automation_level) > 0:
                automation_lower = str(automation_level[0]).lower()
        else:
            automation_lower = str(automation_level).lower()

        method = top_match.get("method", "") if top_match else ""

        if automation_lower == "automated" and not specific_timing_found:
            suggestions.append(
                "Automated controls require precise timing for scheduling. Add specific frequency or schedule."
            )
        elif automation_lower == "automated" and "implicit" in method:
            suggestions.append(
                "Automated controls need explicit timing parameters. Replace with specific schedule or trigger."
            )
        elif automation_lower == "hybrid" and not specific_timing_found:
            suggestions.append(
                "Hybrid controls need clear timing for both automated and manual components."
            )

    return suggestions


def build_empty_result(message: str) -> Dict[str, Any]:
    """
    Build default result for empty input

    Args:
        message: Error message

    Returns:
        Default result dictionary
    """
    return {
        "candidates": [],
        "top_match": None,
        "score": 0,
        "extracted_keywords": [],
        "multi_frequency_detected": False,
        "frequencies": [],
        "validation": {"is_valid": False, "message": message},
        "vague_terms": [],
        "improvement_suggestions": ["No text provided to analyze"],
        "specific_timing_found": False,
        "primary_vague_term": False
    }


def build_procedure_only_result() -> Dict[str, Any]:
    """
    Build result for procedure-only references

    Returns:
        Result dictionary for procedure-only case
    """
    return {
        "candidates": [],
        "top_match": None,
        "score": 0,  # Zero score for procedure-only references
        "extracted_keywords": [],
        "multi_frequency_detected": False,
        "frequencies": [],
        "validation": {"is_valid": False, "message": "Only procedure reference without timing"},
        "vague_terms": [],
        "improvement_suggestions": [
            "Add specific frequency (daily, weekly, monthly) instead of just referencing a procedure."
        ],
        "specific_timing_found": False,
        "primary_vague_term": False
    }


def build_vague_term_result(primary_vague: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build result when primary vague term is detected

    Args:
        primary_vague: Primary vague term dictionary

    Returns:
        Result dictionary for vague term case
    """
    return {
        "candidates": [{
            "text": primary_vague["text"],
            "method": "vague_timing",
            "score": 0.1,
            "is_vague": True,
            "is_primary": True
        }],
        "top_match": {
            "text": primary_vague["text"],
            "method": "vague_timing",
            "score": 0.1,
            "is_vague": True,
            "is_primary": True
        },
        "score": 0,  # Zero score to ensure missing element
        "extracted_keywords": [primary_vague["text"]],
        "multi_frequency_detected": False,
        "frequencies": [],
        "validation": {"is_valid": False, "message": "Vague timing detected"},
        "vague_terms": [{
            "text": primary_vague["text"],
            "is_primary": True,
            "suggested_replacement": primary_vague.get("suggested_replacement",
                                                       "specific frequency (daily, weekly, monthly)")
        }],
        "improvement_suggestions": [
            f"Replace vague term '{primary_vague['text']}' with specific frequency (daily, weekly, monthly)."
        ],
        "specific_timing_found": False,
        "primary_vague_term": True
    }


def is_valid_cycle_reference(text: str, context: str) -> bool:
    """
    Validate that a cycle reference is actually indicating timing and not just process

    Args:
        text: Cycle reference text
        context: Surrounding context

    Returns:
        True if valid timing reference, False otherwise
    """
    # Check if the cycle reference includes timing words
    timing_indicators = [
        "every", "each", "before", "after", "during", "upon", "following",
        "prior to", "at", "when", "within"
    ]

    if any(indicator in text.lower() for indicator in timing_indicators):
        return True

    # Check the surrounding context for timing indicators
    context_lower = context.lower()
    if any(indicator in context_lower for indicator in timing_indicators):
        return True

    # Check for verbs that typically indicate periodic actions
    action_verbs = [
        "perform", "conduct", "execute", "run", "complete", "do", "implement",
        "carry out", "undertake"
    ]

    if any(verb in context_lower for verb in action_verbs):
        return True

    return False


def get_context_window(text: str, start: int, end: int, window_size: int = 30) -> str:
    """
    Get a context window around a match

    Args:
        text: Full text
        start: Start position of match
        end: End position of match
        window_size: Size of context window

    Returns:
        Context window text
    """
    return text[max(0, start - window_size):min(len(text), end + window_size)]


def suggest_specific_alternative(vague_term: str) -> str:
    """
    Suggest specific alternatives to vague timing terms

    Args:
        vague_term: Vague term to get suggestion for

    Returns:
        Suggestion string
    """
    suggestions = {
        "periodically": "specific frequency (daily, weekly, monthly)",
        "regularly": "specific frequency (daily, weekly, monthly)",
        "as needed": "triggering conditions (e.g., 'when discrepancies are found')",
        "when necessary": "specific conditions that necessitate the control",
        "as appropriate": "specific criteria for appropriateness",
        "as required": "specific requirements that trigger the control",
        "on a regular basis": "specific frequency (daily, weekly, monthly)",
        "timely": "specific timeframe (e.g., 'within 3 business days')",
        "promptly": "specific timeframe (e.g., 'within 24 hours')",
        "from time to time": "specific frequency or conditions",
        "when appropriate": "specific criteria for when the control is appropriate",
        "where appropriate": "specific criteria for where the control is appropriate",
        "if needed": "specific conditions that trigger the control",
        "if necessary": "specific conditions that trigger the control",
        "occasionally": "specific frequency (monthly, quarterly)",
        "sometimes": "specific frequency or triggering conditions",
        "at times": "specific timing or frequency",
        "now and then": "specific frequency",
        "intermittently": "specific schedule or pattern",
        "at intervals": "specific interval period (e.g., 'every 14 days')",
        "frequently": "specific frequency (daily, twice weekly)",
        "infrequently": "specific frequency (quarterly, annually)",
        "as applicable": "specific criteria for applicability",
        "as deemed necessary": "specific criteria for necessity",
        "may vary": "specific frequency with defined parameters",
        "may change": "specific frequency with defined parameters",
        "may differ": "specific frequency with defined parameters",
        "may": "specific frequency (daily, weekly, monthly)"
    }

    return suggestions.get(vague_term.lower(), "a specific timeframe or frequency")


def get_config_value(config: Dict, key: str, default: Any) -> Any:
    """
    Safely get a value from configuration with default fallback

    Args:
        config: Configuration dictionary
        key: Configuration key
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    if not config:
        return default

    return config.get(key, default)