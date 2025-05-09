"""
Enhanced WHEN detection module with optimized performance.
Identifies and analyzes timing information in control descriptions.
"""

import re
from typing import Dict, List, Any, Optional

# Pre-compile frequent regex patterns for performance
# Common frequency patterns
DAILY_PATTERN = re.compile(r'\b(daily|each\s+day|every\s+day|on\s+a\s+daily\s+basis|day\b|daily\s+basis)\b',
                           re.IGNORECASE)
WEEKLY_PATTERN = re.compile(r'\b(weekly|each\s+week|every\s+week|on\s+a\s+weekly\s+basis|week\b|weekly\s+basis)\b',
                            re.IGNORECASE)
MONTHLY_PATTERN = re.compile(
    r'\b(monthly|each\s+month|every\s+month|on\s+a\s+monthly\s+basis|month\b|monthly\s+basis)\b', re.IGNORECASE)
QUARTERLY_PATTERN = re.compile(
    r'\b(quarterly|each\s+quarter|every\s+quarter|on\s+a\s+quarterly\s+basis|quarter\b|once\s+per\s+quarter|each\s+fiscal\s+quarter|every\s+three\s+months)\b',
    re.IGNORECASE)
ANNUALLY_PATTERN = re.compile(
    r'\b(annually|yearly|each\s+year|every\s+year|annual\b|on\s+an\s+annual\s+basis|year\b)\b', re.IGNORECASE)
ADHOC_PATTERN = re.compile(r'\b(ad[\s-]hoc|on\s+an\s+ad[\s-]hoc\s+basis)\b', re.IGNORECASE)

# Weekday patterns
WEEKDAY_PATTERN = re.compile(r'\b(every|each|on)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b',
                             re.IGNORECASE)

# Period end patterns (consolidated from multiple similar patterns)
PERIOD_END_PATTERN = re.compile(
    r'\b(at|during|before|after)\s+(the\s+)?(fiscal|calendar)?\s*(year|quarter|month)[\s-]end(\s+close)?\b',
    re.IGNORECASE)
CLOSE_PERIOD_PATTERN = re.compile(r'\b(at|during)\s+(each|every|the)\s+closing\s+(cycle|period|process)\b',
                                  re.IGNORECASE)

# Timeline patterns (within, after, before, etc.)
TIMELINE_PATTERN = re.compile(r'\b(within|after|before|prior\s+to|following|upon|by)\s+', re.IGNORECASE)

# Vague terms consolidated
VAGUE_TERMS_PATTERN = re.compile(
    r'\b(as\s+needed|when\s+needed|if\s+needed|as\s+appropriate|when\s+appropriate|'
    r'if\s+appropriate|as\s+required|when\s+required|if\s+required|periodically|'
    r'occasionally|from\s+time\s+to\s+time|regularly|'
    r'may\s+vary|may\s+change|may\s+differ|may\s+be|may\s+not|'
    r'on\s+demand|as\s+and\s+when|upon\s+request|when\s+requested|non[\s-]scheduled)\b',
    re.IGNORECASE
)

# Problematic may pattern (but not month of May)
PROBLEMATIC_MAY_PATTERN = re.compile(r'\bmay\s+(?:vary|differ|change|be|not|need)\b', re.IGNORECASE)

# Consolidated business cycle patterns
BUSINESS_CYCLE_PATTERN = re.compile(
    r'\b(during|after|before|at|upon|following|as\s+part\s+of)\s+(the\s+)?(each|every)?\s*'
    r'(audit|review|assessment|reporting|close|closing)\s+(cycle|period|process)\b',
    re.IGNORECASE
)

# Procedure reference pattern
PROCEDURE_REFERENCE_PATTERN = re.compile(
    r'\b(defined|outlined|described|according|per|as\s+per|based|in\s+accordance)\s+'
    r'(in|on|with|to)\s+(procedure|policy|document|standard)\b',
    re.IGNORECASE
)

# Event trigger patterns
EVENT_TRIGGER_PATTERN = re.compile(
    r'\b(upon|after|when|following|immediately|promptly)\s+'
    r'(receipt|notification|identification|detection|discovery|system|application|platform|database)\b',
    re.IGNORECASE
)

# Cached suggestions for vague terms
VAGUE_TERM_SUGGESTIONS = {
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
    "may": "specific frequency (daily, weekly, monthly)",
    "ad-hoc": "specific triggering conditions for the ad-hoc review",
    "ad hoc": "specific triggering conditions for the ad-hoc review",
    "on an ad-hoc basis": "specific triggering conditions for the ad-hoc review",
    "on demand": "specific triggering conditions",
    "as and when": "specific triggering conditions",
    "upon request": "defined request process with frequency",
    "when requested": "defined request process with frequency",
    "non-scheduled": "regular schedule (daily, weekly, monthly)"
}


def _suggest_specific_alternative(vague_term):
    """Suggest specific alternatives to vague timing terms"""
    # Use the cached suggestions dictionary defined at module level
    return VAGUE_TERM_SUGGESTIONS.get(vague_term.lower(), "a specific timeframe or frequency")


def enhance_when_detection(text: str, nlp, control_type: Optional[str] = None,
                           existing_keywords: Optional[List[str]] = None,
                           frequency_metadata: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhanced WHEN detection with improved handling of complex timing patterns,
    multi-frequency detection, and special handling for semi-vague terms like ad-hoc.
    Optimized for better performance.

    Args:
        text: Control description text
        nlp: Loaded spaCy model
        control_type: Optional control type (preventive, detective, etc.) for context-aware scoring
        existing_keywords: Optional dictionary or list of timing keywords to supplement defaults
        frequency_metadata: Optional declared frequency from metadata for validation

    Returns:
        Dictionary with detection results, scores, and improvement suggestions
    """
    if not text or text.strip() == '':
        return {
            "candidates": [],
            "top_match": None,
            "score": 0,
            "extracted_keywords": [],
            "multi_frequency_detected": False,
            "frequencies": [],
            "validation": {"is_valid": False, "message": "No text provided"},
            "vague_terms": [],
            "improvement_suggestions": []
        }

    try:
        # Normalize input text for case-insensitive matching
        text_lower = text.lower()

        # Initialize result containers
        when_candidates = []
        detected_frequencies = []
        vague_terms_found = []
        specific_timing_found = False
        improvement_suggestions = []

        # ---------- PHASE 1: Critical early checks for common patterns ----------

        # Handle problematic "may" (excluding month of May)
        if " may " in text_lower and not re.search(r'\b(?:in|of|during|by|for|before|after)\s+may\b',
                                                   text_lower) and not re.search(
                r'may\s+(?:\d{1,2}|\d{4})', text_lower):
            if PROBLEMATIC_MAY_PATTERN.search(text_lower):
                vague_match = "may"
                return {
                    "candidates": [{
                        "text": "may",
                        "method": "vague_timing",
                        "score": 0.1,
                        "is_vague": True,
                        "is_primary": True
                    }],
                    "top_match": {
                        "text": "may",
                        "method": "vague_timing",
                        "score": 0.1,
                        "is_vague": True,
                        "is_primary": True
                    },
                    "score": 0,
                    "extracted_keywords": ["may"],
                    "multi_frequency_detected": False,
                    "frequencies": [],
                    "validation": {"is_valid": False, "message": "Vague timing detected"},
                    "vague_terms": [{
                        "text": "may",
                        "is_primary": True,
                        "suggested_replacement": "specific frequency (daily, weekly, monthly)"
                    }],
                    "improvement_suggestions": [
                        "Replace vague term 'may' with specific frequency (daily, weekly, monthly)."
                    ],
                    "specific_timing_found": False,
                    "primary_vague_term": True
                }

        # Check if text starts with a vague term (except ad-hoc)
        vague_start_match = VAGUE_TERMS_PATTERN.match(text_lower)
        if vague_start_match:
            vague_match = vague_start_match.group(0)
            return {
                "candidates": [{
                    "text": vague_match,
                    "method": "vague_timing",
                    "score": 0.1,
                    "is_vague": True,
                    "is_primary": True
                }],
                "top_match": {
                    "text": vague_match,
                    "method": "vague_timing",
                    "score": 0.1,
                    "is_vague": True,
                    "is_primary": True
                },
                "score": 0,
                "extracted_keywords": [vague_match],
                "multi_frequency_detected": False,
                "frequencies": [],
                "validation": {"is_valid": False, "message": "Vague timing detected"},
                "vague_terms": [{
                    "text": vague_match,
                    "is_primary": True,
                    "suggested_replacement": _suggest_specific_alternative(vague_match)
                }],
                "improvement_suggestions": [
                    f"Replace vague timing term '{vague_match}' with specific frequency (daily, weekly, monthly)."
                ],
                "specific_timing_found": False,
                "primary_vague_term": True
            }

        # Check for procedure reference without timing
        if PROCEDURE_REFERENCE_PATTERN.search(text_lower) and not any(
                pattern.search(text_lower) for pattern in [
                    DAILY_PATTERN, WEEKLY_PATTERN, MONTHLY_PATTERN,
                    QUARTERLY_PATTERN, ANNUALLY_PATTERN, ADHOC_PATTERN
                ]):
            return {
                "candidates": [],
                "top_match": None,
                "score": 0,
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

        # ---------- PHASE 2: Find and record all vague terms ----------
        for match in VAGUE_TERMS_PATTERN.finditer(text_lower):
            vague_terms_found.append({
                "text": match.group(),
                "span": [match.start(), match.end()],
                "suggested_replacement": _suggest_specific_alternative(match.group())
            })

        # ---------- PHASE 3: Special handling for ad-hoc ----------
        # Ad-hoc is a valid frequency but considered semi-vague (could be improved)
        ad_hoc_match = ADHOC_PATTERN.search(text_lower)
        if ad_hoc_match:
            start, end = ad_hoc_match.span()
            surrounding_text = text[max(0, start - 30):min(len(text), end + 30)]

            # Found an ad-hoc timing expression - consider it specific but not ideal
            specific_timing_found = True

            # Add to detected frequencies
            if "ad-hoc" not in detected_frequencies:
                detected_frequencies.append("ad-hoc")

            when_candidates.append({
                "text": ad_hoc_match.group(),
                "method": "adhoc_frequency",
                "score": 0.7,  # Lower score than standard frequencies but still passing
                "span": [start, end],
                "frequency": "ad-hoc",
                "is_primary": True,
                "is_vague": False,  # Not vague in the system, but not ideal
                "is_semi_vague": True,  # New property to indicate it could be improved
                "context": surrounding_text
            })

            # Add improvement suggestion
            improvement_suggestions.append(
                "While 'ad-hoc' is an allowed frequency, the control would be stronger if it specified what triggers the ad-hoc review."
            )

        # ---------- PHASE 4: Standard frequency patterns ----------
        frequency_checks = [
            (DAILY_PATTERN, "daily"),
            (WEEKLY_PATTERN, "weekly"),
            (MONTHLY_PATTERN, "monthly"),
            (QUARTERLY_PATTERN, "quarterly"),
            (ANNUALLY_PATTERN, "annually"),
        ]

        for pattern, freq_name in frequency_checks:
            for match in pattern.finditer(text_lower):
                start, end = match.span()

                # Skip if this is part of a vague term
                if any(vague["span"][0] <= start and vague["span"][1] >= end for vague in vague_terms_found):
                    continue

                # Found a specific timing expression
                specific_timing_found = True

                # Determine if this is the main control frequency
                surrounding_text = text[max(0, start - 30):min(len(text), end + 30)]
                is_primary = any(
                    word in surrounding_text.lower() for word in
                    ["review", "verify", "check", "ensure", "validate"]
                )

                if freq_name not in detected_frequencies:
                    detected_frequencies.append(freq_name)

                when_candidates.append({
                    "text": match.group(),
                    "method": "explicit_frequency",
                    "score": 0.9,
                    "span": [start, end],
                    "frequency": freq_name,
                    "is_primary": is_primary,
                    "is_vague": False,
                    "context": surrounding_text
                })

        # ---------- PHASE 5: Check for weekday patterns ----------
        if not when_candidates or (len(when_candidates) == 1 and "ad-hoc" in detected_frequencies):
            for match in WEEKDAY_PATTERN.finditer(text_lower):
                start, end = match.span()

                # Skip if this is part of a vague term
                if any(vague["span"][0] <= start and vague["span"][1] >= end for vague in vague_terms_found):
                    continue

                # Found a specific timing expression
                specific_timing_found = True

                # Add to detected frequencies
                if "weekly" not in detected_frequencies:
                    detected_frequencies.append("weekly")

                when_candidates.append({
                    "text": match.group(),
                    "method": "weekly_schedule",
                    "score": 0.9,
                    "span": [start, end],
                    "frequency": "weekly",
                    "is_primary": True,  # Usually primary
                    "is_vague": False,
                    "context": text[max(0, start - 30):min(len(text), end + 30)]
                })

        # ---------- PHASE 6: Check for period end patterns ----------
        if not when_candidates or (len(when_candidates) == 1 and "ad-hoc" in detected_frequencies):
            # Check for period end patterns
            for match in PERIOD_END_PATTERN.finditer(text_lower):
                start, end = match.span()

                # Skip if this is part of a vague term
                if any(vague["span"][0] <= start and vague["span"][1] >= end for vague in vague_terms_found):
                    continue

                # Found a specific timing expression
                specific_timing_found = True

                # Determine frequency based on period
                period_type = re.search(r'(year|quarter|month)', match.group())
                period_freq = "annually" if period_type and "year" in period_type.group() else \
                    "quarterly" if period_type and "quarter" in period_type.group() else \
                        "monthly"

                if period_freq not in detected_frequencies:
                    detected_frequencies.append(period_freq)

                when_candidates.append({
                    "text": match.group(),
                    "method": "period_end_pattern",
                    "score": 0.85,
                    "span": [start, end],
                    "frequency": period_freq,
                    "is_primary": True,
                    "is_vague": False,
                    "context": text[max(0, start - 30):min(len(text), end + 30)]
                })

            # Check for closing period patterns
            for match in CLOSE_PERIOD_PATTERN.finditer(text_lower):
                start, end = match.span()

                # Skip if this is part of a vague term
                if any(vague["span"][0] <= start and vague["span"][1] >= end for vague in vague_terms_found):
                    continue

                # Found a specific timing expression
                specific_timing_found = True

                # Default to monthly for closure cycles
                if "monthly" not in detected_frequencies:
                    detected_frequencies.append("monthly")

                when_candidates.append({
                    "text": match.group(),
                    "method": "close_period_pattern",
                    "score": 0.85,
                    "span": [start, end],
                    "frequency": "monthly",
                    "is_primary": True,
                    "is_vague": False,
                    "context": text[max(0, start - 30):min(len(text), end + 30)]
                })

        # ---------- PHASE 7: Check for timeline patterns (within, after, before) ----------
        if not when_candidates or (len(when_candidates) == 1 and "ad-hoc" in detected_frequencies):
            within_pattern = re.compile(r'within\s+(\d+)\s+(day|week|month|business day|working day)s?', re.IGNORECASE)
            within_match = within_pattern.search(text_lower)

            if within_match:
                start, end = within_match.span()

                # Found a specific timing expression
                specific_timing_found = True

                when_candidates.append({
                    "text": within_match.group(),
                    "method": "timeline_pattern",
                    "score": 0.85,
                    "span": [start, end],
                    "pattern_type": "timeframe",
                    "is_primary": True,
                    "is_vague": False,
                    "context": text[max(0, start - 30):min(len(text), end + 30)]
                })

            # Check for other timeline patterns if "within X" pattern not found
            elif TIMELINE_PATTERN.search(text_lower):
                for match in TIMELINE_PATTERN.finditer(text_lower):
                    start, end = match.span()

                    # Get context after the timeline word
                    after_context = text_lower[end:min(len(text_lower), end + 20)]

                    # Skip if what follows isn't substantive
                    if len(after_context.strip()) < 5:
                        continue

                    # Skip if this is part of a vague term
                    if any(vague["span"][0] <= start and vague["span"][1] >= end for vague in vague_terms_found):
                        continue

                    # Determine the full phrase - grab 5-20 words after the timeline keyword
                    full_text = match.group() + after_context

                    # Truncate if encounter end of sentence
                    if '.' in full_text:
                        full_text = full_text.split('.')[0]

                    # Found a specific timing expression
                    specific_timing_found = True

                    when_candidates.append({
                        "text": full_text,
                        "method": "timeline_pattern",
                        "score": 0.80,
                        "span": [start, start + len(full_text)],
                        "pattern_type": "sequential",
                        "is_primary": True,
                        "is_vague": False,
                        "context": text[max(0, start - 30):min(len(text), start + len(full_text) + 30)]
                    })

                    # Only need one strong timeline pattern
                    break

        # ---------- PHASE 8: Check for business cycle and event patterns ----------
        if not when_candidates or (len(when_candidates) == 1 and "ad-hoc" in detected_frequencies):
            # Business cycle patterns
            for match in BUSINESS_CYCLE_PATTERN.finditer(text_lower):
                start, end = match.span()

                # Skip if this is part of a vague term
                if any(vague["span"][0] <= start and vague["span"][1] >= end for vague in vague_terms_found):
                    continue

                # Found a specific timing expression
                specific_timing_found = True

                when_candidates.append({
                    "text": match.group(),
                    "method": "business_cycle_pattern",
                    "score": 0.75,
                    "span": [start, end],
                    "pattern_type": "business_cycle",
                    "is_primary": True,
                    "is_vague": False,
                    "context": text[max(0, start - 30):min(len(text), end + 30)]
                })

                # Only need one business cycle pattern
                break

            # Event trigger patterns
            if not when_candidates or (len(when_candidates) == 1 and "ad-hoc" in detected_frequencies):
                for match in EVENT_TRIGGER_PATTERN.finditer(text_lower):
                    start, end = match.span()

                    # Skip if this is part of a vague term
                    if any(vague["span"][0] <= start and vague["span"][1] >= end for vague in vague_terms_found):
                        continue

                    # Found a specific timing expression
                    specific_timing_found = True

                    when_candidates.append({
                        "text": match.group(),
                        "method": "event_trigger_pattern",
                        "score": 0.75,
                        "span": [start, end],
                        "pattern_type": "event_trigger",
                        "is_primary": True,
                        "is_vague": False,
                        "context": text[max(0, start - 30):min(len(text), end + 30)]
                    })

                    # Only need one event trigger pattern
                    break

        # ---------- PHASE 9: Use spaCy for implicit timing detection only if needed ----------
        if not when_candidates or (len(when_candidates) == 1 and "ad-hoc" in detected_frequencies):
            # Only create spaCy doc if needed (performance optimization)
            doc = nlp(text)

            # Look for temporal modifiers
            for token in doc:
                if token.dep_ == "npadvmod" and token.head.pos_ == "VERB":
                    # Check if it's likely a time-related modifier
                    if any(time_word in token.text.lower() for time_word in
                           ["time", "moment", "instance", "point", "period"]):

                        # Skip if this is part of a vague term
                        token_start = token.idx
                        token_end = token.idx + len(token.text)
                        if any(vague["span"][0] <= token_start and vague["span"][1] >= token_end for vague in
                               vague_terms_found):
                            continue

                        # Skip if it's in an excluded context
                        context_window = text_lower[max(0, token_start - 5):min(len(text_lower), token_end + 15)]
                        if any(excluded in context_window for excluded in
                               ["real-time", "one time", "first time", "last time",
                                "reporting period", "accounting period", "time period"]):
                            continue

                        # Consider this a potential specific timing expression
                        specific_timing_found = True

                        when_candidates.append({
                            "text": token.text,
                            "method": "implicit_temporal_modifier",
                            "score": 0.6,
                            "span": [token.i, token.i + 1],
                            "is_primary": False,
                            "is_vague": False,
                            "context": text[max(0, token.idx - 30):min(len(text), token.idx + len(token.text) + 30)]
                        })

        # ---------- PHASE 10: Add vague terms to candidates ----------
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
                "context": text[max(0, vague["span"][0] - 30):min(len(text), vague["span"][1] + 30)]
            })

        # ---------- PHASE 11: Generate multi-frequency and improvement suggestions ----------

        # Multi-frequency analysis
        is_multi_frequency = len(detected_frequencies) > 1

        if is_multi_frequency:
            improvement_suggestions.append(
                "Multiple frequencies detected. Consider whether this is describing a process rather than a single control."
            )

        if not specific_timing_found and not ad_hoc_match:
            improvement_suggestions.append(
                "No specific timing information detected. Add specific frequency (daily, weekly, monthly) or timing (within X days)."
            )

        for vague_term in vague_terms_found:
            improvement_suggestions.append(
                f"Replace vague timing term '{vague_term['text']}' with {vague_term['suggested_replacement']}."
            )

        # ---------- PHASE 12: Validate against metadata frequency if provided ----------
        validation_result = {"is_valid": True, "message": "No frequency metadata provided for validation"}
        if frequency_metadata:
            normalized_metadata = frequency_metadata.lower().strip()

            # Check if any detected frequency matches metadata
            metadata_match = False

            # Handle ad-hoc specially for validation
            if "ad-hoc" in detected_frequencies and (
                    "ad-hoc" in normalized_metadata or "ad hoc" in normalized_metadata):
                metadata_match = True
            else:
                # Check standard frequencies
                for freq in detected_frequencies:
                    # Simple match
                    if freq == normalized_metadata:
                        metadata_match = True
                        break

                    # Check for variations using the regex patterns
                    patterns_by_freq = {
                        "daily": DAILY_PATTERN,
                        "weekly": WEEKLY_PATTERN,
                        "monthly": MONTHLY_PATTERN,
                        "quarterly": QUARTERLY_PATTERN,
                        "annually": ANNUALLY_PATTERN
                    }

                    if freq in patterns_by_freq and patterns_by_freq[freq].search(normalized_metadata):
                        metadata_match = True
                        break

            if metadata_match:
                validation_result = {"is_valid": True,
                                     "message": f"Frequency in description matches metadata ({normalized_metadata})"}
            else:
                validation_result = {
                    "is_valid": False,
                    "message": f"Frequency in description ({', '.join(detected_frequencies) if detected_frequencies else 'none'}) does not match metadata ({normalized_metadata})"
                }
                improvement_suggestions.append(
                    f"Align the frequency in the description with the declared frequency ({normalized_metadata})"
                )

        # ---------- PHASE 13: Calculate final score ----------
        # Check for primary vague terms (vague terms that are the main timing indicator)
        primary_vague_term = any(c.get("is_primary", False) and c.get("is_vague", False) for c in when_candidates)

        if not specific_timing_found or primary_vague_term:
            # Missing specific timing or primary timing is vague - ZERO SCORE
            final_score = 0
        else:
            # Get the best specific timing score
            specific_scores = [c.get("score", 0) for c in when_candidates if not c.get("is_vague", True)]
            if specific_scores:
                final_score = max(specific_scores)

                # Apply a small penalty for secondary vague terms
                secondary_vague_terms = [vague for vague in vague_terms_found if not any(
                    c.get("is_primary", False) and c.get("is_vague", False) and c.get("text") == vague["text"]
                    for c in when_candidates
                )]

                if secondary_vague_terms:
                    # 10% penalty per secondary vague term, up to 30%
                    penalty = min(0.3, len(secondary_vague_terms) * 0.1)
                    final_score *= (1 - penalty)

            # Context-aware scoring based on control type
            if control_type:
                if control_type.lower() == "detective" and final_score > 0:
                    # Detective controls need very clear timing
                    if not any(c.get("score", 0) >= 0.8 and not c.get("is_vague", False) for c in when_candidates):
                        final_score *= 0.8  # Reduce score if timing isn't very specific
                elif control_type.lower() == "preventive" and final_score > 0:
                    # Preventive controls might have implicit timing
                    final_score = min(1.0, final_score * 1.2)  # Slightly boost score

        # Ensure score is in valid range
        final_score = max(0, min(1, final_score))

        # Find the top match for return
        if when_candidates:
            # Prefer specific timing over vague
            specific_candidates = [c for c in when_candidates if not c.get("is_vague", True)]
            if specific_candidates:
                top_match = max(specific_candidates, key=lambda x: x.get("score", 0))
            else:
                top_match = when_candidates[0]
        else:
            top_match = None

        # Create final result
        result = {
            "candidates": when_candidates,
            "top_match": top_match,
            "score": final_score,
            "extracted_keywords": [c["text"] for c in when_candidates],
            "multi_frequency_detected": is_multi_frequency,
            "frequencies": detected_frequencies,
            "validation": validation_result,
            "vague_terms": vague_terms_found,
            "improvement_suggestions": improvement_suggestions,
            "specific_timing_found": specific_timing_found,
            "primary_vague_term": primary_vague_term
        }

        return result

    except Exception as e:
        print(f"Error in WHEN detection: {str(e)}")
        # Return default empty results on error
        return {
            "candidates": [],
            "top_match": None,
            "score": 0,
            "extracted_keywords": [],
            "multi_frequency_detected": False,
            "frequencies": [],
            "validation": {"is_valid": False, "message": f"Error in analysis: {str(e)}"},
            "vague_terms": [],
            "improvement_suggestions": ["Unable to analyze timing due to an error."]
        }