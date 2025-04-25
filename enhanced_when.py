import spacy
from typing import Dict, List, Any, Optional, Tuple, Union
import re


def enhance_when_detection(text, nlp, control_type=None, existing_keywords=None, frequency_metadata=None):
    """
    Enhanced WHEN detection with improved handling of complex timing patterns,
    multi-frequency detection, and strict handling of vague terms.

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
        # Process the text
        doc = nlp(text)

        # Simplest, most direct approach to handle common vague terms
        text_lower = text.lower()

        # SPECIAL HANDLING FOR "MAY" - Explicit check for "may" used in an uncertain sense
        # "Reconciliations are performed but may vary..." should be flagged as vague
        if re.search(r'may\s+(?:vary|differ|change|be|not|need)', text_lower) or " may " in text_lower:
            # Exclude cases where "may" is clearly used as the month name
            if not re.search(r'\b(?:in|of|during|by|for|before|after)\s+may\b', text_lower) and not re.search(
                    r'may\s+(?:\d{1,2}|\d{4})', text_lower):
                # "may" used as uncertainty term - treat it as a vague timing term
                vague_match = "may"

                # Create result with near-zero score
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
                    "score": 0.05,  # Near-zero score
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

        # Direct check for problematic primary vague terms at start of control
        starts_with_vague = any(text_lower.strip().startswith(term) for term in [
            "as needed", "when needed", "if needed", "as appropriate", "when appropriate",
            "if appropriate", "as required", "when required", "if required", "periodically",
            "occasionally", "from time to time", "regularly"
        ])

        # Also check for other vague term patterns that shouldn't be missed
        common_vague_patterns = [
            r'^exceptions are (?:addressed|resolved|reviewed|handled) (when|as|if) (needed|appropriate|required)',
            r'^(as|when|if) (appropriate|needed|required)',
            r'^on an (as needed|when needed|if needed) basis',
            r'performed (periodically|occasionally|regularly|as necessary)'
        ]

        for pattern in common_vague_patterns:
            if re.search(pattern, text_lower):
                starts_with_vague = True
                break

        # If control starts with vague term, short-circuit to near-zero score
        if starts_with_vague:
            # Collect the vague term for reporting
            vague_match = None
            for term in ["as needed", "when needed", "if needed", "as appropriate", "when appropriate",
                         "if appropriate", "as required", "when required", "if required", "periodically",
                         "occasionally", "from time to time", "regularly"]:
                if term in text_lower:
                    vague_match = term
                    break

            # Create minimal result with near-zero score
            return {
                "candidates": [{
                    "text": vague_match if vague_match else "vague timing term",
                    "method": "vague_timing",
                    "score": 0.1,
                    "is_vague": True,
                    "is_primary": True
                }],
                "top_match": {
                    "text": vague_match if vague_match else "vague timing term",
                    "method": "vague_timing",
                    "score": 0.1,
                    "is_vague": True,
                    "is_primary": True
                },
                "score": 0.05,  # Near-zero score
                "extracted_keywords": [vague_match] if vague_match else ["vague timing"],
                "multi_frequency_detected": False,
                "frequencies": [],
                "validation": {"is_valid": False, "message": "Vague timing detected"},
                "vague_terms": [{
                    "text": vague_match if vague_match else "vague timing term",
                    "is_primary": True,
                    "suggested_replacement": "specific frequency (daily, weekly, monthly)"
                }],
                "improvement_suggestions": [
                    f"Replace vague timing term '{vague_match if vague_match else 'vague timing term'}' with specific frequency (daily, weekly, monthly)."
                ],
                "specific_timing_found": False,
                "primary_vague_term": True
            }

        # Default frequency patterns - specific timing expressions
        default_frequency_patterns = {
            "daily": ["daily", "each day", "every day", "on a daily basis", "day", "daily basis"],
            "weekly": ["weekly", "each week", "every week", "on a weekly basis", "week", "weekly basis"],
            "monthly": ["monthly", "each month", "every month", "on a monthly basis", "month", "monthly basis"],
            "quarterly": ["quarterly", "each quarter", "every quarter", "on a quarterly basis", "quarter"],
            "annually": ["annually", "yearly", "each year", "every year", "annual", "on an annual basis", "year"]
        }

        # IMPORTANT: Filter out "may" from any frequency patterns to avoid confusion with the auxiliary verb
        for freq, patterns in default_frequency_patterns.items():
            default_frequency_patterns[freq] = [p for p in patterns if p.lower() != "may"]

        # Handle the existing_keywords parameter properly
        frequency_patterns = default_frequency_patterns
        if existing_keywords is not None:
            # Check if existing_keywords is a dictionary
            if isinstance(existing_keywords, dict):
                # Filter out "may" from any patterns
                frequency_patterns = {k: [p for p in v if p.lower() != "may"] for k, v in existing_keywords.items()}
            # If it's a list, assume it's just additional keywords without frequency categorization
            elif isinstance(existing_keywords, list):
                # Add all keywords (except "may") to a generic "other" category
                frequency_patterns = default_frequency_patterns.copy()
                frequency_patterns["other"] = [k for k in existing_keywords if k.lower() != "may"]
            else:
                # Fallback to default if the type is unexpected
                print(f"Warning: existing_keywords has unexpected type {type(existing_keywords)}, using defaults")

        # Flatten the patterns for easier searching
        all_patterns = []
        pattern_to_frequency = {}
        for freq, patterns in frequency_patterns.items():
            for pattern in patterns:
                all_patterns.append(pattern)
                pattern_to_frequency[pattern] = freq

        # Vague timing terms to flag
        vague_timing_terms = [
            "periodically", "regularly", "as needed", "when necessary", "as appropriate",
            "as required", "on a regular basis", "timely", "promptly", "from time to time",
            "when appropriate", "where appropriate", "if needed", "if necessary",
            "occasionally", "sometimes", "at times", "now and then", "intermittently",
            "at intervals", "frequently", "infrequently", "as applicable", "as deemed necessary",
            "when needed", "if appropriate", "where necessary", "as necessary",
            "may vary", "may change", "may differ", "may be", "may not"
        ]

        # Track all timing-related matches
        when_candidates = []
        detected_frequencies = []
        vague_terms_found = []
        specific_timing_found = False  # Flag to track if any specific timing is found
        secondary_vague_terms = []  # Track vague terms that appear after specific timing

        # Detect vague timing terms FIRST to mark them
        for term in vague_timing_terms:
            term_regex = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(term_regex, text.lower()):
                start, end = match.span()

                vague_terms_found.append({
                    "text": match.group(),
                    "span": [start, end],
                    "suggested_replacement": _suggest_specific_alternative(term)
                })

        # Explicit frequency term matching (specific timing)
        for pattern in all_patterns:
            pattern_regex = r'\b' + re.escape(pattern) + r'\b'
            for match in re.finditer(pattern_regex, text.lower()):
                start, end = match.span()
                detected_freq = pattern_to_frequency.get(pattern, "unknown")

                # Skip if this is part of a vague term (e.g., "on a regular basis")
                if any(vague["span"][0] <= start and vague["span"][1] >= end for vague in vague_terms_found):
                    continue

                # Found a specific timing expression
                specific_timing_found = True

                # Determine if this is the main control frequency or a supporting activity
                # Look for verbs around this timing indicator
                surrounding_text = text[max(0, start - 30):min(len(text), end + 30)]
                is_primary = any(
                    word in surrounding_text.lower() for word in ["review", "verify", "check", "ensure", "validate"])

                if detected_freq not in detected_frequencies:
                    detected_frequencies.append(detected_freq)

                when_candidates.append({
                    "text": match.group(),
                    "method": "explicit_frequency",
                    "score": 0.9,
                    "span": [start, end],
                    "frequency": detected_freq,
                    "is_primary": is_primary,
                    "is_vague": False,
                    "context": surrounding_text
                })

        # Detect complex temporal patterns
        # Complex patterns like "within X days of", "after completion of", etc.
        complex_patterns = [
            (r'within\s+(\d+)\s+(day|week|month|business day|working day)s?', "timeframe"),
            (r'after\s+([\w\s]+?)\s+(is|are|has been|have been)', "sequential"),
            (r'prior\s+to\s+([\w\s]+)', "precondition"),
            (r'following\s+([\w\s]+)', "sequential"),
            (r'upon\s+([\w\s]+)', "trigger"),
            (r'by\s+the\s+(\d+)(?:st|nd|rd|th)\s+(day|week|month)', "deadline")
        ]

        for pattern, pattern_type in complex_patterns:
            for match in re.finditer(pattern, text.lower()):
                start, end = match.span()
                # Skip if this is part of a vague term
                if any(vague["span"][0] <= start and vague["span"][1] >= end for vague in vague_terms_found):
                    continue

                # Found a specific timing expression
                specific_timing_found = True
                surrounding_text = text[max(0, start - 30):min(len(text), end + 30)]

                when_candidates.append({
                    "text": match.group(),
                    "method": f"complex_pattern_{pattern_type}",
                    "score": 0.85,
                    "span": [start, end],
                    "pattern_type": pattern_type,
                    "is_primary": True,  # Complex patterns are usually significant
                    "is_vague": False,
                    "context": surrounding_text
                })

        # Check for implicit timing using spaCy's dependency parsing
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

                # Track secondary vague terms for penalty calculation
                if not is_primary:
                    secondary_vague_terms.append(vague)

            when_candidates.append({
                "text": vague["text"],
                "method": "vague_timing",
                "score": 0.1,  # Very low score for vague terms
                "span": vague["span"],
                "is_primary": is_primary,
                "is_vague": True,
                "context": text[max(0, vague["span"][0] - 30):min(len(text), vague["span"][1] + 30)]
            })

        # Multi-frequency analysis
        is_multi_frequency = len(detected_frequencies) > 1

        # Generate improvement suggestions
        improvement_suggestions = []

        if is_multi_frequency:
            improvement_suggestions.append(
                "Multiple frequencies detected. Consider whether this is describing a process rather than a single control."
            )

        if not specific_timing_found:
            improvement_suggestions.append(
                "No specific timing information detected. Add specific frequency (daily, weekly, monthly) or timing (within X days)."
            )

        for vague_term in vague_terms_found:
            improvement_suggestions.append(
                f"Replace vague timing term '{vague_term['text']}' with {vague_term['suggested_replacement']}."
            )

        # Validate against metadata frequency if provided
        validation_result = {"is_valid": True, "message": "No frequency metadata provided for validation"}
        if frequency_metadata:
            normalized_metadata = frequency_metadata.lower().strip()

            # Check if any detected frequency matches metadata
            metadata_match = False
            for freq in detected_frequencies:
                if freq == normalized_metadata or any(
                        pattern.lower() in normalized_metadata for pattern in frequency_patterns.get(freq, [])):
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

        # Calculate final score based on quality of timing information
        final_score = 0

        # Check for primary vague terms (vague terms that are the main timing indicator)
        primary_vague_term = any(c.get("is_primary", False) and c.get("is_vague", False) for c in when_candidates)

        if not specific_timing_found or primary_vague_term:
            # Missing specific timing or primary timing is vague
            final_score = 0.05  # Near zero score (effectively missing)
        else:
            # Get the best specific timing score
            specific_scores = [c.get("score", 0) for c in when_candidates if not c.get("is_vague", True)]
            if specific_scores:
                final_score = max(specific_scores)

                # Apply a small penalty for secondary vague terms
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


def _suggest_specific_alternative(vague_term):
    """Suggest specific alternatives to vague timing terms"""
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