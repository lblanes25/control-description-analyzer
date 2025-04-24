import spacy
from typing import Dict, List, Any, Optional, Tuple, Union
import re


def enhance_when_detection(text, nlp, control_type=None, existing_keywords=None, frequency_metadata=None):
    """
    Enhanced WHEN detection with improved handling of complex timing patterns,
    multi-frequency detection, and vague term identification.

    Args:
        text: Control description text
        nlp: Loaded spaCy model
        control_type: Optional control type (preventive, detective, etc.) for context-aware scoring
        existing_keywords: Optional list of timing keywords to supplement defaults
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

        # Basic frequency terms with variations
        frequency_patterns = existing_keywords or {
            "daily": ["daily", "each day", "every day", "on a daily basis", "day", "daily basis"],
            "weekly": ["weekly", "each week", "every week", "on a weekly basis", "week", "weekly basis"],
            "monthly": ["monthly", "each month", "every month", "on a monthly basis", "month", "monthly basis"],
            "quarterly": ["quarterly", "each quarter", "every quarter", "on a quarterly basis", "quarter"],
            "annually": ["annually", "yearly", "each year", "every year", "annual", "on an annual basis", "year"]
        }

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
            "as required", "on a regular basis", "timely", "promptly", "from time to time"
        ]

        # Track all timing-related matches
        when_candidates = []
        detected_frequencies = []
        vague_terms_found = []

        # 1. Explicit frequency term matching
        for pattern in all_patterns:
            pattern_regex = r'\b' + re.escape(pattern) + r'\b'
            for match in re.finditer(pattern_regex, text.lower()):
                start, end = match.span()
                detected_freq = pattern_to_frequency.get(pattern, "unknown")

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
                    "context": surrounding_text
                })

        # 2. Detect complex temporal patterns
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
                surrounding_text = text[max(0, start - 30):min(len(text), end + 30)]

                when_candidates.append({
                    "text": match.group(),
                    "method": f"complex_pattern_{pattern_type}",
                    "score": 0.85,
                    "span": [start, end],
                    "pattern_type": pattern_type,
                    "is_primary": True,  # Complex patterns are usually significant
                    "context": surrounding_text
                })

        # 3. Detect vague timing terms
        for term in vague_timing_terms:
            term_regex = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(term_regex, text.lower()):
                start, end = match.span()
                vague_terms_found.append({
                    "text": match.group(),
                    "span": [start, end],
                    "suggested_replacement": _suggest_specific_alternative(term)
                })

        # 4. Check for implicit timing using spaCy's dependency parsing
        # Look for temporal modifiers
        for token in doc:
            if token.dep_ == "npadvmod" and token.head.pos_ == "VERB":
                # Check if it's likely a time-related modifier
                if any(time_word in token.text.lower() for time_word in
                       ["time", "moment", "instance", "point", "period"]):
                    when_candidates.append({
                        "text": token.text,
                        "method": "implicit_temporal_modifier",
                        "score": 0.6,
                        "span": [token.i, token.i + 1],
                        "is_primary": False,
                        "context": text[max(0, token.idx - 30):min(len(text), token.idx + len(token.text) + 30)]
                    })

        # Multi-frequency analysis
        is_multi_frequency = len(detected_frequencies) > 1

        # Generate improvement suggestions
        improvement_suggestions = []

        if is_multi_frequency:
            improvement_suggestions.append(
                "Multiple frequencies detected. Consider whether this is describing a process rather than a single control."
            )

        if not when_candidates:
            improvement_suggestions.append(
                "No timing information detected. Add specific frequency (daily, weekly, monthly) or timing (within X days)."
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
        if when_candidates:
            # Base score on the highest confidence match
            when_candidates.sort(key=lambda x: x["score"], reverse=True)
            final_score = when_candidates[0]["score"]

            # Penalty for vague terms
            final_score -= 0.1 * len(vague_terms_found)

            # Context-aware scoring based on control type
            if control_type:
                if control_type.lower() == "detective" and final_score > 0:
                    # Detective controls need very clear timing
                    if not any(c["score"] >= 0.8 for c in when_candidates):
                        final_score *= 0.8  # Reduce score if timing isn't very specific
                elif control_type.lower() == "preventive" and final_score > 0:
                    # Preventive controls might have implicit timing
                    final_score = min(1.0, final_score * 1.2)  # Slightly boost score

            # Ensure score is in valid range
            final_score = max(0, min(1, final_score))

        # Create final result
        result = {
            "candidates": when_candidates,
            "top_match": when_candidates[0] if when_candidates else None,
            "score": final_score,
            "extracted_keywords": [c["text"] for c in when_candidates],
            "multi_frequency_detected": is_multi_frequency,
            "frequencies": detected_frequencies,
            "validation": validation_result,
            "vague_terms": vague_terms_found,
            "improvement_suggestions": improvement_suggestions
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
        "from time to time": "specific frequency or conditions"
    }
    return suggestions.get(vague_term.lower(), "a specific timeframe or frequency")