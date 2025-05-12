"""
Enhanced WHY Element Detection Module

This module analyzes control descriptions to identify purpose statements using
multiple detection strategies, including pattern matching, semantic analysis,
and alignment with risk descriptions.

The implementation prioritizes reliability and consistent detection across
different control phrasing patterns with configurable thresholds and improves
handling of escalation targets vs. purpose statements.
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Union


def enhance_why_detection(text: str, nlp, risk_description: Optional[str] = None,
                          config: Optional[Dict] = None, control_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhanced WHY detection with improved pattern recognition and contextual understanding.

    This function identifies the purpose/rationale behind a control using:
    1. Multiple pattern detection strategies for explicit purpose statements
    2. Inference of implicit purposes from control actions
    3. Risk alignment analysis
    4. Contextual understanding for purpose categorization
    5. Improved handling of escalation targets vs. purpose statements

    Args:
        text: The control description text
        nlp: The spaCy NLP model
        risk_description: Optional risk description text for alignment analysis
        config: Optional configuration dictionary
        control_id: Optional control identifier for reference

    Returns:
        Dictionary with comprehensive WHY detection results
    """
    # Handle empty text
    if not text or text.strip() == '':
        return create_empty_result()

    # Get configuration settings
    why_config = get_why_config(config)

    # Process text with spaCy for deeper analysis
    doc = nlp(text)

    # Check for multi-control descriptions and analyze each separately if needed
    control_segments = split_multi_control_description(text)
    multi_control = len(control_segments) > 1

    # Initialize results for each control segment
    segment_results = []

    for segment in control_segments:
        segment_doc = nlp(segment)

        # PHASE 1: Extract explicit and implicit purposes
        # Find all explicit purpose statements using comprehensive pattern matching
        explicit_purposes = detect_explicit_purposes(segment, segment_doc, why_config)

        # Extract semantic concepts to support deeper understanding
        control_concepts = extract_semantic_concepts(segment_doc)

        # Check for escalation targets being mistaken as purposes and filter them
        explicit_purposes = filter_escalation_targets(explicit_purposes, segment, nlp)

        # Detect implicit purposes based on control actions when explicit purposes aren't clear
        action_concepts = [c for c in control_concepts if c["type"] == "action"]
        implicit_purposes = infer_implicit_purposes(action_concepts, segment, why_config)

        # Handle special case: temporal prevention patterns (e.g., "before processing")
        if re.search(r"(before|prior to)\s+\w+ing", segment, re.IGNORECASE):
            temporal_purposes = find_temporal_prevention_patterns(segment)
            implicit_purposes.extend(temporal_purposes)

        # PHASE 2: Evaluate purpose quality and check risk alignment
        # Identify vague purpose phrases and apply penalties
        vague_phrases = identify_vague_phrases(explicit_purposes, why_config)

        # Align with mapped risk if provided
        risk_alignment = None
        if risk_description:
            risk_alignment = align_with_risk_description(
                segment, risk_description, explicit_purposes, implicit_purposes,
                control_concepts, nlp, why_config, control_id
            )

        # PHASE 3: Select the best purpose and calculate scores
        # Find the best purpose match and calculate confidence score
        top_match, confidence_score = select_best_purpose(
            explicit_purposes, implicit_purposes, risk_alignment, why_config
        )

        # Classify the purpose intent based on verbs and content
        intent_category = classify_purpose_intent(top_match, segment_doc, risk_description, why_config)

        # Store this segment's results
        segment_results.append({
            "text": segment,
            "explicit_why": explicit_purposes,
            "implicit_why": implicit_purposes,
            "top_match": top_match,
            "score": confidence_score,
            "vague_why_phrases": vague_phrases,
            "intent_category": intent_category,
            "risk_alignment": risk_alignment
        })

    # PHASE 4: Combine results and generate feedback
    combined_results = combine_segment_results(segment_results, multi_control)

    # Generate improvement suggestions
    improvement_suggestions = generate_improvement_suggestions(
        combined_results["explicit_why"],
        combined_results["implicit_why"],
        combined_results["vague_why_phrases"],
        combined_results.get("risk_alignment"),
        combined_results["top_match"],
        why_config,
        multi_control
    )

    # Check for additional quality indicators
    has_success_criteria = detect_success_criteria(text)
    is_actual_mitigation = detect_mitigation_verbs(text, why_config)

    # Extract all keywords for reporting - but ONLY when we have explicit purposes
    extracted_keywords = []
    if combined_results["explicit_why"]:
        extracted_keywords = [p["text"] for p in combined_results["explicit_why"] if p]
    elif combined_results["top_match"] and "implied_purpose" in combined_results["top_match"]:
        # If we have an implied purpose in the top match, include it
        extracted_keywords = [combined_results["top_match"]["implied_purpose"]]

    # Categorize purpose using keywords from config
    why_category = categorize_purpose(combined_results["top_match"], why_config)

    # Build and return the complete result
    result = {
        "explicit_why": combined_results["explicit_why"],
        "implicit_why": combined_results["implicit_why"],
        "top_match": combined_results["top_match"],
        "why_category": why_category,
        "score": combined_results["score"],  # Raw confidence score (0-1)
        "is_inferred": combined_results.get("is_inferred", False),  # Clearly mark if the purpose is inferred
        "risk_alignment_score": combined_results.get("risk_alignment", {}).get("score") if combined_results.get(
            "risk_alignment") else None,
        "risk_alignment_feedback": combined_results.get("risk_alignment", {}).get("feedback") if combined_results.get(
            "risk_alignment") else None,
        "extracted_keywords": extracted_keywords,  # Now properly handled for empty cases
        "has_success_criteria": has_success_criteria,
        "vague_why_phrases": combined_results["vague_why_phrases"],
        "is_actual_mitigation": is_actual_mitigation,
        "intent_category": combined_results["intent_category"],
        "improvement_suggestions": improvement_suggestions,
        "is_multi_control": multi_control
    }

    return result


def split_multi_control_description(text: str) -> List[str]:
    """
    Split a control description into separate control segments if it contains multiple controls.

    Args:
        text: Control description text

    Returns:
        List of control segment texts
    """
    # Check for explicit control numbering
    numbered_controls = re.findall(r'Control\s+\d+\s*:([^(Control\s+\d+\s*:)]+)', text)
    if numbered_controls:
        return [segment.strip() for segment in numbered_controls]

    # Check for bullet points or numbered lists
    list_patterns = [
        r'\d+\.\s*([^\d\.]+?)(?=\d+\.\s*|\Z)',  # Numbered list items
        r'•\s*([^•]+?)(?=•|\Z)',                # Bullet points
        r'-\s*([^-]+?)(?=-|\Z)'                 # Dash bullet points
    ]

    for pattern in list_patterns:
        segments = re.findall(pattern, text)
        if len(segments) > 1:
            return [segment.strip() for segment in segments]

    # Check for sentences that appear to be separate controls
    # This is a simplistic approach; for production a more sophisticated NLP-based approach would be better
    if len(text) > 100:  # Only try to split longer text
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if len(s.strip()) > 15]

        # Check if sentences contain different timing patterns or different performers
        timing_patterns = ["daily", "weekly", "monthly", "quarterly", "annually"]
        has_different_timing = False

        timing_counts = {}
        for pattern in timing_patterns:
            for i, sentence in enumerate(sentences):
                if re.search(r'\b' + pattern + r'\b', sentence, re.IGNORECASE):
                    if pattern not in timing_counts:
                        timing_counts[pattern] = []
                    timing_counts[pattern].append(i)

        # If different timing patterns are in different sentences, likely multi-control
        if len(timing_counts) > 1 and any(len(indices) == 1 for indices in timing_counts.values()):
            return sentences

        # If different performers are mentioned in different sentences
        performers = []
        for sentence in sentences:
            # Simple heuristic: look for capitalized words that might be roles/departments
            role_candidates = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', sentence)
            if role_candidates:
                performers.append(role_candidates)

        # If different performers in different sentences, likely multi-control
        if len(performers) > 1 and len(set(tuple(p) for p in performers if p)) > 1:
            return sentences

    # Default: return the whole text as one segment
    return [text]


def filter_escalation_targets(purpose_candidates: List[Dict], text: str, nlp) -> List[Dict]:
    """
    Filter out escalation targets that have been mistakenly identified as purpose statements.

    Args:
        purpose_candidates: List of purpose candidates
        text: The control description text
        nlp: spaCy NLP model

    Returns:
        Filtered list of purpose candidates
    """
    # Process the text to identify escalation contexts
    doc = nlp(text.lower())

    filtered_candidates = []

    # Common escalation verbs and their noun forms
    escalation_terms = [
        "escalate", "escalation", "escalated", "report", "notify", "alert",
        "inform", "communicate", "refer", "forward", "route", "transmit"
    ]

    # Common roles that might be escalation targets
    role_indicators = [
        "manager", "director", "supervisor", "team", "department", "committee",
        "board", "officer", "executive", "cfo", "ceo", "cio", "head", "chief",
        "president", "vp", "vice president", "group", "lead", "leadership"
    ]

    for candidate in purpose_candidates:
        purpose_text = candidate.get("text", "").lower()

        # Skip if the candidate doesn't start with "to"
        if not purpose_text.startswith("to "):
            filtered_candidates.append(candidate)
            continue

        # Check if this looks like an escalation target
        words = purpose_text.split()
        if len(words) >= 2:
            # "to [someone]" pattern check
            second_word = words[1]

            # Check for "to the [role]" pattern
            if len(words) >= 3 and words[1] == "the":
                second_word = words[2]

            # Check if it's a role or contains a role indicator
            is_role = second_word in role_indicators or any(indicator in purpose_text for indicator in role_indicators)

            # Check for preceding escalation context
            has_escalation_context = False
            span_start = candidate.get("span", [0, 0])[0]

            # Check words before the purpose statement
            preceding_text = text[:span_start].lower()
            has_escalation_context = any(term in preceding_text for term in escalation_terms)

            # If it's a role and in escalation context, likely an escalation target, not a purpose
            if is_role and has_escalation_context:
                # Skip this candidate
                continue

            # Additional check: Validate that the purpose has a verb after "to"
            # "to ensure" is a purpose, "to the manager" is not
            purpose_verb_pattern = r'\bto\s+(\w+)'
            verb_match = re.search(purpose_verb_pattern, purpose_text)

            if verb_match:
                verb = verb_match.group(1)
                # Get the token for this verb to check if it's actually a verb
                verb_doc = nlp(verb)
                if verb_doc[0].pos_ != "VERB" and not is_role:
                    # If it's not a verb and not a role, it might be a recipient, so skip
                    continue

        # If it passed all checks, keep it
        filtered_candidates.append(candidate)

    return filtered_candidates


def combine_segment_results(segment_results: List[Dict], is_multi_control: bool) -> Dict:
    """
    Combine results from multiple control segments into a unified result.

    Args:
        segment_results: List of results from each control segment
        is_multi_control: Whether the control description contains multiple controls

    Returns:
        Combined results dictionary
    """
    if not segment_results:
        return {
            "explicit_why": [],
            "implicit_why": [],
            "top_match": None,
            "score": 0,
            "vague_why_phrases": [],
            "intent_category": None,
            "risk_alignment": None
        }

    if len(segment_results) == 1:
        # Single control, just return its results
        return {
            "explicit_why": segment_results[0]["explicit_why"],
            "implicit_why": segment_results[0]["implicit_why"],
            "top_match": segment_results[0]["top_match"],
            "score": segment_results[0]["score"],
            "vague_why_phrases": segment_results[0]["vague_why_phrases"],
            "intent_category": segment_results[0]["intent_category"],
            "risk_alignment": segment_results[0].get("risk_alignment"),
            "is_inferred": segment_results[0]["top_match"] and segment_results[0]["top_match"].get("method", "") in [
                "inferred_from_action", "temporal_pattern", "derived_from_risk"
            ] if segment_results[0]["top_match"] else False
        }

    # Multiple control segments - need to combine results
    all_explicit = []
    all_implicit = []
    all_vague = []

    for result in segment_results:
        all_explicit.extend(result["explicit_why"])
        all_implicit.extend(result["implicit_why"])
        all_vague.extend(result["vague_why_phrases"])

    # Find the best top_match across all segments
    best_score = 0
    best_match = None
    best_intent = None
    best_risk_alignment = None

    for result in segment_results:
        if result["top_match"] and result["score"] > best_score:
            best_score = result["score"]
            best_match = result["top_match"]
            best_intent = result["intent_category"]
            best_risk_alignment = result.get("risk_alignment")

    # Check if the best match is inferred
    is_inferred = best_match and best_match.get("method", "") in [
        "inferred_from_action", "temporal_pattern", "derived_from_risk"
    ] if best_match else False

    return {
        "explicit_why": all_explicit,
        "implicit_why": all_implicit,
        "top_match": best_match,
        "score": best_score,
        "vague_why_phrases": all_vague,
        "intent_category": best_intent,
        "risk_alignment": best_risk_alignment,
        "is_inferred": is_inferred
    }


def get_why_config(config: Optional[Union[Dict, List]]) -> Dict:
    """
    Extract WHY-specific configuration or provide defaults.

    Args:
        config: Full configuration dictionary, list of keywords, or None

    Returns:
        WHY-specific configuration with defaults applied
    """
    # Define default configuration
    default_config = {
        "weight": 11,  # Default weight for the WHY element

        # Purpose phrase indicators
        "keywords": [
            "to ensure", "in order to", "for the purpose of", "designed to",
            "intended to", "so that", "purpose", "objective", "goal",
            "prevent", "detect", "mitigate", "risk", "error", "fraud",
            "misstatement", "compliance", "regulatory", "requirement",
            "accuracy", "completeness", "validity", "integrity"
        ],

        # Regex patterns for finding purpose phrases - enhanced for reliability
        "purpose_patterns": [
            # Direct purpose statements - improved to avoid matching escalation targets
            r'(?i)to\s+(ensure|verify|confirm|validate|prevent|detect|mitigate|comply|adhere|demonstrate|maintain|support|achieve|provide)\s+([^\.;,]{3,50})',

            # Modified pattern for "to [verb]" that excludes common role words
            r'(?i)to\s+(?!(?:the|a|an|our|their)\s)(\w+)\s+([^\.;,]{3,50})',

            # Specific pattern for common start-of-text WHY statements
            r'(?i)^to\s+(?!(?:the|a|an|our|their)\s)(\w+)\s+([^\.;,]{3,50})',

            r'(?i)in\s+order\s+to\s+([^\.;,]{3,50})',
            r'(?i)for\s+the\s+purpose\s+of\s+([^\.;,]{3,50})',
            r'(?i)designed\s+to\s+([^\.;,]{3,50})',
            r'(?i)intended\s+to\s+([^\.;,]{3,50})',
            r'(?i)so\s+that\s+([^\.;,]{3,50})',
            r'(?i)with\s+the\s+aim\s+(?:of|to)\s+([^\.;,]{3,50})',
            r'(?i)in\s+an\s+effort\s+to\s+([^\.;,]{3,50})',

            # Explicit lead-in phrases (these should score very high)
            r'(?i)the\s+purpose\s+(?:of\s+this\s+control\s+is|is)\s+to\s+([^\.;,]{3,50})',
            r'(?i)this\s+control\s+(?:is\s+designed|exists|is\s+in\s+place)\s+to\s+([^\.;,]{3,50})',
            r'(?i)the\s+objective\s+(?:of\s+this\s+control\s+is|is)\s+to\s+([^\.;,]{3,50})',
            r'(?i)this\s+control\s+helps\s+(?:to|in)\s+([^\.;,]{3,50})',

            # Middle-of-sentence purpose phrases
            r'(?i)(?:is|are|will\s+be)\s+(?:performed|executed|conducted|carried\s+out)\s+to\s+([^\.;,]{3,50})',
            r'(?i)(?:which|that)\s+(?:helps|serves)\s+to\s+([^\.;,]{3,50})'
        ],

        # Terms indicating a vague purpose
        "vague_terms": [
            "proper functioning", "appropriate", "adequately", "properly",
            "as needed", "as required", "as appropriate", "correct functioning",
            "effective", "efficient", "functioning", "operational", "successful",
            "appropriate action", "necessary action", "properly functioning"
        ],

        # Verbs indicating different types of intent
        "intent_verbs": {
            "preventive": ["prevent", "avoid", "stop", "block", "prohibit", "restrict", "limit"],
            "detective": ["detect", "identify", "discover", "find", "monitor", "review", "reconcile"],
            "corrective": ["correct", "resolve", "address", "fix", "remediate", "rectify", "resolve"],
            "compliance": ["comply", "adhere", "conform", "follow", "meet", "satisfy", "fulfill"]
        },

        # Verbs that indicate actual mitigation (not just detection)
        "mitigation_verbs": [
            "resolve", "correct", "address", "remediate", "fix", "prevent",
            "block", "stop", "deny", "restrict", "escalate", "alert",
            "notify", "disable", "lockout", "report"
        ],

        # Mapping of control verbs to implied purposes
        "verb_purpose_mapping": {
            "review": {
                "default": "to ensure accuracy and completeness",
                "approval": "to ensure proper authorization",
                "changes": "to prevent unauthorized changes",
                "access": "to ensure appropriate access"
            },
            "approve": {
                "default": "to ensure proper authorization",
                "changes": "to prevent unauthorized changes",
                "transactions": "to ensure authorized transactions",
                "documents": "to ensure document validity"
            },
            "reconcile": {
                "default": "to ensure data integrity and accuracy"
            },
            "verify": {
                "default": "to confirm accuracy and validity",
                "approval": "to ensure proper authorization"
            },
            "validate": {
                "default": "to ensure compliance and accuracy",
                "approval": "to validate authorization"
            },
            "monitor": {
                "default": "to detect anomalies or non-compliance",
                "changes": "to identify unauthorized changes"
            },
            "check": {
                "default": "to identify errors or inconsistencies",
                "approval": "to verify authorization"
            }
        },

        # Purpose categories with their keywords
        "categories": {
            "risk_mitigation": ["risk", "prevent", "mitigate", "reduce", "avoid", "minimize"],
            "compliance": ["comply", "compliance", "regulatory", "regulation", "requirement", "policy", "standard"],
            "accuracy": ["accuracy", "accurate", "correct", "error-free", "integrity", "reliable"],
            "completeness": ["complete", "completeness", "all", "comprehensive"],
            "authorization": ["authorize", "approval", "permission", "authorization"]
        },

        # Confidence score thresholds for different types of matches
        "confidence_thresholds": {
            "explicit_lead_in": 0.95,  # Explicit purpose statements
            "direct_purpose": 0.9,  # Direct purpose phrases
            "indirect_purpose": 0.75,  # Indirect purpose references
            "implied_purpose": 0.6,  # Purposes inferred from actions
            "minimal_purpose": 0.4  # Bare minimum indicators
        }
    }

    # Handle case where config is a list
    if isinstance(config, list):
        # If config is a list, assume it's a list of keywords
        custom_config = default_config.copy()
        custom_config["keywords"] = config
        return custom_config

    # If no config provided, return defaults
    if not config:
        return default_config

    # Get WHY element config if available
    why_config = config.get("elements", {}).get("WHY", {})

    # Start with defaults and override with provided values
    merged_config = default_config.copy()

    # Apply provided configurations
    for key, value in why_config.items():
        if key in default_config:
            # For lists, either replace or extend based on append flag
            if isinstance(value, list) and isinstance(default_config[key], list):
                append_flag = why_config.get(f"append_{key}", True)
                if append_flag:
                    # Combine lists without duplicates
                    merged_values = default_config[key].copy()
                    for item in value:
                        if item not in merged_values:
                            merged_values.append(item)
                    merged_config[key] = merged_values
                else:
                    # Replace the entire list
                    merged_config[key] = value
            # For dictionaries, do a deep merge
            elif isinstance(value, dict) and isinstance(default_config[key], dict):
                for subkey, subvalue in value.items():
                    if subkey in default_config[key]:
                        if isinstance(subvalue, dict) and isinstance(default_config[key][subkey], dict):
                            # Deep merge for nested dictionaries
                            merged_config[key][subkey] = {**default_config[key][subkey], **subvalue}
                        else:
                            # Replace value
                            merged_config[key][subkey] = subvalue
                    else:
                        # Add new entry
                        merged_config[key][subkey] = subvalue
            else:
                # Simple replacement for scalar values
                merged_config[key] = value

    # Always ensure we have the weight
    if "weight" not in merged_config:
        merged_config["weight"] = config.get("elements", {}).get("WHY", {}).get("weight", 11)

    return merged_config


def create_empty_result() -> Dict:
    """
    Create a default result structure for when no text is provided.

    Returns:
        Empty result dictionary
    """
    return {
        "explicit_why": [],
        "implicit_why": [],
        "top_match": None,
        "why_category": None,
        "score": 0,
        "is_inferred": False,
        "risk_alignment_score": None,
        "risk_alignment_feedback": None,
        "extracted_keywords": [],  # Empty list, NOT the original text
        "has_success_criteria": False,
        "vague_why_phrases": [],
        "is_actual_mitigation": False,
        "intent_category": None,
        "improvement_suggestions": ["No clear purpose statement detected in the control description."],
        "is_multi_control": False
    }


def detect_explicit_purposes(text: str, doc, config: Dict) -> List[Dict]:
    """
    Identify explicit purpose phrases in the control description using
    comprehensive pattern matching with case-insensitive detection.

    Args:
        text: Control description text
        doc: spaCy Doc object
        config: WHY configuration dictionary

    Returns:
        List of purpose phrase candidates with metadata
    """
    purpose_patterns = config.get("purpose_patterns", [])
    purpose_keywords = config.get("keywords", [])
    purpose_candidates = []
    confidence_thresholds = config.get("confidence_thresholds", {})

    # Calculate text word count early to avoid issues later
    text_word_count = len(text.split())

    # Define max_percentage value with special handling for short controls
    # Allow the entire text to be a purpose statement if very short (≤ 12 words)
    if text_word_count <= 12:
        max_percentage = 1.0  # Allow 100% of text for very short controls
    elif text_word_count <= 30:
        max_percentage = 0.85  # More lenient for short-medium controls
    else:
        max_percentage = 0.7   # Standard limit for longer controls

    # Special case for entire control being a purpose statement
    # If the control starts with "To prevent", "To ensure", etc.
    # Special case for short descriptions that are likely just the WHY phrase
    if re.match(r'(?i)^to\s+\w+\s+', text) and text_word_count <= 12:
        # Try to extract just the purpose phrase using regex
        match = re.match(r'(?i)(to\s+\w+\s+[^\.;,]{3,50})', text.strip())
        if match:
            matched_phrase = match.group(1).strip()

            # Extract verb for context check
            verb_match = re.match(r'(?i)to\s+(\w+)', matched_phrase)
            purpose_verb = verb_match.group(1) if verb_match else None

            # Only accept recognized verbs to reduce false positives
            purpose_verbs = [
                "ensure", "verify", "confirm", "validate", "prevent", "detect",
                "mitigate", "comply", "adhere", "demonstrate", "maintain",
                "support", "achieve", "provide", "identify"
            ]

            if purpose_verb and purpose_verb.lower() in purpose_verbs:
                purpose_candidates.append({
                    "text": matched_phrase,
                    "verb": purpose_verb,
                    "method": "direct_purpose_statement",
                    "score": 0.95,
                    "span": [0, len(matched_phrase)],
                    "context": matched_phrase
                })

            return purpose_candidates

    # Method 1: Pattern-based detection (primary method)
    # Use regex patterns with case-insensitive flag
    for pattern in purpose_patterns:
        for match in re.finditer(pattern, text):
            purpose_text = match.group(0)

            # Get the purpose verb if available (e.g., "ensure", "prevent")
            purpose_verb = None
            if "to " in purpose_text.lower() and len(match.groups()) > 0:
                verb_match = re.search(r'to\s+(\w+)', purpose_text.lower())
                if verb_match:
                    purpose_verb = verb_match.group(1)

            # Determine confidence score based on pattern type
            if any(lead_in in pattern.lower() for lead_in in [
                "purpose of this control", "this control is designed",
                "this control exists", "objective of this control"
            ]):
                # Explicit lead-in phrases get highest confidence
                confidence = confidence_thresholds.get("explicit_lead_in", 0.95)
                detection_method = "explicit_lead_in"
            elif purpose_text.lower().startswith("to "):
                # Direct purpose statements score high
                confidence = confidence_thresholds.get("direct_purpose", 0.9)
                detection_method = "direct_purpose_statement"
            elif any(phrase in purpose_text.lower() for phrase in [
                "in order to", "for the purpose of", "designed to",
                "intended to", "so that"
            ]):
                # Indirect purpose statements score well
                confidence = confidence_thresholds.get("indirect_purpose", 0.75)
                detection_method = "indirect_purpose_reference"
            else:
                # Other patterns score moderately
                confidence = 0.7
                detection_method = "pattern_match"

            # Special case boost for start-of-text WHY statements
            if match.start() == 0:
                confidence = min(0.95, confidence * 1.1)  # Boost confidence for statements at beginning

            # Validate the match - now more flexible for short controls
            if len(purpose_text.split()) / text_word_count <= max_percentage:
                purpose_candidates.append({
                    "text": purpose_text,
                    "verb": purpose_verb,
                    "method": detection_method,
                    "score": confidence,
                    "span": [match.start(), match.end()],
                    "context": text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
                })

    # Method 2: Keyword-based detection (backup method)
    # Only used when pattern detection finds nothing
    if not purpose_candidates:
        for keyword in purpose_keywords:
            # Case-insensitive search
            keyword_lower = keyword.lower()
            text_lower = text.lower()

            if keyword_lower in text_lower:
                # Find position of keyword
                pos = text_lower.find(keyword_lower)

                # Find the containing sentence
                sentence = None
                for sent in doc.sents:
                    if pos >= sent.start_char and pos < sent.end_char:
                        sentence = sent
                        break

                # Only consider valid sentences that aren't too long
                if sentence and len(sentence.text.split()) / text_word_count <= max_percentage:
                    # Verify the keyword is used in a purpose context
                    keyword_pos = sentence.text.lower().find(keyword_lower)

                    # Check if keyword is at sentence start or after certain punctuation
                    if (keyword_pos == 0 or
                            sentence.text[keyword_pos - 1] in " .,;:(" or
                            re.search(r'\b(is|are|was|were|be)\s+' + re.escape(keyword_lower),
                                      sentence.text.lower())):
                        purpose_candidates.append({
                            "text": sentence.text,
                            "method": "keyword_match",
                            "score": confidence_thresholds.get("minimal_purpose", 0.4),  # Lower confidence
                            "span": [sentence.start_char, sentence.end_char],
                            "context": text[max(0, sentence.start_char - 20):min(len(text), sentence.end_char + 20)]
                        })

    # Method 3: Additional detection for mid-sentence purpose clauses
    # This catches purpose phrases that might be embedded in longer text
    mid_sentence_patterns = [
        r'\s+(?:which|that|to)\s+(?:ensures?|prevents?|detects?|mitigates?)\s+([^\.;,]{3,40})',
        r'\s+(?:which|that|to)\s+(?:helps?|serves?)\s+(?:ensure|prevent|detect|mitigate)\s+([^\.;,]{3,40})'
    ]

    for pattern in mid_sentence_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            mid_text = match.group(0)

            # Only add if this doesn't overlap with an existing candidate
            overlaps = False
            for candidate in purpose_candidates:
                if (match.start() >= candidate["span"][0] and
                        match.start() <= candidate["span"][1]):
                    overlaps = True
                    break

            if not overlaps:
                purpose_candidates.append({
                    "text": mid_text.strip(),
                    "method": "mid_sentence_purpose",
                    "score": confidence_thresholds.get("minimal_purpose", 0.4),
                    "span": [match.start(), match.end()],
                    "context": text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
                })

    return purpose_candidates


def extract_semantic_concepts(doc) -> List[Dict]:
    """
    Extract semantic concepts from the document including actions, objects,
    modifiers, and attributes to enable deeper semantic understanding.

    Args:
        doc: spaCy Doc object

    Returns:
        List of concept dictionaries
    """
    concepts = []

    # Extract action-object pairs
    for token in doc:
        # Find verbs (actions)
        if token.pos_ == "VERB":
            # Create action concept
            action = {
                "type": "action",
                "verb": token.lemma_,
                "text": token.text,
                "modifiers": [],
                "objects": []
            }

            # Find objects and modifiers
            for child in token.children:
                # Objects (what the action is applied to)
                if child.dep_ in ["dobj", "pobj", "attr"]:
                    # Get the complete noun phrase
                    obj_text = child.text
                    for chunk in doc.noun_chunks:
                        if child.i >= chunk.start and child.i < chunk.end:
                            obj_text = chunk.text
                            break

                    action["objects"].append({
                        "text": obj_text,
                        "lemma": child.lemma_,
                        "has_modifiers": any(c.dep_ in ["amod", "compound"] for c in child.children)
                    })

                # Modifiers (how the action is performed)
                elif child.dep_ in ["advmod", "amod", "aux"]:
                    action["modifiers"].append({
                        "text": child.text,
                        "lemma": child.lemma_
                    })

            # Only add actions with objects or modifiers
            if action["objects"] or action["modifiers"]:
                concepts.append(action)

    # Extract negations (e.g., "without approval")
    for token in doc:
        if token.dep_ == "neg" or token.text.lower() in ["without", "no", "lack", "missing"]:
            head = token.head
            concepts.append({
                "type": "negation",
                "target": head.lemma_,
                "text": f"{token.text} {head.text}",
                "negates": "approval" if "approv" in head.lemma_ else head.lemma_
            })

    # Extract important attribute modifiers
    for token in doc:
        if token.dep_ == "amod" and token.text.lower() in ["appropriate", "proper", "unauthorized", "authorized"]:
            head = token.head
            concepts.append({
                "type": "attribute",
                "modifier": token.lemma_,
                "target": head.lemma_,
                "text": f"{token.text} {head.text}"
            })

    return concepts


def infer_implicit_purposes(action_concepts: List[Dict], text: str, config: Dict) -> List[Dict]:
    """
    Infer implicit purposes from control actions using verb-purpose mappings.

    Args:
        action_concepts: List of action concept dictionaries
        text: Original control text
        config: WHY configuration

    Returns:
        List of implicit purpose candidates
    """
    verb_purpose_mapping = config.get("verb_purpose_mapping", {})
    confidence_thresholds = config.get("confidence_thresholds", {})
    implicit_purposes = []

    for action in action_concepts:
        verb = action["verb"]

        # Check if we have a purpose mapping for this verb
        if verb in verb_purpose_mapping:
            # Default purpose
            purpose_key = "default"

            # Try to find a more specific purpose based on objects
            for obj in action["objects"]:
                obj_text = obj["text"].lower()
                for key in verb_purpose_mapping[verb]:
                    if key != "default" and key in obj_text:
                        purpose_key = key
                        break

            # Special case for approval with temporal prevention pattern
            if verb == "approve" and re.search(r"(before|prior to)\s+\w+ing", text, re.IGNORECASE):
                purpose = "to prevent unauthorized actions"
            else:
                purpose = verb_purpose_mapping[verb][purpose_key]

            # Calculate confidence score based on specificity
            if purpose_key != "default":
                # Higher confidence for context-specific purposes
                confidence = min(0.75, confidence_thresholds.get("implied_purpose", 0.6) + 0.05)
            else:
                confidence = confidence_thresholds.get("implied_purpose", 0.6)

            # Create the implicit purpose candidate
            implicit_purposes.append({
                "text": f"{verb} {' '.join([obj['text'] for obj in action['objects']])}",
                "implied_purpose": purpose,
                "score": confidence,
                "method": "inferred_from_action",
                "context": "actions"
            })

    return implicit_purposes


def find_temporal_prevention_patterns(text: str) -> List[Dict]:
    """
    Find temporal patterns that indicate prevention focus.

    Args:
        text: Control description text

    Returns:
        List of implicit prevention purposes
    """
    purposes = []

    # Look for before/prior to patterns
    pattern = r"(before|prior to)\s+\w+ing"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        # Extract the action being prevented
        action_match = re.search(r"(before|prior to)\s+(\w+ing)", text, re.IGNORECASE)
        action = action_match.group(2) if action_match else "action"

        purposes.append({
            "text": match.group(0),
            "implied_purpose": f"to prevent unauthorized {action}",
            "score": 0.7,  # Good confidence for temporal prevention
            "method": "temporal_pattern",
            "context": "temporal_prevention"
        })

    return purposes


def identify_vague_phrases(purpose_candidates: List[Dict], config: Dict) -> List[Dict]:
    """
    Identify vague purpose phrases and apply score penalties.

    Args:
        purpose_candidates: List of purpose candidates
        config: WHY configuration

    Returns:
        List of vague phrase dictionaries
    """
    vague_terms = config.get("vague_terms", [])
    vague_phrases = []

    for candidate in purpose_candidates:
        candidate_text = candidate["text"].lower()

        for vague_term in vague_terms:
            if vague_term.lower() in candidate_text:
                vague_phrases.append({
                    "text": candidate_text,
                    "vague_term": vague_term,
                    "suggested_replacement": "specific risk, impact, or compliance requirement"
                })

                # Apply score penalty for vague terms
                candidate["score"] = candidate["score"] * 0.7
                candidate["has_vague_term"] = True

    return vague_phrases


def align_with_risk_description(text: str, risk_description: str,
                                explicit_purposes: List[Dict], implicit_purposes: List[Dict],
                                control_concepts: List[Dict], nlp, config: Dict,
                                control_id: Optional[str] = None) -> Dict:
    """
    Analyze alignment between control purpose and risk description.

    Args:
        text: Control description text
        risk_description: Risk description text
        explicit_purposes: Explicit purpose candidates
        implicit_purposes: Implicit purpose candidates
        control_concepts: Control semantic concepts
        nlp: spaCy model
        config: WHY configuration
        control_id: Optional control ID for reference

    Returns:
        Dictionary with alignment score and feedback
    """
    # Process risk text
    risk_doc = nlp(risk_description.lower())
    risk_concepts = extract_semantic_concepts(risk_doc)

    # Extract risk aspects for partial matching
    risk_aspects = extract_risk_aspects(risk_description)

    # Identify semantic relationships between control and risk
    relationships = identify_concept_relationships(control_concepts, risk_concepts)

    # Check for special case: approval controls addressing unauthorized changes
    approval_terms = ["approv", "authoriz", "review"]
    change_terms = ["chang", "modif", "updat"]

    # Check for patterns in control
    control_has_approval = any(term in text.lower() for term in approval_terms)
    control_has_changes = any(term in text.lower() for term in change_terms)

    # Check for patterns in risk
    risk_has_approval = any(term in risk_description.lower() for term in approval_terms)
    risk_has_changes = any(term in risk_description.lower() for term in change_terms)
    risk_has_negation = "without" in risk_description.lower() or "no " in risk_description.lower()

    # Special case: approval controls addressing unauthorized changes
    if control_has_approval and control_has_changes and risk_has_approval and risk_has_changes and risk_has_negation:
        # Add strong relationship
        relationships.append({
            "relationship": "addresses_unauthorized_changes",
            "score": 0.95,
            "description": "Control implements approval process to address unauthorized changes"
        })

        # Add explicit purpose derived from risk
        explicit_purposes.append({
            "text": f"to prevent {risk_description.lower()}",
            "method": "derived_from_risk",
            "score": 0.85,
            "span": [0, len(text)],
            "context": "Derived from risk description"
        })

    # Calculate alignment score
    if relationships:
        # Use maximum relationship score as base
        max_rel_score = max(rel["score"] for rel in relationships)

        # Calculate term overlap
        control_terms = set(t.lemma_.lower() for t in nlp(text)
                            if not t.is_stop and t.pos_ in ["NOUN", "VERB", "ADJ"])
        risk_terms = set(t.lemma_.lower() for t in nlp(risk_description)
                         if not t.is_stop and t.pos_ in ["NOUN", "VERB", "ADJ"])

        term_overlap = len(control_terms.intersection(risk_terms)) / len(risk_terms) if risk_terms else 0

        # Combined score with relationship focus
        alignment_score = (0.7 * max_rel_score) + (0.3 * term_overlap)

        # Generate feedback based on alignment level
        if alignment_score >= 0.7:
            # Strong alignment
            top_rel = max(relationships, key=lambda x: x["score"])
            feedback = f"Strong alignment with mapped risk. Control {top_rel.get('description', 'addresses the risk directly')}."
        elif alignment_score >= 0.4:
            # Moderate alignment
            feedback = "Moderate alignment with mapped risk."
            if len(risk_aspects) > 1:
                feedback += f" Primarily addresses: '{risk_aspects[0]}'."
        else:
            # Weak alignment
            feedback = f"Weak alignment with mapped risk: '{risk_description}'. Consider explicitly addressing how this control mitigates this specific risk."

            # Add specific improvement suggestions
            if "approval" in risk_description.lower() and "approval" not in text.lower():
                feedback += " Consider explicitly mentioning the approval process."
            elif "change" in risk_description.lower() and "change" not in text.lower():
                feedback += " Consider explicitly addressing the change management aspect."

            # Add control ID reference if provided
            if control_id:
                feedback += f" (Control {control_id})"

        return {
            "score": alignment_score,
            "feedback": feedback,
            "relationships": relationships,
            "term_overlap": term_overlap,
            "risk_aspects": risk_aspects
        }
    else:
        # No relationships found
        return {
            "score": 0.1,
            "feedback": f"Very weak alignment with mapped risk: '{risk_description}'. The control purpose does not clearly address this risk.",
            "relationships": [],
            "term_overlap": 0,
            "risk_aspects": risk_aspects
        }


def extract_risk_aspects(risk_text: str) -> List[str]:
    """
    Extract different components of a risk description to enable partial matching.

    Args:
        risk_text: Risk description text

    Returns:
        List of risk aspects or components
    """
    if not risk_text:
        return []

    aspects = []

    # Split on "and" for compound risks
    if " and " in risk_text:
        parts = risk_text.split(" and ")
        for part in parts:
            if len(part.split()) > 3:  # Only substantial phrases
                aspects.append(part.strip())

    # Split on impact markers
    impact_markers = ["resulting in", "leading to", "causing", "which may cause", "which could result in"]
    for marker in impact_markers:
        if marker in risk_text.lower():
            parts = risk_text.lower().split(marker)
            if len(parts) > 1:
                # Add cause
                aspects.append(parts[0].strip())
                # Add effect
                aspects.append(parts[1].strip())

    # Split by sentences
    if not aspects:
        sentences = [s.strip() for s in re.split(r'[.;]', risk_text) if len(s.strip()) > 10]
        aspects.extend(sentences)

    # Fall back to whole text
    if not aspects:
        aspects = [risk_text]

    return aspects


def identify_concept_relationships(control_concepts: List[Dict], risk_concepts: List[Dict]) -> List[Dict]:
    """
    Identify semantic relationships between control and risk concepts.

    Args:
        control_concepts: Concepts from control description
        risk_concepts: Concepts from risk description

    Returns:
        List of relationship dictionaries
    """
    relationships = []

    # Define relationship patterns
    patterns = [
        # Pattern 1: Control addresses a negation in risk
        {
            "control_type": "action",
            "risk_type": "negation",
            "match_fn": lambda c, r: c["verb"] == r["target"] or any(
                obj["lemma"] == r["target"] for obj in c.get("objects", [])),
            "score": 0.9,
            "relationship": "mitigates_negation"
        },
        # Pattern 2: Control implements attribute in risk
        {
            "control_type": "attribute",
            "risk_type": "attribute",
            "match_fn": lambda c, r: c["modifier"] == r["modifier"] or c["target"] == r["target"],
            "score": 0.8,
            "relationship": "implements_attribute"
        },
        # Pattern 3: Control verb directly addresses risk verb
        {
            "control_type": "action",
            "risk_type": "action",
            "match_fn": lambda c, r: c["verb"] == r["verb"] or any(
                c["verb"] == obj["lemma"] for obj in r.get("objects", [])),
            "score": 0.7,
            "relationship": "verb_match"
        },
        # Pattern 4: Approval-authorization relationship
        {
            "special_case": "approval",
            "match_fn": lambda c, r: (("approv" in c["text"].lower() and "approv" in r["text"].lower()) or
                                      ("authoriz" in c["text"].lower() and "authoriz" in r["text"].lower())),
            "score": 0.85,
            "relationship": "approval_authorization"
        }
    ]

    # Check each control concept against each risk concept
    for c_concept in control_concepts:
        for r_concept in risk_concepts:
            # Process each pattern
            for pattern in patterns:
                if "special_case" in pattern:
                    # Handle special cases
                    if pattern["match_fn"](c_concept, r_concept):
                        relationships.append({
                            "control_concept": c_concept,
                            "risk_concept": r_concept,
                            "relationship": pattern["relationship"],
                            "score": pattern["score"],
                            "description": f"Control {pattern['relationship']} in risk"
                        })
                elif c_concept.get("type") == pattern["control_type"] and r_concept.get("type") == pattern["risk_type"]:
                    if pattern["match_fn"](c_concept, r_concept):
                        relationships.append({
                            "control_concept": c_concept,
                            "risk_concept": r_concept,
                            "relationship": pattern["relationship"],
                            "score": pattern["score"],
                            "description": f"Control {pattern['relationship']} in risk"
                        })

    return relationships


def select_best_purpose(explicit_purposes: List[Dict], implicit_purposes: List[Dict],
                        risk_alignment: Optional[Dict], config: Dict) -> Tuple[Optional[Dict], float]:
    """
    Select the best purpose match and calculate the overall confidence score.

    Args:
        explicit_purposes: Explicit purpose candidates
        implicit_purposes: Implicit purpose candidates
        risk_alignment: Risk alignment result if available
        config: WHY configuration

    Returns:
        Tuple of (best_purpose, confidence_score)
    """
    # Sort by score
    explicit_purposes.sort(key=lambda x: x.get("score", 0), reverse=True)
    implicit_purposes.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Select top match and calculate score
    if explicit_purposes:
        top_match = explicit_purposes[0]
        confidence = top_match.get("score", 0)

        # Enhance confidence based on quality indicators
        if "purpose of this control" in top_match.get("text", "").lower():
            confidence = min(1.0, confidence * 1.1)  # Boost for explicit purpose statement
    elif implicit_purposes:
        top_match = implicit_purposes[0]
        # Apply discount factor for implicit purposes
        confidence = top_match.get("score", 0) * 0.8
    else:
        top_match = None
        confidence = 0

    # Boost confidence for strong risk alignment
    if risk_alignment and risk_alignment.get("score", 0) > 0.7:
        if confidence < 0.5:
            # Significant boost for controls with strong risk alignment
            confidence = max(confidence, 0.5)

        # Add derived purpose from risk if no other match
        if not top_match and risk_alignment.get("score", 0) > 0:
            # Extract risk text from feedback
            risk_text = risk_alignment.get("feedback", "").split("with mapped risk:")[-1].split(".")[0].strip()
            if risk_text:
                derived_purpose = f"to prevent {risk_text}"
                top_match = {
                    "text": derived_purpose,
                    "implied_purpose": derived_purpose,
                    "method": "derived_from_risk",
                    "score": 0.5
                }

    return top_match, confidence


def classify_purpose_intent(top_match: Optional[Dict], doc,
                            risk_description: Optional[str],
                            config: Dict) -> Optional[str]:
    """
    Classify the control purpose intent (preventive, detective, etc.).

    Args:
        top_match: Top purpose match
        doc: spaCy doc for control
        risk_description: Risk description if available
        config: WHY configuration

    Returns:
        Intent classification string or None
    """
    # Return None if no match
    if not top_match:
        return None

    # Get intent verbs from config
    intent_verbs = config.get("intent_verbs", {})
    categories = config.get("categories", {})

    top_match_text = top_match.get("text", "").lower()
    # Also check implied purpose if available
    implied_purpose = top_match.get("implied_purpose", "").lower()
    combined_text = top_match_text + " " + implied_purpose

    # Method 1: Check intent verbs in top match
    for intent, verbs in intent_verbs.items():
        if any(verb in combined_text for verb in verbs):
            return intent.capitalize()

    # Method 2: Check category keywords in top match
    for category, keywords in categories.items():
        if any(keyword in combined_text for keyword in keywords):
            return category.replace("_", " ").capitalize()

    # Method 3: Check verbs in entire document
    doc_verbs = [token.lemma_.lower() for token in doc if token.pos_ == "VERB"]

    for intent, verbs in intent_verbs.items():
        if any(verb in doc_verbs for verb in verbs):
            return intent.capitalize()

    # Default to risk mitigation if risk is provided
    if risk_description:
        return "Risk mitigation"

    return None


def categorize_purpose(top_match: Optional[Dict], config: Dict) -> Optional[str]:
    """
    Categorize the purpose into predefined categories.

    Args:
        top_match: Top purpose match
        config: WHY configuration

    Returns:
        Category string or None
    """
    if not top_match:
        return None

    categories = config.get("categories", {})

    # Check both the text and implied purpose
    text = top_match.get("text", "").lower()
    implied_purpose = top_match.get("implied_purpose", "").lower()
    combined_text = text + " " + implied_purpose

    for category, keywords in categories.items():
        if any(keyword in combined_text for keyword in keywords):
            return category.replace("_", " ").capitalize()

    return None


def generate_improvement_suggestions(explicit_purposes: List[Dict],
                                     implicit_purposes: List[Dict],
                                     vague_phrases: List[Dict],
                                     risk_alignment: Optional[Dict],
                                     top_match: Optional[Dict],
                                     config: Dict,
                                     is_multi_control: bool = False) -> List[str]:
    """
    Generate improvement suggestions for the WHY element.

    Args:
        explicit_purposes: Explicit purpose candidates
        implicit_purposes: Implicit purpose candidates
        vague_phrases: Vague purpose phrases
        risk_alignment: Risk alignment result
        top_match: Top purpose match
        config: WHY configuration
        is_multi_control: Whether this is a multi-control description

    Returns:
        List of improvement suggestions
    """
    suggestions = []

    # Missing purpose
    if not top_match:
        suggestions.append(
            "No clear purpose or objective detected. Add an explicit statement of why the control exists."
        )

    # Vague terms
    for vague in vague_phrases:
        suggestions.append(
            f"Replace vague term '{vague['vague_term']}' with {vague['suggested_replacement']}."
        )

    # Poor risk alignment
    if risk_alignment and risk_alignment.get("score", 0) < 0.4:
        alignment_feedback = risk_alignment.get("feedback", "")
        if "Consider" in alignment_feedback:
            suggestion = alignment_feedback.split("Consider")[1].split("(")[0].strip()
            suggestions.append(f"Consider{suggestion}")

    # Implicit purpose that should be explicit
    if top_match and top_match.get("method") in ["inferred_from_action", "temporal_pattern"]:
        suggestions.append(
            "Make the control purpose explicit by adding a clear statement of why the control exists."
        )

    # Suggest converting implicit to explicit
    if not explicit_purposes and implicit_purposes:
        top_implicit = implicit_purposes[0]
        if "implied_purpose" in top_implicit:
            suggestions.append(
                f"Add explicit purpose statement: '{top_implicit['implied_purpose']}'"
            )

    # Multi-control specific suggestions
    if is_multi_control:
        # If we have multiple controls with different purposes, suggest clarifying
        if len(explicit_purposes) > 1:
            suggestions.append(
                "Multiple purpose statements detected. Consider separating into distinct controls or clarifying the primary purpose."
            )
        # If we have multiple controls but only one purpose, suggest adding purposes for each
        elif len(explicit_purposes) <= 1:
            suggestions.append(
                "Multi-control description detected, but not all controls have clear purposes. Add a specific purpose for each control."
            )

    return suggestions


def detect_success_criteria(text: str) -> bool:
    """
    Detect if the control has specific success criteria.

    Args:
        text: Control description text

    Returns:
        Boolean indicating presence of success criteria
    """
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in [
        r'(\$\d+[,\d]*|\d+\s*%|\d+\s*percent)',
        r'greater than|less than|at least|at most|minimum|maximum',
        r'threshold of|limit of|tolerance of',
        r'within \d+\s*(day|hour|minute|week|month)',
        r'criteria|criterion|standard|benchmark'
    ])


def detect_mitigation_verbs(text: str, config: Dict) -> bool:
    """
    Detect if the control has actual mitigation verbs (not just detection).

    Args:
        text: Control description text
        config: WHY configuration

    Returns:
        Boolean indicating presence of mitigation verbs
    """
    mitigation_verbs = config.get("mitigation_verbs", [])
    return any(re.search(r"\b" + re.escape(verb.lower()) + r"\b", text.lower()) for verb in mitigation_verbs)