"""
Enhanced WHAT Detection Module

This module implements a sophisticated detection system for the WHAT element in control descriptions.
It identifies actions being performed in controls using a layered approach of detection methods
with explicit fallback strategies, and also handles WHERE components within action phrases.

The implementation leverages control type information to validate action alignment and is
configurable via YAML, with clear tracing of scores and detection methods.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import re


def enhance_what_detection(text: str, nlp, existing_keywords: Optional[List[str]] = None,
                           control_type: Optional[str] = None, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Enhanced WHAT detection with improved verb categorization, strength analysis, and context handling.
    Uses a layered detection approach for robust action identification and validates against control type.

    Args:
        text: The control description text to analyze
        nlp: spaCy NLP model
        existing_keywords: Optional list of action keywords to consider
        control_type: Optional control type (preventive, detective, corrective) for validation
        config: Optional configuration dictionary (overrides default settings)

    Returns:
        Dictionary containing detailed analysis of action elements:
        - primary_action: The main action identified in the control
        - secondary_actions: Additional significant actions identified
        - actions: All action candidates found
        - score: Overall score for the WHAT element
        - voice: Detected voice (active/passive/mixed)
        - suggestions: Improvement suggestions
        - is_process: Boolean indicating if this appears to be a process rather than a control
        - control_type_alignment: Assessment of alignment with provided control type
    """
    if not text or text.strip() == '':
        return {
            "primary_action": None,
            "secondary_actions": [],
            "actions": [],
            "score": 0,
            "voice": None,
            "suggestions": ["No text provided to analyze"],
            "is_process": False,
            "control_type_alignment": {"is_aligned": False, "message": "No text provided for analysis"}
        }

    try:
        # Process the text with spaCy
        doc = nlp(text.lower())

        # Setup verb categories and config
        verb_categories = setup_verb_categories(config, existing_keywords)

        # Initialize tracking variables
        active_voice_count = 0
        passive_voice_count = 0
        action_candidates = []

        # Phase 1: Primary detection - Identify control verbs with dependency parsing
        verb_candidates = extract_control_verb_candidates(doc, verb_categories, config)

        # Update voice tracking
        for candidate in verb_candidates:
            if candidate["is_passive"]:
                passive_voice_count += 1
            else:
                active_voice_count += 1

        action_candidates.extend(verb_candidates)

        # Phase 2: If no strong primary actions found, use pattern-based detection
        if not has_strong_primary_action(verb_candidates):
            pattern_candidates = extract_action_patterns(text, doc, config)
            action_candidates.extend(pattern_candidates)

        # Phase 3: If still no viable actions, use noun phrase fallback
        if len(action_candidates) == 0:
            noun_chunk_candidates = extract_from_noun_chunks(doc, nlp, config)
            action_candidates.extend(noun_chunk_candidates)

        # Filter and rank candidates
        filtered_candidates = filter_action_candidates(action_candidates)

        # Determine primary and secondary actions
        primary_action = None
        secondary_actions = []

        if filtered_candidates:
            # Sort by score
            filtered_candidates.sort(key=lambda x: x["score"], reverse=True)

            # Assign primary action
            primary_action = filtered_candidates[0]

            # Assign secondary actions
            secondary_actions = filtered_candidates[1:3] if len(filtered_candidates) > 1 else []

        # Determine voice
        voice = determine_voice(active_voice_count, passive_voice_count)

        # Evaluate control type alignment if control_type is provided
        control_type_alignment = evaluate_control_type_alignment(
            primary_action,
            secondary_actions,
            control_type,
            text,
            config
        )

        # Calculate final score
        final_score = calculate_final_score(
            primary_action,
            secondary_actions,
            text,
            control_type_alignment["is_aligned"] if control_type else True
        )

        # Generate suggestions
        suggestions = generate_what_suggestions(
            filtered_candidates,
            text,
            control_type,
            control_type_alignment,
            config
        )

        # Determine if this is describing a process rather than a single control
        # Indicators: lots of distinct verbs, length of text, sequence markers
        is_process = determine_if_process(filtered_candidates, text)

        return {
            "primary_action": primary_action,
            "secondary_actions": secondary_actions,
            "actions": filtered_candidates,
            "score": final_score,
            "voice": voice,
            "suggestions": suggestions,
            "is_process": is_process,
            "control_type_alignment": control_type_alignment
        }
    except Exception as e:
        print(f"Error in WHAT detection: {str(e)}")
        import traceback
        traceback.print_exc()

        # Return minimal result on error
        return {
            "primary_action": None,
            "secondary_actions": [],
            "actions": [],
            "score": 0,
            "voice": None,
            "suggestions": [f"Error analyzing text: {str(e)}"],
            "is_process": False,
            "control_type_alignment": {"is_aligned": False, "message": "Error during analysis"}
        }


def has_strong_primary_action(candidates: List[Dict]) -> bool:
    """
    Determine if we have at least one strong action candidate

    Args:
        candidates: List of action candidates

    Returns:
        Boolean indicating presence of a strong action
    """
    return any(c["score"] >= 0.7 for c in candidates)


def determine_if_process(candidates: List[Dict], text: str) -> bool:
    """
    Determine if this is describing a process rather than a single control

    Args:
        candidates: List of action candidates
        text: The full control description

    Returns:
        Boolean indicating if this is likely a process description
    """
    # Indicators:
    # 1. More than 3 distinct action verbs
    # 2. Presence of sequence markers
    # 3. Length of text (very long descriptions often describe processes)

    # Count distinct verbs
    distinct_verbs = set(c["verb_lemma"] for c in candidates)

    # Process indicators score
    process_indicators = 0

    if len(distinct_verbs) > 3:
        process_indicators += 2

    # Check for sequence markers
    sequence_markers = ["then", "after", "before", "next", "subsequently",
                        "finally", "lastly", "following"]

    for marker in sequence_markers:
        if marker in text.lower():
            process_indicators += 1
            break

    # Check text length
    if len(text.split()) > 50:  # If more than 50 words
        process_indicators += 1

    return process_indicators >= 3


def evaluate_control_type_alignment(primary_action: Optional[Dict],
                                    secondary_actions: List[Dict],
                                    control_type: Optional[str],
                                    text: str,
                                    config: Optional[Dict]) -> Dict[str, Any]:
    """
    Evaluate alignment between detected actions and declared control type

    Args:
        primary_action: The primary action or None
        secondary_actions: List of secondary actions
        control_type: Declared control type (preventive, detective, corrective) or None
        text: The full control description
        config: Optional configuration dictionary

    Returns:
        Dictionary with alignment information
    """
    if not control_type:
        # If no control type specified, alignment is irrelevant
        return {
            "is_aligned": True,
            "message": "No control type specified for validation",
            "score": 1.0
        }

    # Normalize control type
    control_type = control_type.lower().strip()

    # If no primary action, alignment is impossible
    if not primary_action:
        return {
            "is_aligned": False,
            "message": "No clear action detected to validate against control type",
            "score": 0.0
        }

    # Get control type indicators from config or use defaults
    control_type_indicators = get_config_value(config, "control_type_indicators", {
        "preventive": [
            "prevent", "block", "restrict", "limit", "prohibit", "stop",
            "validate before", "approve before", "check before",
            "authorize", "authenticate", "gate", "control access"
        ],
        "detective": [
            "detect", "identify", "discover", "find", "review", "monitor",
            "check", "validate", "verify", "reconcile", "compare",
            "inspect", "examine", "audit", "analyze", "assess"
        ],
        "corrective": [
            "correct", "remediate", "fix", "resolve", "address", "adjust",
            "update", "change", "modify", "repair", "rectify", "restore"
        ]
    })

    # Check if control type is valid
    if control_type not in control_type_indicators:
        return {
            "is_aligned": False,
            "message": f"Unknown control type: {control_type}",
            "score": 0.0
        }

    # Find indicators for the declared control type
    indicators = control_type_indicators[control_type]

    # Check primary action for alignment
    primary_verb = primary_action["verb_lemma"]
    primary_phrase = primary_action["full_phrase"].lower()

    # Check if primary action verb is in indicators list
    direct_match = any(indicator in primary_verb for indicator in indicators)

    # Check if full phrase contains an indicator
    phrase_match = any(indicator in primary_phrase for indicator in indicators)

    # Look for indicators in secondary actions
    secondary_match = False
    if secondary_actions:
        for action in secondary_actions:
            if any(indicator in action["verb_lemma"] for indicator in indicators) or \
                    any(indicator in action["full_phrase"].lower() for indicator in indicators):
                secondary_match = True
                break

    # Check full text for indicators (lower confidence)
    text_match = any(indicator in text.lower() for indicator in indicators)

    # Determine alignment score and result
    if direct_match:
        # Strong alignment - primary verb directly matches
        return {
            "is_aligned": True,
            "message": f"Primary action '{primary_verb}' aligns with {control_type} control type",
            "score": 1.0,
            "match_type": "direct"
        }
    elif phrase_match:
        # Good alignment - primary phrase contains indicator
        return {
            "is_aligned": True,
            "message": f"Primary action phrase contains indicators of {control_type} control type",
            "score": 0.8,
            "match_type": "phrase"
        }
    elif secondary_match:
        # Moderate alignment - secondary action contains indicator
        return {
            "is_aligned": True,
            "message": f"Secondary action aligns with {control_type} control type",
            "score": 0.6,
            "match_type": "secondary"
        }
    elif text_match:
        # Weak alignment - only found in full text
        return {
            "is_aligned": True,
            "message": f"Control description contains indicators of {control_type} control type",
            "score": 0.4,
            "match_type": "text"
        }
    else:
        # No alignment found
        return {
            "is_aligned": False,
            "message": f"No alignment found between actions and {control_type} control type",
            "score": 0.0,
            "match_type": "none",
            "suggested_type": suggest_control_type(primary_action, secondary_actions, text, control_type_indicators)
        }


def suggest_control_type(primary_action: Dict, secondary_actions: List[Dict], text: str,
                         type_indicators: Dict[str, List[str]]) -> Optional[str]:
    """
    Suggest a control type based on the detected actions

    Args:
        primary_action: The primary action
        secondary_actions: List of secondary actions
        text: The full control description
        type_indicators: Dictionary of control type indicators

    Returns:
        Suggested control type or None
    """

    # Function to check for indicators in an action
    def check_action_for_type(action, control_type):
        indicators = type_indicators[control_type]
        return any(indicator in action["verb_lemma"] for indicator in indicators) or \
            any(indicator in action["full_phrase"].lower() for indicator in indicators)

    # Check primary action for each control type
    for control_type in type_indicators:
        if check_action_for_type(primary_action, control_type):
            return control_type

    # Check secondary actions
    for action in secondary_actions:
        for control_type in type_indicators:
            if check_action_for_type(action, control_type):
                return control_type

    # Check full text as last resort
    text_lower = text.lower()
    for control_type, indicators in type_indicators.items():
        if any(indicator in text_lower for indicator in indicators):
            return control_type

    # Default to most likely type based on context
    # If we see words like "review", default to detective
    if "review" in text_lower or "monitor" in text_lower or "reconcile" in text_lower:
        return "detective"

    # Default to detective as the most common type
    return "detective"


def setup_verb_categories(config: Optional[Dict], existing_keywords: Optional[List[str]]) -> Dict[
    str, Dict[str, float]]:
    """
    Load verb strength categories from config with fallbacks to default values

    Args:
        config: Configuration dictionary or None
        existing_keywords: Optional list of action keywords

    Returns:
        Dictionary of verb categories with their strength scores
    """
    # Default verb categories if not provided
    default_categories = {
        "high_strength_verbs": {
            "approve": 1.0, "authorize": 1.0, "reconcile": 1.0, "validate": 1.0,
            "certify": 1.0, "sign-off": 1.0, "verify": 1.0, "confirm": 0.9,
            "test": 0.9, "enforce": 0.9, "authenticate": 0.9,
            "audit": 0.9, "inspect": 0.9, "investigate": 0.9, "scrutinize": 0.9,
            "compare": 0.9, "review": 0.85,  # Moved review up from medium
            "check": 0.85, "notify": 0.85, "route": 0.85  # Added more high-value control verbs
        },
        "medium_strength_verbs": {
            "examine": 0.7, "analyze": 0.7, "evaluate": 0.7, "assess": 0.7,
            "track": 0.7, "document": 0.7, "record": 0.7, "maintain": 0.6,
            "prepare": 0.6, "generate": 0.6, "update": 0.6, "calculate": 0.6,
            "process": 0.6, "recalculate": 0.7, "monitor": 0.65,  # Moved up from low
            "revoke": 0.7, "disable": 0.7, "remove": 0.7, "limit": 0.7,
            "restrict": 0.7, "age": 0.7, "receive": 0.65, "resolve": 0.7
        },
        "low_strength_verbs": {
            "look": 0.2, "observe": 0.3, "view": 0.2, "consider": 0.2,
            "watch": 0.2, "note": 0.3, "see": 0.1, "handle": 0.2,
            "manage": 0.3, "coordinate": 0.3, "facilitate": 0.2, "oversee": 0.4,
            "run": 0.3, "perform": 0.3, "address": 0.2, "raise": 0.4
        },
        "problematic_verbs": {
            "use": 0.1, "launch": 0.1, "set": 0.1, "meet": 0.1, "include": 0.1,
            "have": 0.1, "be": 0.1, "exist": 0.1, "contain": 0.1, "sound": 0.1,
            "store": 0.2, "engage": 0.2, "schedule": 0.2, "used": 0.1, "log": 0.3
        }
    }

    # Get categories from config if available
    if config and "WHAT" in config.get("elements", {}):
        config_categories = config["elements"]["WHAT"]

        # Merge with defaults for each category
        for category in default_categories:
            if category in config_categories:
                if isinstance(config_categories[category], dict):
                    # The config provides a dict with verb->score mappings
                    default_categories[category].update(config_categories[category])
                elif isinstance(config_categories[category], list):
                    # The config provides a list of verbs (assign default scores)
                    default_score = 0.8 if category == "high_strength_verbs" else 0.6 if category == "medium_strength_verbs" else 0.3
                    for verb in config_categories[category]:
                        if isinstance(verb, dict):
                            # Handle {"verb": score} format
                            for v, score in verb.items():
                                default_categories[category][v] = score
                        else:
                            # Handle string format
                            default_categories[category][verb] = default_score

    # Integrate existing_keywords if provided
    if existing_keywords:
        for keyword in existing_keywords:
            # Skip multi-word phrases
            if " " not in keyword.lower():
                # If not already in any category, add to medium_strength_verbs
                if (keyword.lower() not in default_categories["high_strength_verbs"] and
                        keyword.lower() not in default_categories["medium_strength_verbs"] and
                        keyword.lower() not in default_categories["low_strength_verbs"] and
                        keyword.lower() not in default_categories["problematic_verbs"]):
                    default_categories["medium_strength_verbs"][keyword.lower()] = 0.6

    return default_categories


def get_config_value(config: Optional[Dict], key: str, default_value: Any) -> Any:
    """
    Helper function to safely get configuration values with defaults

    Args:
        config: Configuration dictionary or None
        key: Configuration key to look for
        default_value: Default value if config is None or key not found

    Returns:
        Configuration value or default
    """
    if not config:
        return default_value

    # Check in WHAT element config
    what_config = config.get("elements", {}).get("WHAT", {})
    if key in what_config:
        return what_config[key]

    # Check in general config
    return config.get(key, default_value)


def extract_control_verb_candidates(doc, verb_categories: Dict, config: Optional[Dict]) -> List[Dict]:
    """
    Extract action candidates from control verbs with confidence scoring.
    Enhanced to better handle passive voice, compound verbs, and coordinated actions.

    Args:
        doc: spaCy document
        verb_categories: Dictionary of verb categories with strength scores
        config: Optional configuration dictionary

    Returns:
        List of action candidate dictionaries
    """
    candidates = []

    # Flatten verb categories for easier lookup
    all_verbs = {}
    for category, verbs in verb_categories.items():
        all_verbs.update(verbs)

    # Get debug mode from config
    debug = get_config_value(config, "debug_mode", False)

    # Lists for timing/process phrases to exclude
    timing_phrases = get_config_value(config, "timing_phrases", [
        r'on\s+an?\s+[a-z-]+\s+basis',
        r'daily|weekly|monthly|quarterly|annually|yearly',
        r'each\s+(day|week|month|quarter|year)',
        r'every\s+(day|week|month|quarter|year)',
        r'when\s+(necessary|needed|required|appropriate)',
        r'as\s+(needed|required|appropriate)',
        r'once\s+[a-z]+',
        r'prior\s+to',
        r'subsequent\s+to',
        r'following\s+the',
        r'after\s+[a-z]+',
        r'before\s+[a-z]+',
        r'during\s+[a-z]+',
    ])

    process_phrases = get_config_value(config, "process_phrases", [
        r'has\s+processes?',
        r'have\s+processes?',
        r'are\s+used\s+for',
        r'is\s+used\s+for',
        r'are\s+stored',
        r'is\s+stored',
        r'are\s+included',
        r'is\s+included',
        r'according\s+to',
        r'based\s+on',
        r'part\s+of',
    ])

    try:
        # Find timing and process phrase spans to exclude
        excluded_spans = []
        text_lower = doc.text.lower()

        for pattern in timing_phrases:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                excluded_spans.append((match.start(), match.end()))

        for pattern in process_phrases:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                excluded_spans.append((match.start(), match.end()))

        # Process each sentence separately to better handle compound actions
        for sent in doc.sents:
            # Find all verbs in this sentence
            sent_verbs = []
            for token in sent:
                # Add main verbs
                if token.pos_ == "VERB" and token.lemma_.lower() in all_verbs:
                    # Skip if this verb is part of an excluded span
                    if any(token.idx >= start and token.idx < end for start, end in excluded_spans):
                        continue

                    # Skip auxiliary verbs (but keep them for passive construction)
                    if token.dep_ == "aux" and not (
                            token.lemma_ == "be" and any(child.tag_ == "VBN" for child in token.children)):
                        continue

                    sent_verbs.append(token)

            # Process standard active voice verbs
            for verb in sent_verbs:
                # Skip if part of auxiliary construction (will be processed with main verb)
                if verb.dep_ == "aux":
                    continue

                # Skip "to be" verbs unless they're part of a passive construction
                if verb.lemma_ == "be" and not any(child.tag_ == "VBN" for child in verb.children):
                    # Special case: "be responsible for" is a valid control action
                    responsible_child = False
                    for child in verb.children:
                        if child.lemma_ == "responsible" or child.lemma_ == "accountable":
                            responsible_child = True
                            break

                    if not responsible_child:
                        continue

                # Get verb category and strength
                verb_strength = get_verb_strength(verb.lemma_.lower(), verb_categories)
                verb_category = get_verb_category(verb.lemma_.lower(), verb_categories)

                # Check if passive (consider both explicit passive and copular + adjective patterns)
                is_passive = is_passive_construction(verb)

                try:
                    # Build the verb phrase (now handles passive better)
                    verb_phrase, where_info = build_verb_phrase(verb, doc, is_passive, config)

                    # Skip if the verb phrase is empty or too short
                    if not verb_phrase or len(verb_phrase.split()) < 2:
                        continue

                    # Get the subject (who is performing the action)
                    subject = get_subject(verb)

                    # For passive voice, also identify what's being acted upon (passive subject)
                    passive_subject = None
                    if is_passive:
                        for child in verb.children:
                            if child.dep_ == "nsubjpass":
                                passive_subject = " ".join(t.text for t in child.subtree)
                                break

                    # Assess object specificity
                    object_specificity = assess_object_specificity(verb, config)

                    # Assess phrase completeness
                    completeness = assess_phrase_completeness(verb_phrase)

                    # Calculate confidence score with optional debug info
                    confidence = calculate_verb_confidence(
                        verb,
                        verb_strength,
                        is_passive,
                        subject is not None,
                        object_specificity,
                        completeness,
                        verb_category,
                        where_info is not None,
                        debug=debug
                    )

                    # Create candidate
                    candidate = {
                        "verb": verb.text,
                        "verb_lemma": verb.lemma_.lower(),
                        "full_phrase": verb_phrase,
                        "subject": subject,
                        "passive_subject": passive_subject,  # Add the passive subject for context
                        "is_passive": is_passive,
                        "strength": verb_strength,
                        "strength_category": verb_category,
                        "object_specificity": object_specificity,
                        "completeness": completeness,
                        "score": confidence,
                        "position": verb.i,
                        "detection_method": "dependency_parsing",
                        "is_core_action": is_core_control_action(verb, verb.lemma_.lower(), verb_category)
                    }

                    # Boost score for passive verbs that are common in controls
                    if is_passive and verb.lemma_.lower() in ["review", "test", "approve", "verify", "reconcile",
                                                              "monitor", "check"]:
                        candidate["score"] = min(1.0, candidate["score"] * 1.2)
                        if debug:
                            print(f"Applied passive control verb boost to: {verb.text}")

                    # Add WHERE component if detected
                    if where_info:
                        candidate["has_where_component"] = True
                        candidate["where_text"] = where_info["text"]
                        candidate["where_type"] = where_info["type"]
                    else:
                        candidate["has_where_component"] = False

                    candidates.append(candidate)
                except Exception as e:
                    if debug:
                        print(f"Error processing verb {verb.text}: {str(e)}")
                    continue

            # Handle compound verbs and coordinated actions (verbs with "and", "or" between them)
            compound_verbs = find_compound_verbs(sent)
            for verb_group in compound_verbs:
                # Skip if empty
                if not verb_group:
                    continue

                # Use the first verb as reference
                main_verb = verb_group[0]

                # Skip if not in our verb categories
                if main_verb.lemma_.lower() not in all_verbs:
                    continue

                # Get common properties
                is_passive = is_passive_construction(main_verb)
                verb_strength = get_verb_strength(main_verb.lemma_.lower(), verb_categories)
                verb_category = get_verb_category(main_verb.lemma_.lower(), verb_categories)

                # Get the subject (active) or the object being acted upon (passive)
                subject = get_subject(main_verb)

                # For all verbs in the group, create a candidate
                for i, verb in enumerate(verb_group):
                    # Skip if not a verb in our categories
                    if verb.lemma_.lower() not in all_verbs:
                        continue

                    try:
                        # Adjust verb strength based on position in group
                        local_verb_strength = get_verb_strength(verb.lemma_.lower(), verb_categories)

                        # Build verb phrase for this specific verb
                        verb_phrase, where_info = build_verb_phrase(verb, doc, is_passive, config)

                        # Skip if the verb phrase is empty or too short
                        if not verb_phrase or len(verb_phrase.split()) < 2:
                            continue

                        # Calculate adjusted score
                        position_penalty = 0.95 ** i  # Slight penalty for later verbs in a chain

                        # Get object specificity and completeness
                        object_specificity = assess_object_specificity(verb, config)
                        completeness = assess_phrase_completeness(verb_phrase)

                        # Calculate confidence score
                        confidence = calculate_verb_confidence(
                            verb,
                            local_verb_strength,
                            is_passive,
                            subject is not None,
                            object_specificity,
                            completeness,
                            verb_category,
                            where_info is not None,
                            debug=debug
                        ) * position_penalty

                        # Create candidate
                        candidate = {
                            "verb": verb.text,
                            "verb_lemma": verb.lemma_.lower(),
                            "full_phrase": verb_phrase,
                            "subject": subject,
                            "is_passive": is_passive,
                            "strength": local_verb_strength,
                            "strength_category": verb_category,
                            "object_specificity": object_specificity,
                            "completeness": completeness,
                            "score": confidence,
                            "position": verb.i,
                            "detection_method": "compound_action",
                            "is_core_action": is_core_control_action(verb, verb.lemma_.lower(), verb_category),
                            "is_part_of_chain": True,
                            "chain_position": i
                        }

                        # Boost score for passive verbs that are common in controls
                        if is_passive and verb.lemma_.lower() in ["review", "test", "approve", "verify", "reconcile",
                                                                  "monitor", "check"]:
                            candidate["score"] = min(1.0, candidate["score"] * 1.2)

                        # Add WHERE component if detected
                        if where_info:
                            candidate["has_where_component"] = True
                            candidate["where_text"] = where_info["text"]
                            candidate["where_type"] = where_info["type"]
                        else:
                            candidate["has_where_component"] = False

                        candidates.append(candidate)
                    except Exception as e:
                        if debug:
                            print(f"Error processing compound verb {verb.text}: {str(e)}")
                        continue

    except Exception as e:
        print(f"Error in control verb extraction: {str(e)}")
        # Still return any candidates we found before the error

    return candidates


def find_compound_verbs(sent) -> List[List]:
    """
    Find compound verbs and coordinated actions in a sentence.

    Args:
        sent: spaCy sentence

    Returns:
        List of verb groups (each group is a list of tokens)
    """
    verb_groups = []
    processed_verbs = set()

    # First find verbs that are part of a coordination
    for token in sent:
        # Skip if already processed
        if token.i in processed_verbs:
            continue

        # Look for verbs with conj dependency
        if token.pos_ == "VERB":
            # Start a group with this verb
            group = [token]
            processed_verbs.add(token.i)

            # Find all coordinated verbs
            for other_token in sent:
                if other_token.i != token.i and other_token.pos_ == "VERB":
                    # Direct coordination (verb and verb)
                    if other_token.head == token and other_token.dep_ == "conj":
                        group.append(other_token)
                        processed_verbs.add(other_token.i)
                    # Nested coordination (verb, verb and verb)
                    elif (other_token.head.head == token and
                          other_token.head.dep_ == "conj" and
                          other_token.dep_ == "conj"):
                        group.append(other_token)
                        processed_verbs.add(other_token.i)

            # Only add if we found a group with multiple verbs
            if len(group) > 1:
                # Sort by position in sentence
                group.sort(key=lambda x: x.i)
                verb_groups.append(group)

    # Now find passive verb chains (are + past participle)
    for token in sent:
        # Skip if already processed
        if token.i in processed_verbs:
            continue

        # Look for auxiliary "be" verbs
        if token.lemma_ == "be" and token.dep_ in ["ROOT", "aux"]:
            past_participles = []

            # Find associated past participles
            for child in token.children:
                if child.tag_ == "VBN" and child.pos_ == "VERB":
                    past_participles.append(child)
                    processed_verbs.add(child.i)

                    # Check for coordinated participles
                    for grandchild in child.children:
                        if grandchild.dep_ == "conj" and grandchild.tag_ == "VBN":
                            past_participles.append(grandchild)
                            processed_verbs.add(grandchild.i)

            # Only add if we found participles
            if past_participles:
                # Sort by position
                past_participles.sort(key=lambda x: x.i)
                verb_groups.append(past_participles)

    return verb_groups


def build_verb_phrase(token, doc, is_passive=False, config: Optional[Dict] = None) -> Tuple[str, Optional[Dict]]:
    """
    Extract the complete verb phrase with enhanced handling for passive voice
    and WHERE component detection.

    Args:
        token: The verb token from spaCy
        doc: The spaCy document
        is_passive: Whether this is a passive construction
        config: Optional configuration dictionary

    Returns:
        Tuple of (verb_phrase, where_component_info)
    """
    try:
        # Start with the verb itself
        phrase_tokens = [token]
        where_component = None

        # Get verb categories for problematic verbs
        problematic_verbs = []
        if config and "elements" in config and "WHAT" in config["elements"]:
            problematic_verbs = config["elements"]["WHAT"].get("problematic_verbs", {}).keys()
        else:
            problematic_verbs = ["use", "launch", "set", "meet", "include", "have", "be",
                                 "exist", "contain", "sound", "store", "engage", "schedule"]

        # Systems and locations for WHERE detection
        where_systems = get_config_value(config, "where_systems", [
            "system", "application", "database", "server", "platform", "sharepoint",
            "erp", "sap", "oracle", "repository", "file", "folder", "directory"
        ])

        where_locations = get_config_value(config, "where_locations", [
            "site", "location", "storage", "share", "repository", "archive", "folder"
        ])

        # Special handling for passive voice
        if is_passive:
            # For passive, we need to find the auxiliary "be" verb and the verb itself
            aux_be = None

            # Check if this is the main verb or the auxiliary
            if token.lemma_ == "be":
                # This is the auxiliary "be" - find the main verb (past participle)
                aux_be = token
                main_verb = None

                for child in token.children:
                    if child.tag_ == "VBN" and child.pos_ == "VERB":
                        main_verb = child
                        break

                if main_verb:
                    # Replace token with the main verb for further processing
                    phrase_tokens = [main_verb]

                    # Find the passive subject (what's being acted upon)
                    passive_subject = None
                    for child in token.children:
                        if child.dep_ == "nsubjpass":
                            # Capture the complete noun phrase
                            subtree_tokens = list(child.subtree)
                            subtree_tokens.sort(key=lambda x: x.i)
                            passive_subject = subtree_tokens
                            break

                    # Add the passive subject to the beginning
                    if passive_subject:
                        phrase_tokens = passive_subject + [aux_be] + phrase_tokens
            else:
                # This is the main verb (past participle) - find the auxiliary
                for ancestor in token.ancestors:
                    if ancestor.lemma_ == "be" and ancestor.dep_ in ["ROOT", "aux"]:
                        aux_be = ancestor
                        break

                if aux_be:
                    # Find the passive subject (what's being acted upon)
                    passive_subject = None
                    for child in aux_be.children:
                        if child.dep_ == "nsubjpass":
                            # Capture the complete noun phrase
                            subtree_tokens = list(child.subtree)
                            subtree_tokens.sort(key=lambda x: x.i)
                            passive_subject = subtree_tokens
                            break

                    # Add the passive subject and auxiliary to the beginning
                    if passive_subject:
                        phrase_tokens = passive_subject + [aux_be] + phrase_tokens

        # For problematic verbs, limit the phrase to direct objects only
        is_problematic = token.lemma_.lower() in problematic_verbs

        # Find any direct objects and prepositional phrases
        for child in token.children:
            # Always include direct objects
            if child.dep_ in ["dobj", "iobj", "attr", "oprd", "pobj"]:
                # Include the token and its children (to get the complete phrase)
                subtree = list(child.subtree)
                phrase_tokens.extend(subtree)

            # For non-problematic verbs, include more dependencies
            elif not is_problematic:
                # Include adverbial clauses that specify manner
                if child.dep_ == "advmod" and child.pos_ == "ADV":
                    phrase_tokens.append(child)

                # Include adverbial clauses (how something is done)
                elif child.dep_ == "advcl":
                    subtree = list(child.subtree)
                    phrase_tokens.extend(subtree)

            # Handle prepositional phrases (potential WHERE components)
            if child.dep_ == "prep" and child.text.lower() in ["in", "on", "with", "using", "through", "via"]:
                # Check for what follows the preposition
                prep_obj = None
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        prep_obj = grandchild
                        break

                if prep_obj:
                    # Create the prep phrase text
                    prep_phrase_tokens = list(child.subtree)
                    prep_phrase_tokens.sort(key=lambda x: x.i)
                    prep_phrase = " ".join(t.text for t in prep_phrase_tokens)

                    # Check if this is a WHERE component
                    is_where = False
                    where_type = None

                    # Check for system indicators
                    if (prep_obj.lemma_.lower() in where_systems or
                            any(term in prep_phrase.lower() for term in where_systems)):
                        is_where = True
                        where_type = "system"

                    # Check for location indicators
                    elif (prep_obj.lemma_.lower() in where_locations or
                          any(term in prep_phrase.lower() for term in where_locations)):
                        is_where = True
                        where_type = "location"

                    # If WHERE component detected, create info dict
                    if is_where:
                        where_component = {
                            "text": prep_phrase,
                            "type": where_type,
                            "token": child
                        }

                    # Include in phrase tokens regardless of WHERE status
                    # This ensures the preposition and its object are included in the phrase
                    phrase_tokens.extend(prep_phrase_tokens)

        # For passive constructions, also consider clauses that follow the verb
        if is_passive:
            # Check for modifiers and complements after the verb
            next_token_idx = token.i + 1
            if next_token_idx < len(doc):
                next_token = doc[next_token_idx]

                # If the next token is a preposition, include its phrase
                if next_token.dep_ == "prep" and next_token.text.lower() in ["in", "on", "with", "using", "through",
                                                                             "via"]:
                    prep_tokens = list(next_token.subtree)
                    phrase_tokens.extend(prep_tokens)

                # Include purpose clauses ("to ensure", "to verify", etc.)
                elif next_token.text.lower() == "to" and next_token.i + 1 < len(doc):
                    purpose_verb = doc[next_token.i + 1]
                    if purpose_verb.pos_ == "VERB" and purpose_verb.lemma_.lower() in ["ensure", "verify", "confirm",
                                                                                       "validate"]:
                        purpose_tokens = list(next_token.subtree)
                        phrase_tokens.extend(purpose_tokens)

        # Sort tokens by their position in the original text
        phrase_tokens.sort(key=lambda x: x.i)

        # Combine into a phrase
        verb_phrase = " ".join(token.text for token in phrase_tokens)

        return verb_phrase, where_component

    except Exception as e:
        print(f"Error building verb phrase for '{token.text}': {str(e)}")
        # Return just the verb token as fallback
        return token.text, None


def is_passive_construction(verb_token) -> bool:
    """
    Determine if a verb is in passive voice with enhanced detection for
    various passive constructions.

    Args:
        verb_token: The verb token to check

    Returns:
        Boolean indicating if the verb is in passive voice
    """
    try:
        # Method 1: Check for explicit passive subject
        if any(token.dep_ == "nsubjpass" for token in verb_token.children):
            return True

        # Method 2: Check for passive auxiliary (be + past participle)
        if verb_token.tag_ == "VBN":  # Past participle
            # Look for form of "be" as an auxiliary
            for ancestor in verb_token.ancestors:
                if ancestor.lemma_ == "be" and ancestor.pos_ == "AUX":
                    return True

        # Method 3: If this IS the "be" verb, check if it has a past participle child
        if verb_token.lemma_ == "be":
            for child in verb_token.children:
                if child.tag_ == "VBN" and child.pos_ == "VERB":
                    return True

        # Method 4: Look for passive pattern in surrounding context
        # Check tokens before this verb for forms of "be" or "get"
        doc = verb_token.doc
        for i in range(max(0, verb_token.i - 3), verb_token.i):
            if i < len(doc) and doc[i].lemma_ in ["be", "get"] and verb_token.tag_ == "VBN":
                return True

        return False
    except Exception:
        # If error occurs, assume active voice
        return False


def is_core_control_action(verb_token, verb_lemma: str, verb_category: str) -> bool:
    """
    Determine if a verb represents a core control action rather than a supporting process action.
    Enhanced to better recognize control verbs in passive constructions.

    Args:
        verb_token: The verb token
        verb_lemma: The lemmatized verb
        verb_category: The verb category

    Returns:
        Boolean indicating if this is a core control action
    """
    try:
        # If it's high strength, it's likely a core action
        if verb_category == "high_strength_verbs":
            # Special handling for passive voice
            if is_passive_construction(verb_token):
                # Passive high-strength verbs are almost always core actions (reviewed, approved, etc.)
                return True

            # For active voice, check if the verb has an object
            has_object = any(child.dep_ in ["dobj", "pobj", "attr"] for child in verb_token.children)
            if not has_object:
                return False
            return True

        # If it's a known problematic verb, it's definitely not a core action
        if verb_category == "problematic_verbs":
            return False

        # Special case for passive voice verbs that are common in controls
        passive_control_verbs = ["test", "review", "approve", "verify", "reconcile", "monitor",
                                 "check", "validate", "examine", "analyze", "evaluate"]
        if is_passive_construction(verb_token) and verb_lemma in passive_control_verbs:
            return True

        # If it's the root verb of the sentence, it's more likely to be a core action
        if verb_token.dep_ == "ROOT":
            # Unless it's passive and a problematic verb
            if verb_category == "problematic_verbs" and is_passive_construction(verb_token):
                return False
            # Check if it has an object
            has_object = any(child.dep_ in ["dobj", "pobj", "attr"] for child in verb_token.children)
            if not has_object:
                return False
            return True

        # Supporting actions are often in subordinate clauses
        if verb_token.dep_ in ["advcl", "relcl", "ccomp"]:
            return False

        # Default to True for medium strength verbs
        if verb_category == "medium_strength_verbs":
            return True

        # Default to False for uncertain cases
        return False
    except Exception:
        # If error occurs, assume not a core action
        return False

def get_subject(verb_token) -> Optional[str]:
    """
    Find the subject of a verb

    Args:
        verb_token: The verb token to find the subject for

    Returns:
        String containing the subject text or None if not found
    """
    try:
        for token in verb_token.children:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                # Return the complete noun phrase, including any modifiers
                return " ".join(t.text for t in token.subtree)

        # If no direct subject found, look for a governing verb's subject (for compound verbs)
        if verb_token.dep_ == "xcomp" and verb_token.head.pos_ == "VERB":
            return get_subject(verb_token.head)

        return None
    except Exception:
        return None


def assess_object_specificity(token, config: Optional[Dict]) -> float:
    """
    Assess how specific the object of a verb is

    Args:
        token: The verb token
        config: Optional configuration dictionary

    Returns:
        Float score between 0.0 and 1.0 indicating specificity
    """
    try:
        # Find direct objects
        objects = []
        for child in token.children:
            if child.dep_ in ["dobj", "pobj", "attr"]:
                # Get the complete subtree
                objects.extend(list(child.subtree))

        if not objects:
            return 0.0

        # Get technical terms from config
        tech_terms = get_config_value(config, "technical_terms", [
            "system", "application", "database", "server", "protocol", "interface",
            "module", "component", "api", "configuration", "parameter", "threshold"
        ])

        # Get vague object terms from config
        vague_object_terms = get_config_value(config, "vague_object_terms", [
            "item", "thing", "stuff", "issue", "matter", "situation", "exception",
            "information", "data", "content", "material", "object"
        ])

        # Assess specificity based on length, modifiers, and noun types
        score = min(1.0, len(objects) / 5.0)  # Longer object phrases tend to be more specific

        # Check for modifiers that make objects more specific
        specificity_indicators = ["specific", "certain", "particular", "defined", "exact"]
        if any(token.lemma_.lower() in specificity_indicators for token in objects):
            score += 0.2

        # Check for numeric modifiers (usually make things more specific)
        if any(token.pos_ == "NUM" for token in objects):
            score += 0.2

        # Check for proper nouns (usually more specific)
        if any(token.pos_ == "PROPN" for token in objects):
            score += 0.2

        # Check for technical terms (usually more specific)
        if any(token.lemma_.lower() in tech_terms for token in objects):
            score += 0.1

        # Check for vague generic terms that reduce specificity
        if any(token.lemma_.lower() in vague_object_terms for token in objects):
            score -= 0.3  # Significant penalty for vague objects

        return min(1.0, max(0.0, score))  # Ensure score is between 0 and 1
    except Exception:
        # If error occurs, return moderate score
        return 0.5


def assess_phrase_completeness(verb_phrase: str) -> float:
    """
    Assess how complete a verb phrase is as a control action

    Args:
        verb_phrase: The verb phrase to assess

    Returns:
        Float score between 0.0 and 1.0 indicating completeness
    """
    # Split into words
    words = verb_phrase.split()

    # Very short phrases are likely incomplete
    if len(words) < 2:
        return 0.4

    # Phrases with just prepositions at the end are incomplete
    if words[-1].lower() in ["to", "for", "with", "by", "on", "at", "in"]:
        return 0.5

    # Phrases ending with process indicators are incomplete
    process_endings = ["according", "based", "part", "accordance"]
    if words[-1].lower() in process_endings:
        return 0.5

    # Long enough phrases with a verb and object are likely complete
    if len(words) >= 3:
        return 1.0

    # Default completeness
    return 0.8


def calculate_verb_confidence(token, verb_strength: float, is_passive: bool, has_subject: bool,
                              object_specificity: float, completeness: float, verb_category: str,
                              has_where_component: bool, debug: bool = False) -> float:
    """
    Calculate confidence score for a verb as a control action with tracing option

    Args:
        token: The verb token
        verb_strength: Base strength score of the verb
        is_passive: Whether the verb is in passive voice
        has_subject: Whether the verb has a subject
        object_specificity: Score for object specificity
        completeness: Score for phrase completeness
        verb_category: Category of the verb
        has_where_component: Whether the verb has a WHERE component
        debug: Whether to print debug tracing information

    Returns:
        Float confidence score between 0.0 and 1.0
    """
    # Start with verb strength as base confidence
    confidence = verb_strength

    if debug:
        print(f"\nConfidence calculation for '{token.text}':")
        print(f"  Base verb strength: {confidence:.2f}")

    # Adjust for voice - higher penalty for passive problematic verbs
    if is_passive:
        if verb_category == "problematic_verbs":
            confidence *= 0.2  # Severe penalty for passive problematic verbs
            if debug:
                print(f"  After passive problematic penalty: {confidence:.2f}")
        else:
            confidence *= 0.4  # Increased penalty for passive voice
            if debug:
                print(f"  After passive voice penalty: {confidence:.2f}")

    # Adjust for subject clarity
    if has_subject:
        confidence *= 1.1
        if debug:
            print(f"  After subject clarity boost: {confidence:.2f}")

    # Adjust for position (normalize by text length)
    position_factor = 1.0 - (token.i / len(token.doc)) * 0.2
    confidence *= position_factor
    if debug:
        print(f"  After position adjustment: {confidence:.2f}")

    # Adjust for object specificity
    confidence *= (1.0 + object_specificity * 0.2)
    if debug:
        print(f"  After object specificity: {confidence:.2f}")

    # Adjust for phrase completeness
    confidence *= completeness
    if debug:
        print(f"  After completeness adjustment: {confidence:.2f}")

    # Apply WHERE component boost
    if has_where_component:
        confidence *= 1.1
        if debug:
            print(f"  After WHERE component boost: {confidence:.2f}")

    # Apply ROOT boost - primary verbs in the sentence are more likely to be control actions
    if token.dep_ == "ROOT":
        confidence *= 1.2
        if debug:
            print(f"  After ROOT boost: {confidence:.2f}")

    # Cap confidence at 1.0
    confidence = min(1.0, confidence)
    if debug:
        print(f"  Final confidence: {confidence:.2f}")

    return confidence


def get_verb_strength(verb_lemma: str, verb_categories: Dict) -> float:
    """
    Get the strength score for a verb

    Args:
        verb_lemma: The verb lemma to check
        verb_categories: Dictionary of verb categories

    Returns:
        Float strength score between 0.0 and 1.0
    """
    # Check each category in order of strength
    if verb_lemma in verb_categories["high_strength_verbs"]:
        return verb_categories["high_strength_verbs"][verb_lemma]
    elif verb_lemma in verb_categories["medium_strength_verbs"]:
        return verb_categories["medium_strength_verbs"][verb_lemma]
    elif verb_lemma in verb_categories["low_strength_verbs"]:
        return verb_categories["low_strength_verbs"][verb_lemma]
    elif verb_lemma in verb_categories["problematic_verbs"]:
        return verb_categories["problematic_verbs"][verb_lemma]
    else:
        # Default to medium strength
        return 0.5


def get_verb_category(verb_lemma: str, verb_categories: Dict) -> str:
    """
    Get the category for a verb

    Args:
        verb_lemma: The verb lemma to check
        verb_categories: Dictionary of verb categories

    Returns:
        String category name
    """
    if verb_lemma in verb_categories["high_strength_verbs"]:
        return "high_strength_verbs"
    elif verb_lemma in verb_categories["medium_strength_verbs"]:
        return "medium_strength_verbs"
    elif verb_lemma in verb_categories["low_strength_verbs"]:
        return "low_strength_verbs"
    elif verb_lemma in verb_categories["problematic_verbs"]:
        return "problematic_verbs"
    else:
        return "unknown"


def is_core_control_action(verb_token, verb_lemma: str, verb_category: str) -> bool:
    """
    Determine if a verb represents a core control action rather than a supporting process action

    Args:
        verb_token: The verb token
        verb_lemma: The lemmatized verb
        verb_category: The verb category

    Returns:
        Boolean indicating if this is a core control action
    """
    try:
        # If it's high strength, it's likely a core action
        if verb_category == "high_strength_verbs":
            # Check if the verb has an object - verbs without objects are less likely to be core actions
            has_object = any(child.dep_ in ["dobj", "pobj", "attr"] for child in verb_token.children)
            if not has_object:
                return False
            return True

        # If it's a known problematic verb, it's definitely not a core action
        if verb_category == "problematic_verbs":
            return False

        # If it's the root verb of the sentence, it's more likely to be a core action
        if verb_token.dep_ == "ROOT":
            # Unless it's passive and a problematic verb
            if verb_category == "problematic_verbs" and is_passive_construction(verb_token):
                return False
            # Check if it has an object
            has_object = any(child.dep_ in ["dobj", "pobj", "attr"] for child in verb_token.children)
            if not has_object:
                return False
            return True

        # Supporting actions are often in subordinate clauses
        if verb_token.dep_ in ["advcl", "relcl", "ccomp"]:
            return False

        # Default to True for medium strength verbs
        if verb_category == "medium_strength_verbs":
            return True

        # Default to False for uncertain cases
        return False
    except Exception:
        # If error occurs, assume not a core action
        return False


def extract_action_patterns(text: str, doc, config: Optional[Dict]) -> List[Dict]:
    """
    Fallback method: Extract actions using regex patterns

    Args:
        text: Raw control description text
        doc: spaCy document
        config: Optional configuration dictionary

    Returns:
        List of action candidates found using patterns
    """
    action_candidates = []
    text_lower = text.lower()

    try:
        # Control verb patterns
        control_patterns = [
            (r'(notify|alert|inform)\s+([a-z\s]+)', 0.75),  # notify management, alert team
            (r'(age|categorize)\s+([a-z\s]+)', 0.7),  # age receivables, categorize items
            (r'(receive|collect)\s+([a-z\s]+)', 0.6),  # receive sign-offs, collect approvals
            (r'(limit|restrict)\s+([a-z\s]+)\s+to\s+([a-z\s]+)', 0.75),  # limit access to authorized
            (r'(route|escalate|forward)\s+([a-z\s]+)\s+to\s+([a-z\s]+)', 0.75)  # route exceptions to support
        ]

        # Special case controls common in security/IT controls
        special_cases = [
            (r'(access|permission)s?\s+(?:are|is)\s+(limited|restricted)\s+to', "limit access to authorized personnel",
             0.8),
            (r'(batch\s+job)s?\s+(?:are|is)\s+(scheduled|run)', "schedule batch jobs", 0.7),
            (r'(exception)s?\s+(?:are|is)\s+(routed|sent)\s+to', "route exceptions", 0.75),
            (r'(notification)s?\s+(?:are|is)\s+(sent|delivered)\s+to', "send notifications", 0.7),
            (r'(system)\s+automatically\s+(ages)', "age receivables automatically", 0.8),
            (r'items\s+are\s+(inventoried|counted)', "inventory items", 0.7)
        ]

        # Apply control patterns
        for pattern, confidence in control_patterns:
            for match in re.finditer(pattern, text_lower):
                verb = match.group(1)
                full_phrase = match.group(0)

                # Check for WHERE component
                where_info = detect_where_component(full_phrase, config)

                action_candidates.append({
                    "verb": verb,
                    "verb_lemma": verb,
                    "full_phrase": full_phrase,
                    "subject": None,
                    "is_passive": False,
                    "strength": confidence,
                    "strength_category": "medium",
                    "object_specificity": 0.6,
                    "completeness": 0.8,
                    "score": confidence,
                    "position": match.start(),
                    "detection_method": "pattern_matching",  # Standardized name
                    "is_core_action": True,
                    "has_where_component": where_info is not None,
                    "where_text": where_info["text"] if where_info else None,
                    "where_type": where_info["type"] if where_info else None
                })

        # Apply special case patterns
        for pattern, replacement, confidence in special_cases:
            if re.search(pattern, text_lower):
                verb = replacement.split()[0]

                # Check for WHERE component
                match = re.search(pattern, text_lower)
                if match:
                    context = text_lower[max(0, match.start() - 20):min(len(text_lower), match.end() + 20)]
                    where_info = detect_where_component(context, config)
                else:
                    where_info = None

                action_candidates.append({
                    "verb": verb,
                    "verb_lemma": verb,
                    "full_phrase": replacement,
                    "subject": None,
                    "is_passive": False,
                    "strength": confidence,
                    "strength_category": "medium",
                    "object_specificity": 0.6,
                    "completeness": 0.9,
                    "score": confidence,
                    "position": 0,  # Dummy position
                    "detection_method": "pattern_matching",  # Standardized name
                    "is_core_action": True,
                    "has_where_component": where_info is not None,
                    "where_text": where_info["text"] if where_info else None,
                    "where_type": where_info["type"] if where_info else None
                })
    except Exception as e:
        print(f"Error in pattern extraction: {str(e)}")
        # Return any candidates we found before the error

    return action_candidates


def extract_from_noun_chunks(doc, nlp, config: Optional[Dict]) -> List[Dict]:
    """
    Fallback method: Extract actions from noun phrases

    Args:
        doc: spaCy document
        nlp: spaCy NLP model
        config: Optional configuration dictionary

    Returns:
        List of action candidates derived from noun chunks
    """
    action_candidates = []

    try:
        # Control activity nouns that often indicate actions
        control_nouns = {
            "review": 0.8,
            "approval": 0.8,
            "validation": 0.8,
            "verification": 0.8,
            "reconciliation": 0.8,
            "assessment": 0.7,
            "monitoring": 0.7,
            "inspection": 0.7,
            "audit": 0.8,
            "analysis": 0.7,
            "confirmation": 0.7,
            "evaluation": 0.7,
            "check": 0.7,
            "authorization": 0.8,
            "testing": 0.7
        }

        # Mapping from noun to verb form
        verb_mapping = {
            "review": "review",
            "approval": "approve",
            "validation": "validate",
            "verification": "verify",
            "reconciliation": "reconcile",
            "assessment": "assess",
            "monitoring": "monitor",
            "inspection": "inspect",
            "audit": "audit",
            "analysis": "analyze",
            "confirmation": "confirm",
            "evaluation": "evaluate",
            "check": "check",
            "authorization": "authorize",
            "testing": "test"
        }

        # Examine noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk) > 1:  # Ignore single word chunks
                head = chunk.root
                head_text = head.text.lower()

                # Check if the head is a control-related noun
                if head_text in control_nouns:
                    verb = verb_mapping.get(head_text, "perform")

                    # Get modifiers to form an action phrase
                    modifiers = [t.text for t in chunk if t.i != head.i]

                    # Construct the phrase: verb + modifiers
                    if modifiers:
                        modified_phrase = " ".join([verb] + modifiers)
                    else:
                        modified_phrase = f"{verb} {head_text}"

                    # Check for WHERE component
                    chunk_text = chunk.text
                    context = doc.text[max(0, chunk.start_char - 20):min(len(doc.text), chunk.end_char + 20)]
                    where_info = detect_where_component(context, config)

                    action_candidates.append({
                        "verb": verb,
                        "verb_lemma": verb,
                        "full_phrase": modified_phrase,
                        "subject": None,
                        "is_passive": False,
                        "strength": control_nouns[head_text],
                        "strength_category": "medium",
                        "object_specificity": 0.5,
                        "completeness": 0.8,
                        "score": control_nouns[head_text] * 0.8,  # Lower confidence for noun-derived actions
                        "position": chunk.start,
                        "detection_method": "noun_chunk_analysis",  # Standardized name
                        "is_core_action": True,
                        "has_where_component": where_info is not None,
                        "where_text": where_info["text"] if where_info else None,
                        "where_type": where_info["type"] if where_info else None
                    })
    except Exception as e:
        print(f"Error in noun chunk extraction: {str(e)}")
        # Return any candidates we found before the error

    return action_candidates


def detect_where_component(text: str, config: Optional[Dict]) -> Optional[Dict]:
    """
    Detect and extract WHERE information from action phrases

    Args:
        text: Text to analyze for WHERE components
        config: Optional configuration dictionary

    Returns:
        Dictionary with WHERE information or None if not found
    """
    try:
        # Systems and locations for WHERE detection
        where_systems = get_config_value(config, "where_systems", [
            "system", "application", "database", "server", "platform", "sharepoint",
            "erp", "sap", "oracle", "repository", "file", "folder", "directory"
        ])

        where_locations = get_config_value(config, "where_locations", [
            "site", "location", "storage", "share", "repository", "archive", "folder"
        ])

        # System patterns
        system_patterns = [
            r'in\s+(?:the\s+)?([a-zA-Z\s]+(?:system|application|database|platform|software))',
            r'(?:using|via|through)\s+(?:the\s+)?([a-zA-Z\s]+(?:system|application|database|platform|software))',
            r'within\s+(?:the\s+)?([a-zA-Z\s]+(?:system|application|database|platform|software))'
        ]

        # Location patterns
        location_patterns = [
            r'in\s+(?:the\s+)?([a-zA-Z\s]+(?:site|location|repository|share|folder|directory))',
            r'at\s+(?:the\s+)?([a-zA-Z\s]+(?:site|location|repository|share|folder|directory))',
            r'to\s+(?:the\s+)?([a-zA-Z\s]+(?:site|location|repository|share|folder|directory))'
        ]

        # Check system patterns
        for pattern in system_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return {
                    "text": match.group(0),
                    "system_name": match.group(1),
                    "type": "system"
                }

        # Check location patterns
        for pattern in location_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return {
                    "text": match.group(0),
                    "location_name": match.group(1),
                    "type": "location"
                }

        # Generic WHERE detection - refactored to reduce code duplication
        text_lower = text.lower()

        # Helper function to check for indicators and create component info
        def check_indicators(indicators, type_name):
            for indicator in indicators:
                if indicator in text_lower:
                    # Find the context around the indicator term
                    start = text_lower.find(indicator)
                    # Look for prepositions before the indicator term
                    context_start = max(0, start - 15)
                    context = text_lower[context_start:start + len(indicator) + 5]

                    # Look for prepositions that indicate WHERE
                    if type_name == "system":
                        prepositions = ["in", "on", "within", "using", "via", "through"]
                    else:
                        prepositions = ["in", "at", "to", "from"]

                    for prep in prepositions:
                        if f" {prep} " in context or context.startswith(f"{prep} "):
                            # Find the phrase containing the preposition and indicator
                            prep_pos = context.find(f" {prep} ") if f" {prep} " in context else 0
                            where_phrase = context[prep_pos:].strip()
                            return {
                                "text": where_phrase,
                                "type": type_name
                            }
            return None

        # Check systems first
        system_component = check_indicators(where_systems, "system")
        if system_component:
            return system_component

        # Then check locations
        location_component = check_indicators(where_locations, "location")
        if location_component:
            return location_component

        # No WHERE component found
        return None
    except Exception as e:
        print(f"Error detecting WHERE component: {str(e)}")
        return None


def filter_action_candidates(candidates: List[Dict]) -> List[Dict]:
    """
    Filter and normalize action candidates

    Args:
        candidates: List of action candidates

    Returns:
        Filtered list of action candidates
    """
    if not candidates:
        return []

    try:
        # Remove duplicates based on verb and phrase
        seen_phrases = set()
        unique_candidates = []

        for candidate in candidates:
            # Create a key for deduplication
            key = (candidate["verb_lemma"], candidate["full_phrase"])

            if key not in seen_phrases:
                seen_phrases.add(key)
                unique_candidates.append(candidate)

        # Filter out low-confidence candidates
        threshold = 0.3
        filtered_candidates = [c for c in unique_candidates if c["score"] >= threshold]

        # If filtering removed everything, keep at least the best one
        if not filtered_candidates and unique_candidates:
            best_candidate = max(unique_candidates, key=lambda x: x["score"])
            filtered_candidates.append(best_candidate)

        return filtered_candidates
    except Exception as e:
        print(f"Error filtering candidates: {str(e)}")
        # Return original candidates if error
        return candidates


def determine_voice(active_count: int, passive_count: int) -> str:
    """
    Determine the dominant voice in the control

    Args:
        active_count: Number of active voice verbs
        passive_count: Number of passive voice verbs

    Returns:
        String voice classification: "active", "passive", "mixed", or "unknown"
    """
    if active_count > passive_count:
        return "active"
    elif passive_count > active_count:
        return "passive"
    elif active_count > 0 or passive_count > 0:
        return "mixed"
    else:
        return "unknown"


def calculate_final_score(primary_action: Optional[Dict], secondary_actions: List[Dict],
                          text: str, control_type_aligned: bool) -> float:
    """
    Calculate final score for the WHAT element

    Args:
        primary_action: Primary action candidate or None
        secondary_actions: List of secondary action candidates
        text: The full control description
        control_type_aligned: Whether actions align with control type

    Returns:
        Float score between 0.0 and 1.0
    """
    try:
        # If no primary action, score is 0
        if not primary_action:
            return 0.0

        # Components of final score:
        # 1. Primary action score (50%)
        # 2. Secondary actions average (20%)
        # 3. Text structure and clarity (20%)
        # 4. Control type alignment (10%)

        # Primary action component
        primary_score = primary_action["score"] * 0.5

        # Secondary actions component
        secondary_score = 0.0
        if secondary_actions:
            avg_secondary = sum(a["score"] for a in secondary_actions) / len(secondary_actions)
            secondary_score = avg_secondary * 0.2

        # Text structure component
        structure_score = 0.2  # Default structure score

        # Adjust for text length (very short or very long descriptions are penalized)
        word_count = len(text.split())
        if word_count < 10:
            structure_score *= 0.7  # Too short
        elif word_count > 100:
            structure_score *= 0.8  # Too long

        # Adjust for sentence structure
        sentences = text.split(".")
        if len(sentences) > 5:
            structure_score *= 0.9  # Too many sentences

        # Control type alignment component
        control_type_score = 0.1 if control_type_aligned else 0.0

        # Combine scores
        final_score = primary_score + secondary_score + structure_score + control_type_score

        # Cap at 1.0
        return min(1.0, final_score)
    except Exception as e:
        print(f"Error calculating final score: {str(e)}")
        # Return primary action score as fallback if available
        return primary_action["score"] if primary_action else 0.0


def generate_what_suggestions(candidates: List[Dict], text: str, control_type: Optional[str],
                              control_type_alignment: Dict, config: Optional[Dict]) -> List[str]:
    """
    Generate improvement suggestions for the WHAT element

    Args:
        candidates: List of action candidates
        text: The full control description
        control_type: Optional control type
        control_type_alignment: Control type alignment info
        config: Optional configuration dictionary

    Returns:
        List of suggestion strings
    """
    suggestions = []

    try:
        # Check if we found any actions
        if not candidates:
            suggestions.append(
                "No clear control action detected. Add a specific verb describing what the control does.")
            return suggestions

        # Get primary action (highest score)
        primary = candidates[0] if candidates else None

        # Suggest improving weak verbs
        if primary and primary["strength_category"] in ["low_strength_verbs", "problematic_verbs"]:
            verb = primary["verb"]
            alternatives = get_specific_alternatives(primary["verb_lemma"], config)
            suggestions.append(f"Replace weak verb '{verb}' with a stronger control verb like {alternatives}")

        # Suggest improving passive voice
        voice_counts = {"active": 0, "passive": 0}
        for candidate in candidates:
            if candidate["is_passive"]:
                voice_counts["passive"] += 1
            else:
                voice_counts["active"] += 1

        if voice_counts["passive"] > voice_counts["active"]:
            suggestions.append("Consider using active voice to clearly indicate who performs the control")

        # Suggest adding WHERE component if missing
        has_where = any(c.get("has_where_component", False) for c in candidates)
        if not has_where and len(text.split()) > 15:
            # Only suggest for longer controls, as shorter ones may not need WHERE
            suggestions.append("Consider specifying WHERE the control is performed (system, application, or location)")

        # Check for vague objects
        if primary and primary.get("object_specificity", 1.0) < 0.5:
            suggestions.append(f"Consider clarifying the object of '{primary['verb_lemma']}' to be more specific")

        # Suggest for multiple actions
        distinct_actions = set(c["verb_lemma"] for c in candidates)
        if len(distinct_actions) > 3:
            suggestions.append(
                "This appears to describe a process with multiple actions. Consider breaking into separate controls.")

        # Add control type alignment suggestions
        if control_type and not control_type_alignment.get("is_aligned", True):
            # If not aligned, suggest appropriate verbs
            control_type_indicators = get_config_value(config, "control_type_indicators", {})
            if control_type.lower() in control_type_indicators:
                indicators = control_type_indicators[control_type.lower()]
                # Just suggest a few examples, not the full list
                examples = indicators[:3] if len(indicators) > 3 else indicators
                suggestions.append(
                    f"This control is marked as '{control_type}' but uses action verbs that don't align. "
                    f"Consider using verbs like: {', '.join(examples)}"
                )
            # If suggested type available, mention it
            if "suggested_type" in control_type_alignment:
                suggestions.append(
                    f"Based on the actions detected, this may be a '{control_type_alignment['suggested_type']}' control "
                    f"rather than a '{control_type}' control."
                )

        return suggestions
    except Exception as e:
        print(f"Error generating suggestions: {str(e)}")
        return ["Error generating suggestions. Please review the control description manually."]


def get_specific_alternatives(verb_lemma: str, config: Optional[Dict]) -> str:
    """
    Provide specific alternative verb suggestions based on the context

    Args:
        verb_lemma: The verb lemma to get alternatives for
        config: Optional configuration dictionary

    Returns:
        String of alternative suggestions
    """
    # Get alternatives from config if available
    if config and "verb_alternatives" in config:
        if verb_lemma in config["verb_alternatives"]:
            return config["verb_alternatives"][verb_lemma]

    # Default alternatives
    alternatives = {
        "perform": "'verify', 'examine', or 'evaluate'",
        "do": "'execute', 'conduct', or 'complete'",
        "look": "'examine', 'inspect', or 'review'",
        "check": "'verify', 'validate', or 'examine'",
        "make": "'create', 'prepare', or 'develop'",
        "handle": "'process', 'resolve', or 'manage'",
        "see": "'identify', 'recognize', or 'detect'",
        "watch": "'monitor', 'observe', or 'track'",
        "address": "'resolve', 'rectify', or 'remediate'",
        "consider": "'evaluate', 'assess', or 'analyze'",
        "manage": "'administer', 'supervise', or 'oversee'",
        "review": "'examine', 'analyze', or 'evaluate'",
        "flag": "'identify', 'mark', or 'highlight'",
        "monitor": "'track', 'supervise', or 'observe'",
        "use": "'utilize', 'apply', or 'implement'",
        "launch": "'initiate', 'deploy', or 'implement'",
        "set": "'establish', 'configure', or 'specify'",
        "age": "'classify by age', 'categorize', or 'track aging of'",
        "meet": "'achieve', 'satisfy', or 'fulfill'",
        "include": "'incorporate', 'integrate', or 'contain'",
        "sound": "'establish', 'implement', or 'maintain'",
        "raise": "'escalate', 'notify', or 'alert'",
        "store": "'secure', 'maintain', or 'place'",
        "log": "'record', 'document', or 'register'",
        "schedule": "'plan', 'arrange', or 'program'"
    }

    # Add control type specific suggestions
    control_type_verbs = {
        "preventive": "'prevent', 'block', or 'restrict'",
        "detective": "'detect', 'identify', or 'monitor'",
        "corrective": "'correct', 'remediate', or 'fix'"
    }

    # If we know this is for a specific control type from context
    control_type = None
    if config and "control_type" in config:
        control_type = config.get("control_type")

    # If we have a known control type and it's a weak verb, suggest control-specific alternatives
    if control_type and control_type.lower() in control_type_verbs and verb_lemma in [
        "perform", "do", "handle", "manage", "address"
    ]:
        return control_type_verbs[control_type.lower()]

    return alternatives.get(verb_lemma, "'verify', 'approve', or 'reconcile'")


def mark_possible_standalone_controls(text: str, nlp) -> list:
    """
    Identify potential standalone controls within a description that might actually be
    describing multiple controls.

    This function detects separate control statements that might be
    better documented as individual controls instead of combined in a single description.

    Args:
        text: Control description text
        nlp: spaCy NLP model

    Returns:
        List of potential standalone control candidates with text and scores
    """
    import re

    # If text is empty or too short, return empty list
    if not text or len(text) < 20:
        return []

    candidates = []

    # Method 1: Look for numbered controls
    numbered_pattern = r'\b(\d+[\.\)]\s+[A-Z][^\.;]{10,100})'
    for match in re.finditer(numbered_pattern, text):
        candidates.append({
            "text": match.group(1).strip(),
            "score": 0.9,  # High confidence for numbered list items
            "action": "Split into separate control"
        })

    # Method 2: Look for sentences with control action verbs
    control_verbs = [
        "review", "approve", "verify", "check", "validate", "ensure", "confirm",
        "examine", "analyze", "evaluate", "assess", "monitor", "track",
        "compare", "reconcile", "match",
    ]

    doc = nlp(text)

    # Process sentences
    for sent in doc.sents:
        # Skip very short sentences
        if len(sent.text.split()) < 5:
            continue

        # Skip if already captured in numbered list
        if any(c["text"] in sent.text for c in candidates):
            continue

        # Check for control verbs at start of sentence
        starts_with_verb = False
        first_tokens = list(sent)[:3]  # Look at first few tokens

        for token in first_tokens:
            if token.lemma_.lower() in control_verbs:
                starts_with_verb = True
                break

        # If sentence starts with a control verb, it might be a separate control
        if starts_with_verb:
            candidates.append({
                "text": sent.text.strip(),
                "score": 0.75,  # Medium-high confidence
                "action": "Consider as separate control"
            })

    # Method 3: Look for explicit control statements with "Control N:"
    control_pattern = r'(Control\s+\d+:?\s+[^\.;]{10,100})'
    for match in re.finditer(control_pattern, text, re.IGNORECASE):
        candidates.append({
            "text": match.group(1).strip(),
            "score": 0.95,  # Very high confidence
            "action": "Split into separate control"
        })

    # Filter duplicates and sort by score
    unique_candidates = []
    seen_texts = set()

    for candidate in candidates:
        normalized_text = candidate["text"].lower()
        if normalized_text not in seen_texts:
            seen_texts.add(normalized_text)
            unique_candidates.append(candidate)

    # Sort by score (descending)
    unique_candidates.sort(key=lambda x: x["score"], reverse=True)

    return unique_candidates