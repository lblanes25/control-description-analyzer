#!/usr/bin/env python3
"""
Enhanced Multi-Control Detection Module

This module analyzes control descriptions to identify when multiple controls
are described in a single text. It uses a comprehensive approach examining
WHO, WHAT, WHEN, and ESCALATION elements to differentiate between:
- Multiple distinct controls
- A single control with multiple aspects
- A single control with escalation paths

The implementation works closely with other detection modules while
maintaining separation of concerns.
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Set


def detect_multi_control(text: str, who_data: Dict, what_data: Dict,
                         when_data: Dict, escalation_data: Dict,
                         config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Detect if multiple distinct controls are described in a single text.

    Args:
        text: The control description text
        who_data: Results from WHO element detection
        what_data: Results from WHAT element detection
        when_data: Results from WHEN element detection
        escalation_data: Results from ESCALATION element detection
        config: Optional configuration dictionary

    Returns:
        Dictionary with detection results including:
        - detected: Boolean indicating if multiple controls detected
        - count: Estimated number of distinct controls
        - candidates: List of potential control candidates
        - confidence: Confidence level of detection
    """
    if not text or text.strip() == '':
        return {
            "detected": False,
            "count": 0,
            "candidates": [],
            "confidence": "low"
        }

    # Use config or defaults
    config = config or {}
    multi_control_config = config.get("multi_control", {})

    # Extract performers (WHO)
    primary_who = who_data.get("primary", {}).get("text", "").lower() if who_data.get("primary") else ""
    secondary_whos = [w.get("text", "").lower() for w in who_data.get("secondary", [])]
    all_performers = [primary_who] + secondary_whos if primary_who else secondary_whos
    all_performers = [p for p in all_performers if p]  # Remove empty strings

    # Extract actions (WHAT)
    primary_action = what_data.get("primary_action", {})
    secondary_actions = what_data.get("secondary_actions", [])
    all_actions = [primary_action] + secondary_actions if primary_action else secondary_actions
    all_actions = [a for a in all_actions if a]  # Remove empty values

    # Extract timing information (WHEN)
    timing_candidates = when_data.get("candidates", [])
    explicit_frequencies = [c for c in timing_candidates if c.get("method", "").startswith("explicit_frequency")]
    complex_patterns = [c for c in timing_candidates if c.get("method", "").startswith("complex_pattern")]

    # Check for multiple frequencies explicitly detected by WHEN module
    multi_frequency_detected = when_data.get("multi_frequency_detected", False)
    detected_frequencies = when_data.get("frequencies", [])

    # Extract escalation information
    has_escalation = escalation_data.get("detected", False)
    escalation_phrases = escalation_data.get("phrases", [])

    # Get escalation and sequence markers
    escalation_markers = get_config_value(multi_control_config, "escalation_markers", [
        "if ", "when ", "exception", "discrepan", "error", "issue",
        "escalate", "notify", "alert", "report to", "exceed", "threshold",
        "not match", "fails", "failure", "variance"
    ])

    # Expanded sequence markers with additional terms
    sequence_markers = get_config_value(multi_control_config, "sequence_markers", [
        "then", "after", "subsequently", "next", "following", "afterward",
        "first", "second", "third", "finally", "lastly", "initially",
        "subsequently", "consequently", "meanwhile", "later", "prior to"
    ])

    # Check for ad-hoc patterns alongside regular frequencies
    adhoc_terms = get_config_value(multi_control_config, "adhoc_terms", [
        "ad hoc", "adhoc", "as needed", "when needed", "if needed",
        "on-demand", "on demand", "as required", "when necessary"
    ])

    # Detect the presence of ad-hoc timing alongside regular frequencies
    has_adhoc_timing = any(term.lower() in text.lower() for term in adhoc_terms)
    has_regular_frequency = len(detected_frequencies) > 0 or len(explicit_frequencies) > 0
    mixed_timing_detected = has_adhoc_timing and has_regular_frequency

    # Categorize each action by context, timing, and performer
    categorized_actions = categorize_actions(
        text, all_actions, escalation_markers, timing_candidates, all_performers
    )

    # Group actions by timing pattern
    timing_groups = group_actions_by_timing(categorized_actions, detected_frequencies)

    # Analyze paragraph structure for separate controls
    paragraph_structure = analyze_paragraph_structure(text)
    has_distinct_paragraphs = paragraph_structure.get("has_distinct_paragraphs", False)

    # Detect sequence markers that indicate process steps vs. separate controls
    sequence_analysis = analyze_sequence_markers(
        text, sequence_markers, escalation_markers, timing_candidates, all_performers
    )
    has_control_sequence = sequence_analysis.get("has_control_sequence", False)

    # Find controls with distinct performers
    performer_groups = group_actions_by_performer(categorized_actions, all_performers)
    has_distinct_performers = len(performer_groups) > 1

    # Build control candidates using the enhanced groupings
    control_candidates = build_control_candidates(
        timing_groups, performer_groups, categorized_actions,
        all_performers, text, has_adhoc_timing, adhoc_terms
    )

    # Check for each clear indicator of multiple controls
    multi_control_indicators = {
        "distinct_timing": len(timing_groups) > 1,
        "mixed_timing": mixed_timing_detected,
        "multi_frequency": multi_frequency_detected,
        "distinct_paragraphs": has_distinct_paragraphs,
        "control_sequence": has_control_sequence,
        "distinct_performers": has_distinct_performers
    }

    # Count positive indicators
    indicator_count = sum(1 for value in multi_control_indicators.values() if value)

    # Make final determination with enhanced logic
    is_multi_control = (
            len(timing_groups) > 1 or  # Different timing patterns
            multi_frequency_detected or  # Multiple frequencies explicitly detected
            (mixed_timing_detected and indicator_count >= 1) or  # Mixed timing with other indicator
            (has_distinct_paragraphs and indicator_count >= 1) or  # Structural separation with other indicator
            (has_distinct_performers and len(control_candidates) > 1) or  # Different performers doing different things
            (has_control_sequence and not is_escalation_sequence(text, escalation_markers))  # True control sequence
    )

    # Calculate confidence level with enhanced factors
    confidence = calculate_confidence(
        is_multi_control, timing_groups, multi_frequency_detected,
        control_candidates, sequence_analysis, has_distinct_performers,
        has_distinct_paragraphs, mixed_timing_detected
    )

    return {
        "detected": is_multi_control,
        "count": len(control_candidates) if is_multi_control else 1,
        "candidates": control_candidates,
        "has_multi_frequency": multi_frequency_detected,
        "timing_groups": list(timing_groups.keys()),
        "has_adhoc_component": has_adhoc_timing,
        "has_sequence_markers": sequence_analysis.get("has_sequence_markers", False),
        "has_distinct_performers": has_distinct_performers,
        "has_distinct_paragraphs": has_distinct_paragraphs,
        "multi_control_indicators": multi_control_indicators,
        "confidence": confidence
    }


def categorize_actions(text: str, all_actions: List[Dict],
                       escalation_markers: List[str],
                       timing_candidates: List[Dict],
                       all_performers: List[str]) -> List[Dict]:
    """
    Categorize actions as regular control actions or escalation actions,
    and associate them with timing information and performers.

    Args:
        text: Control description text
        all_actions: List of action dictionaries
        escalation_markers: List of markers indicating escalation
        timing_candidates: List of timing candidates
        all_performers: List of identified performers

    Returns:
        List of categorized action dictionaries
    """
    categorized_actions = []
    text_lower = text.lower()

    # First, create a map of timing terms to their positions
    timing_positions = {}
    for timing in timing_candidates:
        timing_text = timing.get("text", "").lower()
        if timing_text:
            span = timing.get("span", [0, 0])
            timing_positions[timing_text] = {
                "span": span,
                "frequency": timing.get("frequency", "unknown"),
                "is_vague": timing.get("is_vague", False)
            }

    # Create a map of performers to their positions
    performer_positions = {}
    for performer in all_performers:
        performer_lower = performer.lower()
        # Find all occurrences of this performer
        for match in re.finditer(r'\b' + re.escape(performer_lower) + r'\b', text_lower):
            performer_positions[performer_lower] = {
                "span": [match.start(), match.end()],
                "text": performer
            }

    for action in all_actions:
        if not action:
            continue

        action_text = action.get("full_phrase", "").lower() if isinstance(action, dict) else ""
        if not action_text:
            continue

        # Find position of this action in text
        action_pos = text_lower.find(action_text)
        if action_pos == -1:
            continue  # Skip if action not found

        action_span = [action_pos, action_pos + len(action_text)]
        action_context = get_surrounding_context(text, action_pos, action_pos + len(action_text), 150)

        # Check if action is associated with distinct timing
        timing_association = find_timing_for_action(action_text, action_context, timing_candidates)

        # Is this likely an escalation action?
        is_escalation = is_escalation_action(action_context, escalation_markers)

        # Find the closest performer to this action
        associated_performer = find_performer_for_action(
            action_span, all_performers, performer_positions, text_lower
        )

        categorized_actions.append({
            "action": action,
            "text": action_text,
            "span": action_span,
            "is_escalation": is_escalation,
            "timing": timing_association,
            "performer": associated_performer,
            "context": action_context
        })

    return categorized_actions


def is_escalation_action(context: str, escalation_markers: List[str]) -> bool:
    """
    Determine if an action is part of an escalation path rather than a primary control.

    Args:
        context: Surrounding context of the action
        escalation_markers: List of escalation indicator terms

    Returns:
        Boolean indicating if this is an escalation action
    """
    # Check for escalation markers
    has_escalation_marker = any(marker.lower() in context.lower() for marker in escalation_markers)

    # Check for conditional phrases that indicate escalation
    conditional_patterns = [
        r'\bif\s+[^\.;,]{3,30}(,\s+|then\s+)',
        r'\bwhen\s+[^\.;,]{3,30}(,\s+|then\s+)',
        r'\bin\s+case\s+of\s+',
        r'\bshould\s+[^\.;,]{3,40}(,\s+|then\s+)',
    ]

    has_conditional = any(re.search(pattern, context.lower()) for pattern in conditional_patterns)

    # Look for escalation-specific verbs
    escalation_verbs = [
        "escalate", "notify", "alert", "report", "inform",
        "communicate", "contact", "forward", "send", "raise"
    ]

    has_escalation_verb = any(re.search(r'\b' + re.escape(verb) + r'\b', context.lower())
                              for verb in escalation_verbs)

    return has_escalation_marker or (has_conditional and has_escalation_verb)


def analyze_paragraph_structure(text: str) -> Dict[str, Any]:
    """
    Analyze the paragraph structure to identify if there are distinct
    control statements separated by paragraph breaks.

    Args:
        text: Control description text

    Returns:
        Dictionary with paragraph analysis results
    """
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n|\n\s*\r\n', text)
    clean_paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # If only one paragraph, check for sentence-based separation
    if len(clean_paragraphs) <= 1:
        return {
            "has_distinct_paragraphs": False,
            "paragraph_count": len(clean_paragraphs)
        }

    # Check if paragraphs likely contain distinct controls
    distinct_control_paragraphs = []
    control_verbs = [
        "review", "approve", "check", "validate", "ensure", "monitor",
        "reconcile", "verify", "examine", "inspect", "audit", "confirm"
    ]

    for para in clean_paragraphs:
        # Check if paragraph contains a main control verb at the beginning
        para_lower = para.lower()
        contains_control_verb = any(re.search(r'\b' + verb + r'\b', para_lower[:50])
                                    for verb in control_verbs)

        # Check if paragraph starts with a likely performer
        performer_start = re.search(r'^(?:The|An|A)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', para)

        # Check if paragraph has a frequency indicator at the beginning
        frequency_start = re.search(r'^(?:Daily|Weekly|Monthly|Quarterly|Annually|Every|Each)', para)

        if contains_control_verb or performer_start or frequency_start:
            distinct_control_paragraphs.append(para)

    has_distinct_paragraphs = len(distinct_control_paragraphs) > 1

    return {
        "has_distinct_paragraphs": has_distinct_paragraphs,
        "paragraph_count": len(clean_paragraphs),
        "distinct_control_paragraphs": len(distinct_control_paragraphs)
    }


def analyze_sequence_markers(text: str, sequence_markers: List[str],
                             escalation_markers: List[str],
                             timing_candidates: List[Dict],
                             all_performers: List[str]) -> Dict[str, Any]:
    """
    Analyze sequence markers to differentiate between process steps within
    a single control and sequences of separate controls.

    Args:
        text: Control description text
        sequence_markers: List of sequence indicator terms
        escalation_markers: List of escalation markers
        timing_candidates: List of timing candidates
        all_performers: List of identified performers

    Returns:
        Dictionary with sequence analysis results
    """
    text_lower = text.lower()

    # Find all sequence markers
    sequence_positions = []
    for marker in sequence_markers:
        for match in re.finditer(r'\b' + re.escape(marker) + r'\b', text_lower):
            sequence_positions.append({
                "marker": marker,
                "span": [match.start(), match.end()],
                "context": get_surrounding_context(text, match.start(), match.end(), 100)
            })

    # No sequence markers found
    if not sequence_positions:
        return {
            "has_sequence_markers": False,
            "has_control_sequence": False,
            "sequence_markers_found": []
        }

    # Analyze each sequence marker for control separation vs. process steps
    control_sequence_markers = []

    for seq in sequence_positions:
        context = seq["context"]
        marker = seq["marker"]

        # Check if marker is surrounded by escalation context
        is_escalation_context = any(escalation in context for escalation in escalation_markers)

        # Check if marker is followed by a different timing pattern
        follows_timing = False
        preceded_by_timing = False

        for timing in timing_candidates:
            timing_span = timing.get("span", [0, 0])
            # Check if timing comes before this sequence marker
            if timing_span[1] < seq["span"][0] and timing_span[1] >= seq["span"][0] - 100:
                preceded_by_timing = True
            # Check if timing comes after this sequence marker
            if timing_span[0] > seq["span"][1] and timing_span[0] <= seq["span"][1] + 100:
                follows_timing = True

        # Check if marker is followed by a different performer
        follows_performer = False
        for performer in all_performers:
            performer_pos = text_lower.find(performer.lower(), seq["span"][1])
            if performer_pos > 0 and performer_pos <= seq["span"][1] + 100:
                follows_performer = True
                break

        # A sequence marker indicates separate controls if:
        # 1. It's not in an escalation context AND
        # 2. It's followed by a different timing pattern OR a different performer
        indicates_control_sequence = (
                not is_escalation_context and (follows_timing or follows_performer)
        )

        if indicates_control_sequence:
            control_sequence_markers.append({
                "marker": marker,
                "span": seq["span"],
                "context": context,
                "follows_timing": follows_timing,
                "follows_performer": follows_performer
            })

    return {
        "has_sequence_markers": len(sequence_positions) > 0,
        "has_control_sequence": len(control_sequence_markers) > 0,
        "sequence_markers_found": [s["marker"] for s in sequence_positions],
        "control_sequence_markers": [s["marker"] for s in control_sequence_markers]
    }


def is_escalation_sequence(text: str, escalation_markers: List[str]) -> bool:
    """
    Determine if sequence markers in the text primarily represent escalation paths
    rather than separate controls.

    Args:
        text: Control description text
        escalation_markers: List of escalation indicator terms

    Returns:
        Boolean indicating if the sequence is primarily escalation
    """
    text_lower = text.lower()

    # Check if sequence is following an escalation pattern
    escalation_context_count = sum(1 for marker in escalation_markers
                                   if marker.lower() in text_lower)

    # Look for if-then patterns indicating escalation
    if_then_patterns = [
        r'if\s+[^\.;]{5,50}(?:,\s+|\.\s+|\s+then\s+)(?:[^\.;]{5,50})?',
        r'when\s+[^\.;]{5,50}(?:,\s+|\.\s+|\s+then\s+)(?:[^\.;]{5,50})?',
        r'in\s+case\s+[^\.;]{5,50}(?:,\s+|\.\s+|\s+then\s+)(?:[^\.;]{5,50})?'
    ]

    has_if_then = any(re.search(pattern, text_lower) for pattern in if_then_patterns)

    # Check for escalation verbs following conditional words
    escalation_verbs = ["escalate", "notify", "alert", "report", "inform"]

    conditional_escalation_patterns = [
        r'if\s+[^\.;]{1,40}(?:' + '|'.join(escalation_verbs) + r')',
        r'when\s+[^\.;]{1,40}(?:' + '|'.join(escalation_verbs) + r')'
    ]

    has_conditional_escalation = any(re.search(pattern, text_lower)
                                     for pattern in conditional_escalation_patterns)

    # If there are multiple escalation markers or conditional patterns
    # with escalation verbs, this is likely an escalation sequence
    return (escalation_context_count >= 2 or
            (has_if_then and has_conditional_escalation))


def group_actions_by_timing(categorized_actions: List[Dict],
                            detected_frequencies: List[str]) -> Dict[str, List[Dict]]:
    """
    Group actions by their timing pattern to identify distinct controls.
    Enhanced to handle ad-hoc alongside regular frequencies and complex timing.

    Args:
        categorized_actions: List of categorized action dictionaries
        detected_frequencies: List of normalized frequencies detected

    Returns:
        Dictionary mapping timing keys to lists of actions
    """
    timing_groups = {}

    # First pass: group by explicit frequency
    for action_info in categorized_actions:
        if action_info["is_escalation"]:
            continue  # Skip escalation actions for grouping

        timing = action_info.get("timing", {})
        timing_key = timing.get("frequency", "unknown")
        timing_method = timing.get("method", "")

        # Special case for handling ad-hoc vs. regular frequencies
        if "adhoc" in timing_key.lower() or "as needed" in timing_key.lower():
            timing_key = "adhoc"  # Normalize ad-hoc terms
        elif "event" in timing_method or "trigger" in timing_method:
            timing_key = "event_triggered"
        elif not timing_key or timing_key == "unknown":
            # If no explicit timing, look for timing words in context
            context = action_info.get("context", "").lower()

            # Check if context suggests a timing pattern
            if any(freq.lower() in context for freq in detected_frequencies):
                for freq in detected_frequencies:
                    if freq.lower() in context:
                        timing_key = freq
                        break

        # Ensure the group exists
        if timing_key not in timing_groups:
            timing_groups[timing_key] = []

        timing_groups[timing_key].append(action_info)

    return timing_groups


def group_actions_by_performer(categorized_actions: List[Dict],
                               all_performers: List[str]) -> Dict[str, List[Dict]]:
    """
    Group actions by their associated performer to identify distinct controls.

    Args:
        categorized_actions: List of categorized action dictionaries
        all_performers: List of performer strings

    Returns:
        Dictionary mapping performer keys to lists of actions
    """
    performer_groups = {}

    for action_info in categorized_actions:
        if action_info["is_escalation"]:
            continue  # Skip escalation actions for grouping

        performer = action_info.get("performer", "unknown")

        # Handle cases where performer wasn't explicitly associated
        if not performer or performer == "unknown":
            # Try to find performer in context
            action_context = action_info.get("context", "").lower()

            for possible_performer in all_performers:
                if possible_performer.lower() in action_context:
                    performer = possible_performer
                    break

        # Ensure the group exists
        if performer not in performer_groups:
            performer_groups[performer] = []

        performer_groups[performer].append(action_info)

    return performer_groups


def build_control_candidates(timing_groups: Dict[str, List[Dict]],
                             performer_groups: Dict[str, List[Dict]],
                             categorized_actions: List[Dict],
                             all_performers: List[str],
                             text: str,
                             has_adhoc_timing: bool,
                             adhoc_terms: List[str]) -> List[Dict]:
    """
    Build control candidates from timing groups, performer groups, and actions.
    Enhanced to better handle mixed timing patterns and performers.

    Args:
        timing_groups: Dictionary mapping timing keys to lists of actions
        performer_groups: Dictionary mapping performer keys to lists of actions
        categorized_actions: List of categorized action dictionaries
        all_performers: List of all performers
        text: Control description text
        has_adhoc_timing: Whether ad-hoc timing is detected
        adhoc_terms: List of ad-hoc timing terms

    Returns:
        List of control candidate dictionaries
    """
    control_candidates = []
    text_lower = text.lower()

    # First, create candidates from timing groups
    for frequency, actions in timing_groups.items():
        for action in actions:
            # Use the associated performer from categorization
            associated_who = action.get("performer", "")

            # If no performer was associated, try to find one
            if not associated_who:
                for performer in all_performers:
                    if performer.lower() in action.get("context", "").lower():
                        associated_who = performer
                        break

            # Still no performer found, use primary if available
            if not associated_who and all_performers:
                associated_who = all_performers[0]

            control_candidates.append({
                "who": associated_who,
                "what": action["text"],
                "when": frequency,
                "is_escalation": False,
                "span": action.get("span", [0, 0]),
                "action": action["action"]  # Include original action data
            })

    # Then, add candidates from distinct performer groups not covered
    performers_added = set(c["who"].lower() for c in control_candidates if c["who"])

    for performer, actions in performer_groups.items():
        # Skip if performer's action is already covered
        if performer.lower() in performers_added:
            # Check that all actions for this performer are covered
            performer_action_texts = set(a["text"].lower() for a in actions)
            candidate_action_texts = set(
                c["what"].lower() for c in control_candidates
                if c["who"].lower() == performer.lower()
            )

            # If all actions are covered, skip this performer
            if performer_action_texts.issubset(candidate_action_texts):
                continue

        # Add actions for this performer that aren't already covered
        for action in actions:
            action_text = action["text"].lower()

            # Skip if this exact action is already covered
            if any(c["what"].lower() == action_text and
                   c["who"].lower() == performer.lower()
                   for c in control_candidates):
                continue

            # Find timing for this action
            timing = action.get("timing", {})
            frequency = timing.get("frequency", "unknown")

            control_candidates.append({
                "who": performer,
                "what": action["text"],
                "when": frequency,
                "is_escalation": False,
                "span": action.get("span", [0, 0]),
                "action": action["action"]
            })

    # Check for ad-hoc controls mixed with regular controls
    if has_adhoc_timing and "adhoc" not in timing_groups:
        adhoc_context = ""
        for term in adhoc_terms:
            if term.lower() in text_lower:
                adhoc_pos = text_lower.find(term.lower())
                if adhoc_pos >= 0:
                    adhoc_context = get_surrounding_context(text, adhoc_pos, adhoc_pos + len(term), 150)
                    break

        if adhoc_context:
            # Find actions in adhoc context not already covered
            for action_info in categorized_actions:
                action_text = action_info["text"].lower()
                action_context = action_info.get("context", "").lower()

                # Skip escalation actions
                if action_info["is_escalation"]:
                    continue

                # Skip if action is already in a candidate
                if any(c["what"].lower() == action_text for c in control_candidates):
                    continue

                # Check if this action appears in adhoc context
                if action_text in adhoc_context:
                    associated_who = action_info.get("performer", "")

                    # If no performer was associated, try to find one
                    if not associated_who:
                        for performer in all_performers:
                            if performer.lower() in adhoc_context:
                                associated_who = performer
                                break

                    # If still no performer, use primary
                    if not associated_who and all_performers:
                        associated_who = all_performers[0]

                    control_candidates.append({
                        "who": associated_who,
                        "what": action_info["text"],
                        "when": "adhoc",
                        "is_escalation": False,
                        "span": action_info.get("span", [0, 0]),
                        "action": action_info["action"]
                    })

    # Deduplicate candidates by finding unique (who, what, when) combinations
    unique_candidates = []
    seen_combinations = set()

    for candidate in control_candidates:
        combo_key = (
            candidate["who"].lower() if candidate["who"] else "",
            candidate["what"].lower() if candidate["what"] else "",
            candidate["when"].lower() if candidate["when"] else ""
        )

        if combo_key not in seen_combinations:
            seen_combinations.add(combo_key)
            unique_candidates.append(candidate)

    return unique_candidates


def calculate_confidence(is_multi_control: bool,
                         timing_groups: Dict[str, List[Dict]],
                         multi_frequency_detected: bool,
                         control_candidates: List[Dict],
                         sequence_analysis: Dict[str, Any],
                         has_distinct_performers: bool,
                         has_distinct_paragraphs: bool,
                         mixed_timing_detected: bool) -> str:
    """
    Calculate confidence level for multi-control detection with enhanced factors.

    Args:
        is_multi_control: Whether multiple controls are detected
        timing_groups: Dictionary mapping timing keys to lists of actions
        multi_frequency_detected: Whether multiple frequencies detected
        control_candidates: List of control candidate dictionaries
        sequence_analysis: Results of sequence marker analysis
        has_distinct_performers: Whether distinct performers detected
        has_distinct_paragraphs: Whether distinct paragraphs detected
        mixed_timing_detected: Whether mixed timing patterns detected

    Returns:
        Confidence level string: "high", "medium", or "low"
    """
    if not is_multi_control:
        return "low"

    # Count strong indicators
    strong_indicators = 0

    # Different timing patterns
    if len(timing_groups) > 1:
        strong_indicators += 1

    # Explicit multiple frequencies
    if multi_frequency_detected:
        strong_indicators += 1

    # Control sequence markers
    if sequence_analysis.get("has_control_sequence", False):
        strong_indicators += 1

    # Distinct performers doing different things
    if has_distinct_performers:
        strong_indicators += 1

    # Distinct paragraphs describing controls
    if has_distinct_paragraphs:
        strong_indicators += 1

    # Mixed regular and ad-hoc timing
    if mixed_timing_detected:
        strong_indicators += 1

    # Convert indicator count to confidence level
    if strong_indicators >= 2:
        return "high"
    elif strong_indicators == 1 and len(control_candidates) > 1:
        return "medium"
    else:
        return "low"


def find_timing_for_action(action_text: str, action_context: str,
                           timing_candidates: List[Dict]) -> Dict:
    """
    Associate an action with its timing information based on textual context.
    Enhanced to handle more timing patterns and contextual clues.

    Args:
        action_text: Action text
        action_context: Context around the action
        timing_candidates: List of timing candidates

    Returns:
        Dictionary with timing information
    """
    action_lower = action_text.lower()
    context_lower = action_context.lower()

    # Extract action start position in the context
    action_pos_in_context = context_lower.find(action_lower)

    best_match = None
    best_score = 0

    # Check each timing candidate
    for candidate in timing_candidates:
        candidate_text = candidate.get("text", "").lower()

        # Skip if timing text is empty
        if not candidate_text:
            continue

        # Check if timing text appears in action context
        if candidate_text in context_lower:
            candidate_pos = context_lower.find(candidate_text)

            # Calculate proximity score (higher for closer timing)
            distance = abs(candidate_pos - action_pos_in_context)
            proximity_score = 150 - min(distance, 150)  # Max distance of 150

            # Determine timing relationship
            timing_before_action = candidate_pos < action_pos_in_context

            # Timing before action gets higher score (more likely to be associated)
            relationship_bonus = 1.2 if timing_before_action else 1.0

            # Calculate final score
            match_score = (proximity_score / 150) * relationship_bonus

            # Skip vague timing if we have a better specific timing
            is_vague = candidate.get("is_vague", False)
            if is_vague and best_match and not best_match.get("is_vague", True):
                continue

            # Update best match if score is better
            if match_score > best_score:
                best_score = match_score
                best_match = {
                    "text": candidate_text,
                    "frequency": candidate.get("frequency", "unknown"),
                    "method": candidate.get("method", ""),
                    "is_vague": is_vague,
                    "proximity_score": match_score
                }

    # If no timing found in direct context, try inference from sentence structure
    if not best_match:
        # Look for timing indicators in the immediate context
        timing_indicators = ["daily", "weekly", "monthly", "quarterly", "annually", "every", "each", "upon"]

        for indicator in timing_indicators:
            if indicator in context_lower:
                # Look for the indicator and surrounding text
                indicator_pos = context_lower.find(indicator)
                if indicator_pos >= 0:
                    # Extract potential timing phrase (up to 5 words)
                    timing_start = max(0, context_lower.rfind(" ", 0, indicator_pos))
                    timing_end = context_lower.find(".", indicator_pos)
                    if timing_end == -1:
                        timing_end = len(context_lower)

                    timing_phrase = context_lower[timing_start:timing_end].strip()

                    # Determine frequency from the timing phrase
                    frequency = "unknown"
                    if "daily" in timing_phrase:
                        frequency = "daily"
                    elif "weekly" in timing_phrase:
                        frequency = "weekly"
                    elif "monthly" in timing_phrase:
                        frequency = "monthly"
                    elif "quarterly" in timing_phrase:
                        frequency = "quarterly"
                    elif "annually" in timing_phrase or "yearly" in timing_phrase:
                        frequency = "annually"

                    best_match = {
                        "text": timing_phrase,
                        "frequency": frequency,
                        "method": "inferred_from_context",
                        "is_vague": False,
                        "proximity_score": 0.5  # Lower confidence for inferred timing
                    }
                    break

    # Return empty dict if no timing found
    if not best_match:
        return {"text": "", "frequency": "unknown", "method": ""}

    return best_match


def find_performer_for_action(action_span: List[int], all_performers: List[str],
                              performer_positions: Dict[str, Dict],
                              text_lower: str) -> str:
    """
    Find the performer most likely associated with a specific action based on proximity
    and syntactic relationships.

    Args:
        action_span: [start, end] positions of the action in text
        all_performers: List of all performers
        performer_positions: Dictionary mapping performers to their positions
        text_lower: Lowercased text

    Returns:
        Performer text or empty string if none found
    """
    if not all_performers:
        return ""

    # Get action position
    action_start, action_end = action_span

    # First try to find performers that appear before the action within a reasonable distance
    closest_performer = None
    min_distance = float('inf')

    for performer in all_performers:
        performer_lower = performer.lower()
        position_info = performer_positions.get(performer_lower)

        if not position_info:
            continue

        performer_span = position_info.get("span", [0, 0])
        performer_start, performer_end = performer_span

        # Look for performers before the action (higher priority)
        if performer_end <= action_start:
            distance = action_start - performer_end

            # Check for sentence boundary between performer and action
            text_between = text_lower[performer_end:action_start]
            has_sentence_boundary = "." in text_between

            # Apply penalty for sentence boundary
            distance_with_penalty = distance * 2 if has_sentence_boundary else distance

            if distance_with_penalty < min_distance:
                min_distance = distance_with_penalty
                closest_performer = performer

        # If no performer before action, check after action (lower priority)
        elif closest_performer is None and performer_start >= action_end:
            distance = performer_start - action_end

            # Check for sentence boundary between action and performer
            text_between = text_lower[action_end:performer_start]
            has_sentence_boundary = "." in text_between

            # Apply larger penalty for performer after action with sentence boundary
            distance_with_penalty = distance * 3 if has_sentence_boundary else distance * 1.5

            if distance_with_penalty < min_distance:
                min_distance = distance_with_penalty
                closest_performer = performer

    # If we found a performer within reasonable distance (e.g., 150 characters)
    if closest_performer and min_distance <= 150:
        return closest_performer

    # If no close performer found, return the primary performer (first in list)
    return all_performers[0] if all_performers else ""


def get_surrounding_context(text: str, start: int, end: int, window_size: int = 100) -> str:
    """
    Get the surrounding context of a phrase within the text.

    Args:
        text: Full text
        start: Start position of phrase
        end: End position of phrase
        window_size: Size of context window in characters

    Returns:
        Context string
    """
    if not text:
        return ""

    context_start = max(0, start - window_size)
    context_end = min(len(text), end + window_size)

    return text[context_start:context_end]


def get_config_value(config: Dict, key: str, default_value: Any) -> Any:
    """
    Helper function to safely get a value from configuration with defaults.

    Args:
        config: Configuration dictionary
        key: Configuration key
        default_value: Default value if key not found

    Returns:
        Configuration value or default
    """
    return config.get(key, default_value)


def mark_possible_standalone_controls(text: str, nlp, config: Optional[Dict] = None) -> List[Dict]:
    """
    Identify possible standalone control statements within a text.
    This is a simplified detection that can be used independently.

    Args:
        text: Control description text
        nlp: spaCy NLP model
        config: Optional configuration dictionary

    Returns:
        List of possible control statements with scores
    """
    config = config or {}

    # Process text
    doc = nlp(text)

    # Split text into sentences
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

    # Control action verbs (simplified list)
    control_verbs = [
        "review", "approve", "verify", "check", "validate", "reconcile",
        "examine", "analyze", "evaluate", "assess", "monitor", "track"
    ]

    # Identify possible standalone controls
    candidates = []

    for i, sent in enumerate(sentences):
        sent_doc = nlp(sent)

        # Check if sentence contains a key control verb
        contains_verb = any(token.lemma_.lower() in control_verbs for token in sent_doc)

        # Check if sentence has a subject (likely performer)
        has_subject = any(token.dep_ == "nsubj" for token in sent_doc)

        # Simple score based on these features
        score = 0
        if contains_verb:
            score += 0.5
        if has_subject:
            score += 0.3

        # Boost score if sentence starts with a capitalized word (likely new thought)
        if sent[0].isupper() and i > 0:
            score += 0.1

        # Check for timing indicators
        has_timing = any(re.search(
            r'\b(daily|weekly|monthly|quarterly|annually|every|each|when|as needed)\b',
            sent, re.IGNORECASE
        ))

        if has_timing:
            score += 0.1

        # Only add if minimum threshold is met
        if score > 0.5:
            verb_tokens = [token for token in sent_doc if token.lemma_.lower() in control_verbs]
            action = verb_tokens[0].lemma_ if verb_tokens else ""

            candidates.append({
                "text": sent,
                "score": score,
                "position": i,
                "action": action,
                "has_subject": has_subject,
                "has_timing": has_timing
            })

    return candidates