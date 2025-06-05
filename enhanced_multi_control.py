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
from dataclasses import dataclass

# Import shared constants from WHO detection module
try:
    from enhanced_who import CONTROL_VERBS, HUMAN_INDICATORS
except ImportError:
    # Fallback constants if import fails
    CONTROL_VERBS = [
        "review", "approve", "verify", "check", "validate", "reconcile",
        "examine", "analyze", "evaluate", "assess", "monitor", "track",
        "investigate", "inspect", "audit", "oversee", "supervise", "ensure",
        "perform", "execute", "conduct", "disable", "enforce", "generate",
        "address", "compare", "maintain", "identify", "correct", "update",
        "submit", "complete", "prepare", "provide", "confirm"
    ]

# Multi-control specific constants
DEFAULT_ESCALATION_MARKERS = [
    "if ", "when ", "exception", "discrepan", "error", "issue",
    "escalate", "notify", "alert", "report to", "exceed", "threshold",
    "not match", "fails", "failure", "variance"
]

DEFAULT_SEQUENCE_MARKERS = [
    "then", "after", "subsequently", "next", "following", "afterward",
    "first", "second", "third", "finally", "lastly", "initially",
    "subsequently", "consequently", "meanwhile", "later", "prior to"
]

DEFAULT_ADHOC_TERMS = [
    "ad hoc", "adhoc", "as needed", "when needed", "if needed",
    "on-demand", "on demand", "as required", "when necessary"
]

ESCALATION_VERBS = [
    "escalate", "notify", "alert", "report", "inform",
    "communicate", "contact", "forward", "send", "raise"
]

CONDITIONAL_PATTERNS = [
    r'\bif\s+[^\.;,]{3,30}(,\s+|then\s+)',
    r'\bwhen\s+[^\.;,]{3,30}(,\s+|then\s+)',
    r'\bin\s+case\s+of\s+',
    r'\bshould\s+[^\.;,]{3,40}(,\s+|then\s+)',
]

TIMING_INDICATORS = ["daily", "weekly", "monthly", "quarterly", "annually", "every", "each", "upon"]

# Context and proximity constants
DEFAULT_CONTEXT_WINDOW = 100
EXTENDED_CONTEXT_WINDOW = 150
MAX_PERFORMER_DISTANCE = 150
SENTENCE_BOUNDARY_PENALTY = 2
AFTER_ACTION_PENALTY = 1.5
AFTER_ACTION_SENTENCE_PENALTY = 3

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 2
MEDIUM_CONFIDENCE_THRESHOLD = 1
MINIMUM_CONTROL_SCORE = 0.5
VERB_SCORE_WEIGHT = 0.5
SUBJECT_SCORE_WEIGHT = 0.3
TIMING_SCORE_WEIGHT = 0.1
NEW_SENTENCE_SCORE_WEIGHT = 0.1


@dataclass
class MultiControlConfig:
    """Configuration object for multi-control detection"""
    escalation_markers: List[str] = None
    sequence_markers: List[str] = None
    adhoc_terms: List[str] = None

    def __post_init__(self):
        if self.escalation_markers is None:
            self.escalation_markers = DEFAULT_ESCALATION_MARKERS.copy()
        if self.sequence_markers is None:
            self.sequence_markers = DEFAULT_SEQUENCE_MARKERS.copy()
        if self.adhoc_terms is None:
            self.adhoc_terms = DEFAULT_ADHOC_TERMS.copy()


def detect_multi_control(text: str, who_data: Dict, what_data: Dict,
                         when_data: Dict, escalation_data: Dict,
                         config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Main orchestrator function for multi-control detection.

    Detect if multiple distinct controls are described in a single text.
    """
    if not text or text.strip() == '':
        return _create_empty_result()

    # Initialize configuration
    multi_config = _initialize_configuration(config)

    # Extract and prepare data
    performers_data = _extract_performers_data(who_data)
    actions_data = _extract_actions_data(what_data)
    timing_data = _extract_timing_data(when_data)
    escalation_info = _extract_escalation_data(escalation_data)

    # Categorize actions with context
    categorized_actions = _categorize_actions_with_context(
        text, actions_data, multi_config, timing_data.timing_candidates, performers_data.all_performers
    )

    # Analyze structure and patterns
    structure_analysis = _analyze_text_structure(text, multi_config, timing_data, performers_data)

    # Group actions by different criteria
    groupings = _create_action_groupings(categorized_actions, timing_data, performers_data)

    # Build control candidates
    control_candidates = _build_enhanced_control_candidates(
        groupings, categorized_actions, performers_data, text, timing_data, multi_config
    )

    # Determine if multiple controls exist
    detection_result = _determine_multi_control_presence(
        groupings, timing_data, structure_analysis, control_candidates, multi_config
    )

    # Calculate confidence
    confidence = _calculate_enhanced_confidence(detection_result, groupings, structure_analysis, control_candidates)

    return _assemble_final_detection_result(
        detection_result, control_candidates, groupings, structure_analysis, timing_data, confidence
    )


@dataclass
class PerformersData:
    """Data structure for performer information"""
    primary_who: str
    secondary_whos: List[str]
    all_performers: List[str]


@dataclass
class TimingData:
    """Data structure for timing information"""
    timing_candidates: List[Dict]
    explicit_frequencies: List[Dict]
    complex_patterns: List[Dict]
    multi_frequency_detected: bool
    detected_frequencies: List[str]
    has_adhoc_timing: bool
    has_regular_frequency: bool
    mixed_timing_detected: bool


@dataclass
class ActionGroupings:
    """Data structure for action groupings"""
    timing_groups: Dict[str, List[Dict]]
    performer_groups: Dict[str, List[Dict]]


def _create_empty_result() -> Dict[str, Any]:
    """Create empty result for invalid input"""
    return {
        "detected": False,
        "count": 0,
        "candidates": [],
        "confidence": "low"
    }


def _initialize_configuration(config: Optional[Dict]) -> MultiControlConfig:
    """Initialize configuration with defaults"""
    if not config:
        return MultiControlConfig()

    multi_control_config = config.get("multi_control", {})
    return MultiControlConfig(
        escalation_markers=_get_config_value(multi_control_config, "escalation_markers", DEFAULT_ESCALATION_MARKERS),
        sequence_markers=_get_config_value(multi_control_config, "sequence_markers", DEFAULT_SEQUENCE_MARKERS),
        adhoc_terms=_get_config_value(multi_control_config, "adhoc_terms", DEFAULT_ADHOC_TERMS)
    )


def _extract_performers_data(who_data: Dict) -> PerformersData:
    """Extract and structure performer data"""
    primary_who = who_data.get("primary", {}).get("text", "").lower() if who_data.get("primary") else ""
    secondary_whos = [w.get("text", "").lower() for w in who_data.get("secondary", [])]
    all_performers = [primary_who] + secondary_whos if primary_who else secondary_whos
    all_performers = [p for p in all_performers if p]  # Remove empty strings

    return PerformersData(primary_who, secondary_whos, all_performers)


def _extract_actions_data(what_data: Dict) -> List[Dict]:
    """Extract and structure action data"""
    primary_action = what_data.get("primary_action", {})
    secondary_actions = what_data.get("secondary_actions", [])
    all_actions = [primary_action] + secondary_actions if primary_action else secondary_actions
    return [a for a in all_actions if a]  # Remove empty values


def _extract_timing_data(when_data: Dict) -> TimingData:
    """Extract and structure timing data"""
    timing_candidates = when_data.get("candidates", [])
    explicit_frequencies = [c for c in timing_candidates if c.get("method", "").startswith("explicit_frequency")]
    complex_patterns = [c for c in timing_candidates if c.get("method", "").startswith("complex_pattern")]

    multi_frequency_detected = when_data.get("multi_frequency_detected", False)
    detected_frequencies = when_data.get("frequencies", [])

    return TimingData(
        timing_candidates=timing_candidates,
        explicit_frequencies=explicit_frequencies,
        complex_patterns=complex_patterns,
        multi_frequency_detected=multi_frequency_detected,
        detected_frequencies=detected_frequencies,
        has_adhoc_timing=False,  #For Will be set later
        has_regular_frequency=len(detected_frequencies) > 0 or len(explicit_frequencies) > 0,
        mixed_timing_detected=False  # Will be set later
    )


def _extract_escalation_data(escalation_data: Dict) -> Dict:
    """Extract escalation information"""
    return {
        "has_escalation": escalation_data.get("detected", False),
        "escalation_phrases": escalation_data.get("phrases", [])
    }


def _categorize_actions_with_context(text: str, all_actions: List[Dict],
                                     multi_config: MultiControlConfig,
                                     timing_candidates: List[Dict],
                                     all_performers: List[str]) -> List[Dict]:
    """
    Enhanced action categorization with context analysis.

    Categorize actions as regular control actions or escalation actions,
    and associate them with timing information and performers.
    """
    categorized_actions = []
    text_lower = text.lower()

    # Create position maps for timing and performers
    timing_positions = _create_timing_position_map(timing_candidates)
    performer_positions = _create_performer_position_map(all_performers, text_lower)

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
        action_context = _get_surrounding_context(text, action_pos, action_pos + len(action_text),
                                                  EXTENDED_CONTEXT_WINDOW)

        # Analyze action characteristics
        timing_association = _find_timing_for_action(action_text, action_context, timing_candidates)
        is_escalation = _is_escalation_action(action_context, multi_config.escalation_markers)
        associated_performer = _find_performer_for_action(action_span, all_performers, performer_positions, text_lower)

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


def _create_timing_position_map(timing_candidates: List[Dict]) -> Dict[str, Dict]:
    """Create a map of timing terms to their positions"""
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
    return timing_positions


def _create_performer_position_map(all_performers: List[str], text_lower: str) -> Dict[str, Dict]:
    """Create a map of performers to their positions"""
    performer_positions = {}
    for performer in all_performers:
        performer_lower = performer.lower()
        # Find all occurrences of this performer
        for match in re.finditer(r'\b' + re.escape(performer_lower) + r'\b', text_lower):
            performer_positions[performer_lower] = {
                "span": [match.start(), match.end()],
                "text": performer
            }
    return performer_positions


def _analyze_text_structure(text: str, multi_config: MultiControlConfig,
                            timing_data: TimingData, performers_data: PerformersData) -> Dict[str, Any]:
    """Analyze text structure for multi-control indicators"""
    # Check for ad-hoc timing patterns
    timing_data.has_adhoc_timing = _detect_adhoc_timing(text, multi_config.adhoc_terms)
    timing_data.mixed_timing_detected = timing_data.has_adhoc_timing and timing_data.has_regular_frequency

    # Analyze paragraph structure
    paragraph_analysis = _analyze_paragraph_structure(text)

    # Analyze sequence markers
    sequence_analysis = _analyze_sequence_markers(
        text, multi_config.sequence_markers, multi_config.escalation_markers,
        timing_data.timing_candidates, performers_data.all_performers
    )

    return {
        "paragraph_analysis": paragraph_analysis,
        "sequence_analysis": sequence_analysis,
        "has_mixed_timing": timing_data.mixed_timing_detected
    }


def _detect_adhoc_timing(text: str, adhoc_terms: List[str]) -> bool:
    """Detect presence of ad-hoc timing terms"""
    return any(term.lower() in text.lower() for term in adhoc_terms)


def _create_action_groupings(categorized_actions: List[Dict], timing_data: TimingData,
                             performers_data: PerformersData) -> ActionGroupings:
    """Create groupings of actions by timing and performer"""
    timing_groups = _group_actions_by_timing(categorized_actions, timing_data.detected_frequencies)
    performer_groups = _group_actions_by_performer(categorized_actions, performers_data.all_performers)

    return ActionGroupings(timing_groups, performer_groups)


def _determine_multi_control_presence(groupings: ActionGroupings, timing_data: TimingData,
                                      structure_analysis: Dict, control_candidates: List[Dict],
                                      multi_config: MultiControlConfig) -> Dict[str, Any]:
    """Determine if multiple controls are present using various indicators"""
    paragraph_analysis = structure_analysis["paragraph_analysis"]
    sequence_analysis = structure_analysis["sequence_analysis"]

    # Check for distinct performers
    has_distinct_performers = len(groupings.performer_groups) > 1

    # Build multi-control indicators
    multi_control_indicators = {
        "distinct_timing": len(groupings.timing_groups) > 1,
        "mixed_timing": timing_data.mixed_timing_detected,
        "multi_frequency": timing_data.multi_frequency_detected,
        "distinct_paragraphs": paragraph_analysis.get("has_distinct_paragraphs", False),
        "control_sequence": sequence_analysis.get("has_control_sequence", False),
        "distinct_performers": has_distinct_performers
    }

    # Count positive indicators
    indicator_count = sum(1 for value in multi_control_indicators.values() if value)

    # Make final determination with enhanced logic
    is_multi_control = (
            len(groupings.timing_groups) > 1 or  # Different timing patterns
            timing_data.multi_frequency_detected or  # Multiple frequencies explicitly detected
            (timing_data.mixed_timing_detected and indicator_count >= 1) or  # Mixed timing with other indicator
            (paragraph_analysis.get("has_distinct_paragraphs",
                                    False) and indicator_count >= 1) or  # Structural separation with other indicator
            (has_distinct_performers and len(control_candidates) > 1) or  # Different performers doing different things
            (sequence_analysis.get("has_control_sequence", False) and not _is_escalation_sequence_check(
                sequence_analysis, multi_config))  # True control sequence
    )

    return {
        "is_multi_control": is_multi_control,
        "indicators": multi_control_indicators,
        "indicator_count": indicator_count,
        "has_distinct_performers": has_distinct_performers
    }


def _is_escalation_sequence_check(sequence_analysis: Dict, multi_config: MultiControlConfig) -> bool:
    """Check if sequence is primarily escalation based on analysis"""
    # This is a simplified check - could be enhanced based on sequence_analysis results
    return False  # Placeholder - implement based on specific sequence analysis


def _calculate_enhanced_confidence(detection_result: Dict, groupings: ActionGroupings,
                                   structure_analysis: Dict, control_candidates: List[Dict]) -> str:
    """Calculate confidence level for multi-control detection with enhanced factors"""
    if not detection_result["is_multi_control"]:
        return "low"

    strong_indicators = detection_result["indicator_count"]

    # Additional factors for confidence
    has_multiple_timing_groups = len(groupings.timing_groups) > 1
    has_multiple_performer_groups = len(groupings.performer_groups) > 1
    has_multiple_candidates = len(control_candidates) > 1

    # Convert indicator count to confidence level
    if strong_indicators >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    elif strong_indicators == MEDIUM_CONFIDENCE_THRESHOLD and has_multiple_candidates:
        return "medium"
    else:
        return "low"


def _assemble_final_detection_result(detection_result: Dict, control_candidates: List[Dict],
                                     groupings: ActionGroupings, structure_analysis: Dict,
                                     timing_data: TimingData, confidence: str) -> Dict[str, Any]:
    """Assemble the final detection result"""
    return {
        "detected": detection_result["is_multi_control"],
        "count": len(control_candidates) if detection_result["is_multi_control"] else 1,
        "candidates": control_candidates,
        "has_multi_frequency": timing_data.multi_frequency_detected,
        "timing_groups": list(groupings.timing_groups.keys()),
        "has_adhoc_component": timing_data.has_adhoc_timing,
        "has_sequence_markers": structure_analysis["sequence_analysis"].get("has_sequence_markers", False),
        "has_distinct_performers": detection_result["has_distinct_performers"],
        "has_distinct_paragraphs": structure_analysis["paragraph_analysis"].get("has_distinct_paragraphs", False),
        "multi_control_indicators": detection_result["indicators"],
        "confidence": confidence
    }


def _is_escalation_action(context: str, escalation_markers: List[str]) -> bool:
    """
    Uses contextual analysis to identify escalation actions.

    Determine if an action is part of an escalation path rather than a primary control.
    """
    # Check for escalation markers
    has_escalation_marker = any(marker.lower() in context.lower() for marker in escalation_markers)

    # Check for conditional phrases that indicate escalation
    has_conditional = any(re.search(pattern, context.lower()) for pattern in CONDITIONAL_PATTERNS)

    # Look for escalation-specific verbs
    has_escalation_verb = any(re.search(r'\b' + re.escape(verb) + r'\b', context.lower())
                              for verb in ESCALATION_VERBS)

    return has_escalation_marker or (has_conditional and has_escalation_verb)


def _analyze_paragraph_structure(text: str) -> Dict[str, Any]:
    """
    Analyzes paragraph structure to identify distinct control statements.

    Analyze the paragraph structure to identify if there are distinct
    control statements separated by paragraph breaks.
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

    for para in clean_paragraphs:
        # Check if paragraph contains a main control verb at the beginning
        para_lower = para.lower()
        contains_control_verb = any(re.search(r'\b' + verb + r'\b', para_lower[:50])
                                    for verb in CONTROL_VERBS)

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


def _analyze_sequence_markers(text: str, sequence_markers: List[str],
                              escalation_markers: List[str],
                              timing_candidates: List[Dict],
                              all_performers: List[str]) -> Dict[str, Any]:
    """
    Analyzes sequence markers to distinguish control sequences from process steps.

    Analyze sequence markers to differentiate between process steps within
    a single control and sequences of separate controls.
    """
    text_lower = text.lower()

    # Find all sequence markers
    sequence_positions = []
    for marker in sequence_markers:
        for match in re.finditer(r'\b' + re.escape(marker) + r'\b', text_lower):
            sequence_positions.append({
                "marker": marker,
                "span": [match.start(), match.end()],
                "context": _get_surrounding_context(text, match.start(), match.end(), DEFAULT_CONTEXT_WINDOW)
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

        # Check if marker is followed by different timing or performer
        follows_timing = _check_timing_follows_marker(seq, timing_candidates)
        follows_performer = _check_performer_follows_marker(seq, all_performers, text_lower)

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


def _check_timing_follows_marker(seq: Dict, timing_candidates: List[Dict]) -> bool:
    """Check if timing pattern follows a sequence marker"""
    for timing in timing_candidates:
        timing_span = timing.get("span", [0, 0])
        # Check if timing comes after this sequence marker within reasonable distance
        if timing_span[0] > seq["span"][1] and timing_span[0] <= seq["span"][1] + DEFAULT_CONTEXT_WINDOW:
            return True
    return False


def _check_performer_follows_marker(seq: Dict, all_performers: List[str], text_lower: str) -> bool:
    """Check if performer follows a sequence marker"""
    for performer in all_performers:
        performer_pos = text_lower.find(performer.lower(), seq["span"][1])
        if performer_pos > 0 and performer_pos <= seq["span"][1] + DEFAULT_CONTEXT_WINDOW:
            return True
    return False


def _group_actions_by_timing(categorized_actions: List[Dict],
                             detected_frequencies: List[str]) -> Dict[str, List[Dict]]:
    """
    Groups actions by timing patterns with enhanced handling.

    Group actions by their timing pattern to identify distinct controls.
    Enhanced to handle ad-hoc alongside regular frequencies and complex timing.
    """
    timing_groups = {}

    # First pass: group by explicit frequency
    for action_info in categorized_actions:
        if action_info["is_escalation"]:
            continue  # Skip escalation actions for grouping

        timing = action_info.get("timing", {})
        timing_key = timing.get("frequency", "unknown")
        timing_method = timing.get("method", "")

        # Add null safety check before calling .lower()
        if timing_key is None:
            timing_key = "unknown"

        # Special case for handling ad-hoc vs. regular frequencies
        timing_key = _normalize_timing_key(timing_key, timing_method)

        # If no explicit timing, try to infer from context
        if timing_key == "unknown":
            timing_key = _infer_timing_from_context(action_info, detected_frequencies)

        # Ensure the group exists
        if timing_key not in timing_groups:
            timing_groups[timing_key] = []

        timing_groups[timing_key].append(action_info)

    return timing_groups


def _normalize_timing_key(timing_key: str, timing_method: str) -> str:
    """Normalize timing key for consistent grouping"""
    if timing_key and ("adhoc" in timing_key.lower() or "as needed" in timing_key.lower()):
        return "adhoc"  # Normalize ad-hoc terms
    elif "event" in timing_method or "trigger" in timing_method:
        return "event_triggered"
    return timing_key


def _infer_timing_from_context(action_info: Dict, detected_frequencies: List[str]) -> str:
    """Infer timing from action context when not explicitly found"""
    context = action_info.get("context", "").lower()

    # Check if context suggests a timing pattern
    for freq in detected_frequencies:
        if freq.lower() in context:
            return freq

    return "unknown"


def _group_actions_by_performer(categorized_actions: List[Dict],
                                all_performers: List[str]) -> Dict[str, List[Dict]]:
    """
    Groups actions by associated performers.

    Group actions by their associated performer to identify distinct controls.
    """
    performer_groups = {}

    for action_info in categorized_actions:
        if action_info["is_escalation"]:
            continue  # Skip escalation actions for grouping

        performer = action_info.get("performer", "unknown")

        # Handle cases where performer wasn't explicitly associated
        if not performer or performer == "unknown":
            performer = _find_performer_in_context(action_info, all_performers)

        # Ensure the group exists
        if performer not in performer_groups:
            performer_groups[performer] = []

        performer_groups[performer].append(action_info)

    return performer_groups


def _find_performer_in_context(action_info: Dict, all_performers: List[str]) -> str:
    """Find performer in action context when not explicitly associated"""
    action_context = action_info.get("context", "").lower()

    for possible_performer in all_performers:
        if possible_performer.lower() in action_context:
            return possible_performer

    return "unknown"


def _build_enhanced_control_candidates(groupings: ActionGroupings, categorized_actions: List[Dict],
                                       performers_data: PerformersData, text: str,
                                       timing_data: TimingData, multi_config: MultiControlConfig) -> List[Dict]:
    """
    Enhanced control candidate building with better handling of mixed patterns.

    Build control candidates from timing groups, performer groups, and actions.
    Enhanced to better handle mixed timing patterns and performers.
    """
    control_candidates = []

    # Build candidates from timing groups
    control_candidates.extend(_build_candidates_from_timing_groups(groupings.timing_groups, performers_data))

    # Add candidates from distinct performer groups not covered
    control_candidates.extend(_build_candidates_from_performer_groups(
        groupings.performer_groups, control_candidates, performers_data
    ))

    # Check for ad-hoc controls mixed with regular controls
    control_candidates.extend(_build_adhoc_candidates(
        text, timing_data, categorized_actions, performers_data, multi_config, control_candidates
    ))

    # Deduplicate candidates
    return _deduplicate_control_candidates(control_candidates)


def _build_candidates_from_timing_groups(timing_groups: Dict[str, List[Dict]],
                                         performers_data: PerformersData) -> List[Dict]:
    """Build candidates from timing groups"""
    candidates = []

    for frequency, actions in timing_groups.items():
        for action in actions:
            associated_who = action.get("performer", "")

            # If no performer was associated, use primary if available
            if not associated_who and performers_data.all_performers:
                associated_who = performers_data.all_performers[0]

            candidates.append({
                "who": associated_who,
                "what": action["text"],
                "when": frequency,
                "is_escalation": False,
                "span": action.get("span", [0, 0]),
                "action": action["action"]  # Include original action data
            })

    return candidates


def _build_candidates_from_performer_groups(performer_groups: Dict[str, List[Dict]],
                                            existing_candidates: List[Dict],
                                            performers_data: PerformersData) -> List[Dict]:
    """Build candidates from performer groups not already covered"""
    candidates = []
    performers_added = set(c["who"].lower() for c in existing_candidates if c["who"])

    for performer, actions in performer_groups.items():
        # Skip if performer's action is already covered
        if performer.lower() in performers_added:
            # Check that all actions for this performer are covered
            performer_action_texts = set(a["text"].lower() for a in actions)
            candidate_action_texts = set(
                c["what"].lower() for c in existing_candidates
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
                   for c in existing_candidates):
                continue

            # Find timing for this action
            timing = action.get("timing", {})
            frequency = timing.get("frequency", "unknown")

            candidates.append({
                "who": performer,
                "what": action["text"],
                "when": frequency,
                "is_escalation": False,
                "span": action.get("span", [0, 0]),
                "action": action["action"]
            })

    return candidates


def _build_adhoc_candidates(text: str, timing_data: TimingData, categorized_actions: List[Dict],
                            performers_data: PerformersData, multi_config: MultiControlConfig,
                            existing_candidates: List[Dict]) -> List[Dict]:
    """Build candidates for ad-hoc controls mixed with regular controls"""
    candidates = []

    if not timing_data.has_adhoc_timing or "adhoc" in {c["when"] for c in existing_candidates}:
        return candidates

    text_lower = text.lower()
    adhoc_context = _find_adhoc_context(text, text_lower, multi_config.adhoc_terms)

    if not adhoc_context:
        return candidates

    # Find actions in adhoc context not already covered
    for action_info in categorized_actions:
        action_text = action_info["text"].lower()

        # Skip escalation actions and already covered actions
        if (action_info["is_escalation"] or
                any(c["what"].lower() == action_text for c in existing_candidates)):
            continue

        # Check if this action appears in adhoc context
        if action_text in adhoc_context:
            associated_who = _find_performer_for_adhoc_action(
                action_info, adhoc_context, performers_data.all_performers
            )

            candidates.append({
                "who": associated_who,
                "what": action_info["text"],
                "when": "adhoc",
                "is_escalation": False,
                "span": action_info.get("span", [0, 0]),
                "action": action_info["action"]
            })

    return candidates


def _find_adhoc_context(text: str, text_lower: str, adhoc_terms: List[str]) -> str:
    """Find context around ad-hoc terms"""
    for term in adhoc_terms:
        if term.lower() in text_lower:
            adhoc_pos = text_lower.find(term.lower())
            if adhoc_pos >= 0:
                return _get_surrounding_context(text, adhoc_pos, adhoc_pos + len(term), EXTENDED_CONTEXT_WINDOW)
    return ""


def _find_performer_for_adhoc_action(action_info: Dict, adhoc_context: str,
                                     all_performers: List[str]) -> str:
    """Find performer for ad-hoc action"""
    associated_who = action_info.get("performer", "")

    # If no performer was associated, try to find one in adhoc context
    if not associated_who:
        for performer in all_performers:
            if performer.lower() in adhoc_context:
                associated_who = performer
                break

    # If still no performer, use primary
    if not associated_who and all_performers:
        associated_who = all_performers[0]

    return associated_who


def _deduplicate_control_candidates(control_candidates: List[Dict]) -> List[Dict]:
    """Deduplicate candidates by finding unique (who, what, when) combinations"""
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


def _find_timing_for_action(action_text: str, action_context: str,
                            timing_candidates: List[Dict]) -> Dict:
    """
    Enhanced timing association with better context analysis.

    Associate an action with its timing information based on textual context.
    Enhanced to handle more timing patterns and contextual clues.
    """
    action_lower = action_text.lower()
    context_lower = action_context.lower()

    # Extract action start position in the context
    action_pos_in_context = context_lower.find(action_lower)

    best_match = _find_best_timing_match(context_lower, action_pos_in_context, timing_candidates)

    # If no timing found in direct context, try inference
    if not best_match:
        best_match = _infer_timing_from_indicators(context_lower)

    # Return empty dict if no timing found
    return best_match if best_match else {"text": "", "frequency": "unknown", "method": ""}


def _find_best_timing_match(context_lower: str, action_pos_in_context: int,
                            timing_candidates: List[Dict]) -> Optional[Dict]:
    """Find the best timing match from candidates"""
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

            # Calculate proximity score
            distance = abs(candidate_pos - action_pos_in_context)
            proximity_score = EXTENDED_CONTEXT_WINDOW - min(distance, EXTENDED_CONTEXT_WINDOW)

            # Determine timing relationship
            timing_before_action = candidate_pos < action_pos_in_context
            relationship_bonus = 1.2 if timing_before_action else 1.0

            # Calculate final score
            match_score = (proximity_score / EXTENDED_CONTEXT_WINDOW) * relationship_bonus

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

    return best_match


def _infer_timing_from_indicators(context_lower: str) -> Optional[Dict]:
    """Infer timing from indicators when no explicit timing found"""
    for indicator in TIMING_INDICATORS:
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
                frequency = _determine_frequency_from_phrase(timing_phrase)

                return {
                    "text": timing_phrase,
                    "frequency": frequency,
                    "method": "inferred_from_context",
                    "is_vague": False,
                    "proximity_score": 0.5  # Lower confidence for inferred timing
                }

    return None


def _determine_frequency_from_phrase(timing_phrase: str) -> str:
    """Determine frequency from timing phrase"""
    if "daily" in timing_phrase:
        return "daily"
    elif "weekly" in timing_phrase:
        return "weekly"
    elif "monthly" in timing_phrase:
        return "monthly"
    elif "quarterly" in timing_phrase:
        return "quarterly"
    elif "annually" in timing_phrase or "yearly" in timing_phrase:
        return "annually"
    return "unknown"


def _find_performer_for_action(action_span: List[int], all_performers: List[str],
                               performer_positions: Dict[str, Dict],
                               text_lower: str) -> str:
    """
    Enhanced performer association with improved proximity analysis.

    Find the performer most likely associated with a specific action based on proximity
    and syntactic relationships.
    """
    if not all_performers:
        return ""

    # Get action position
    action_start, action_end = action_span

    # Find closest performer using enhanced logic
    closest_performer = _find_closest_performer(
        action_start, action_end, all_performers, performer_positions, text_lower
    )

    # If we found a performer within reasonable distance
    if closest_performer and closest_performer["distance"] <= MAX_PERFORMER_DISTANCE:
        return closest_performer["performer"]

    # If no close performer found, return the primary performer
    return all_performers[0] if all_performers else ""


def _find_closest_performer(action_start: int, action_end: int, all_performers: List[str],
                            performer_positions: Dict[str, Dict], text_lower: str) -> Optional[Dict]:
    """Find the closest performer to an action"""
    closest_performer = None
    min_distance = float('inf')

    for performer in all_performers:
        performer_lower = performer.lower()
        position_info = performer_positions.get(performer_lower)

        if not position_info:
            continue

        performer_span = position_info.get("span", [0, 0])
        performer_start, performer_end = performer_span

        # Calculate distance and penalties
        distance_info = _calculate_performer_distance(
            action_start, action_end, performer_start, performer_end, text_lower
        )

        if distance_info["distance_with_penalty"] < min_distance:
            min_distance = distance_info["distance_with_penalty"]
            closest_performer = {
                "performer": performer,
                "distance": distance_info["raw_distance"],
                "has_sentence_boundary": distance_info["has_sentence_boundary"]
            }

    return closest_performer


def _calculate_performer_distance(action_start: int, action_end: int,
                                  performer_start: int, performer_end: int, text_lower: str) -> Dict:
    """Calculate distance between performer and action with penalties"""
    # Look for performers before the action (higher priority)
    if performer_end <= action_start:
        raw_distance = action_start - performer_end
        text_between = text_lower[performer_end:action_start]
        has_sentence_boundary = "." in text_between
        distance_with_penalty = raw_distance * SENTENCE_BOUNDARY_PENALTY if has_sentence_boundary else raw_distance

    # If no performer before action, check after action (lower priority)
    elif performer_start >= action_end:
        raw_distance = performer_start - action_end
        text_between = text_lower[action_end:performer_start]
        has_sentence_boundary = "." in text_between

        # Apply larger penalty for performer after action with sentence boundary
        if has_sentence_boundary:
            distance_with_penalty = raw_distance * AFTER_ACTION_SENTENCE_PENALTY
        else:
            distance_with_penalty = raw_distance * AFTER_ACTION_PENALTY
    else:
        # Performer overlaps with action - shouldn't happen but handle gracefully
        raw_distance = 0
        has_sentence_boundary = False
        distance_with_penalty = 0

    return {
        "raw_distance": raw_distance,
        "distance_with_penalty": distance_with_penalty,
        "has_sentence_boundary": has_sentence_boundary
    }


def _get_surrounding_context(text: str, start: int, end: int, window_size: int = DEFAULT_CONTEXT_WINDOW) -> str:
    """
    Enhanced context extraction with boundary handling.

    Get the surrounding context of a phrase within the text.
    """
    if not text:
        return ""

    context_start = max(0, start - window_size)
    context_end = min(len(text), end + window_size)

    return text[context_start:context_end]


def _get_config_value(config: Dict, key: str, default_value: Any) -> Any:
    """
    Enhanced configuration value retrieval with type safety.

    Helper function to safely get a value from configuration with defaults.
    """
    return config.get(key, default_value)


def mark_possible_standalone_controls(text: str, nlp_model, config: Optional[Dict] = None) -> List[Dict]:
    """
    Enhanced standalone control detection with shared constants.

    Identify possible standalone control statements within a text.
    This is a simplified detection that can be used independently.
    """
    config = config or {}

    # Process text
    doc = nlp_model(text)

    # Split text into sentences
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

    # Identify possible standalone controls
    candidates = []

    for i, sent in enumerate(sentences):
        sent_doc = nlp_model(sent)

        # Check if sentence contains a key control verb
        contains_verb = any(token.lemma_.lower() in CONTROL_VERBS for token in sent_doc)

        # Check if sentence has a subject (likely performer)
        has_subject = any(token.dep_ == "nsubj" for token in sent_doc)

        # Calculate score based on features
        score = _calculate_control_score(contains_verb, has_subject, sent, i)

        # Check for timing indicators
        has_timing = _check_sentence_timing(sent)
        if has_timing:
            score += TIMING_SCORE_WEIGHT

        # Only add if minimum threshold is met
        if score > MINIMUM_CONTROL_SCORE:
            verb_tokens = [token for token in sent_doc if token.lemma_.lower() in CONTROL_VERBS]
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


def _calculate_control_score(contains_verb: bool, has_subject: bool, sent: str, position: int) -> float:
    """Calculate score for potential standalone control"""
    score = 0

    if contains_verb:
        score += VERB_SCORE_WEIGHT
    if has_subject:
        score += SUBJECT_SCORE_WEIGHT

    # Boost score if sentence starts with a capitalized word (likely new thought)
    if sent[0].isupper() and position > 0:
        score += NEW_SENTENCE_SCORE_WEIGHT

    return score


def _check_sentence_timing(sent: str) -> bool:
    """Check if sentence contains timing indicators"""
    return bool(re.search(
        r'\b(daily|weekly|monthly|quarterly|annually|every|each|when|as needed)\b',
        sent, re.IGNORECASE
    ))