import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Constants
CONTROL_VERBS = [
    "review", "approve", "verify", "check", "validate", "reconcile",
    "examine", "analyze", "evaluate", "assess", "monitor", "track",
    "investigate", "inspect", "audit", "oversee", "supervise", "ensure",
    "perform", "execute", "conduct", "disable", "enforce", "generate",
    "address", "compare", "maintain", "identify", "correct", "update",
    "submit", "complete", "prepare", "provide", "confirm"
]

HUMAN_INDICATORS = [
    # Roles and titles
    "manager", "director", "supervisor", "analyst", "specialist", "officer",
    "coordinator", "lead", "team", "staff", "personnel", "employee",
    "individual", "person", "accountant", "controller", "auditor", "administrator",
    "executive", "chief", "head", "president", "ceo", "cfo", "cio", "vp",
    "vice president", "secretary", "treasurer", "owner", "preparer", "reviewer",
    # Departments and groups
    "finance", "accounting", "accounts receivable", "accounts payable", "treasury",
    "financial reporting", "tax", "payroll", "billing", "credit", "collections",
    "committee", "department", "group", "unit", "division", "office", "board",
    # Specific roles
    "financial controller", "financial analyst", "budget analyst", "compliance",
    "audit", "risk", "governance", "oversight", "management", "leadership"
]

SYSTEM_INDICATORS = [
    "system", "application", "software", "platform", "database", "server",
    "program", "script", "job", "batch", "workflow", "algorithm", "automated",
    "automatic", "tool", "module", "interface", "api", "service", "function",
    "scheduled", "timer", "daemon", "bot", "routine", "task"
]

NON_PERFORMER_INDICATORS = [
    "limit", "threshold", "policy", "procedure", "standard", "regulation",
    "account", "transaction", "balance", "report", "document", "record",
    "exception", "error", "issue", "finding", "discrepancy", "review",
    "control", "entry", "activity", "access", "instance", "item",
    "questionnaire", "form", "template", "results", "assessment",
    "risk", "compliance", "process", "monitoring", "levels", "ensure"
]

TEMPORAL_PREFIXES = ["annually", "monthly", "quarterly", "weekly", "daily", "periodically", "regularly"]

PROBLEM_PATTERNS = [
    r'\brisk\b', r'\baccess levels\b', r'\bensure compliance\b', r'\bmonitor\b',
    r'\blimit\b', r'\bprocess\b', r'\bcontrol\b', r'\bcompliance\b', r'\baccess\b',
    r'\blevels\b', r'\bensure\b'
]

NON_PERFORMER_TERMS = [
    "exception", "error", "transaction", "record", "issue",
    "item", "reconciliation", "document", "report",
    "questionnaire", "form", "template", "results",
    "risk", "compliance", "access", "levels", "limit", "process", "customer", "data"
]

# Confidence score constants
BASE_CONFIDENCE_SCORE = 0.7
HUMAN_TYPE_BOOST = 0.2
MANAGERIAL_ROLE_BOOST = 0.2
SPECIFIC_FINANCE_ROLE_BOOST = 0.3
TEAM_BOOST = 0.1
COMMITTEE_BOOST = 0.15
ACRONYM_TEAM_BOOST = 0.2
SYSTEM_TYPE_BOOST = 0.1
NON_PERFORMER_PENALTY = 0.1
PASSIVE_VOICE_PENALTY = 0.1
MAIN_SUBJECT_BOOST = 1.2
BY_PHRASE_BOOST = 1.3
RESPONSIBILITY_BOOST = 1.3
TEMPORAL_BOOST = 1.2
FALLBACK_PENALTY = 0.7
REGEX_FALLBACK_PENALTY = 0.6
AUTOMATED_CONTROL_BOOST = 0.2
MANUAL_CONTROL_BOOST = 0.2
EXECUTIVE_FREQUENCY_PENALTY = 0.1
CONTINUOUS_SYSTEM_BOOST = 0.1
MINIMUM_CONFIDENCE = 0.1
MAXIMUM_CONFIDENCE = 1.0
MINIMUM_FALLBACK_CONFIDENCE = 0.3


@dataclass
class DetectionParameters:
    """Parameter object for WHO detection configuration"""
    control_type: Optional[str] = None
    frequency: Optional[str] = None
    existing_keywords: Optional[List[str]] = None


def enhanced_who_detection_v2(text: str, nlp_model, control_type: Optional[str] = None,
                              frequency: Optional[str] = None, existing_keywords: Optional[List[str]] = None):
    """
    Main orchestrator function that coordinates WHO detection through multiple strategies.

    Enhanced WHO detection with dependency parsing focus for more robust performer identification.
    This function uses spaCy's dependency parsing to identify performers in control descriptions,
    handling various sentence structures including active voice, passive voice, temporal prefixes,
    responsibility clauses, and complex phrases.
    """
    if not text or text.strip() == '':
        return {
            "primary": None,
            "secondary": [],
            "confidence": 0,
            "message": "No text provided"
        }

    try:
        doc = nlp_model(text)
        params = DetectionParameters(control_type, frequency, existing_keywords)

        # Initialize candidates list
        who_candidates = []

        # Execute detection strategies
        who_candidates.extend(_detect_config_keywords(text, params))

        verb_objects = _identify_verb_objects(doc, params)
        main_subjects = _find_main_subjects(doc, verb_objects, params)
        who_candidates.extend(_process_main_subjects(main_subjects, params))

        who_candidates.extend(_detect_prepositional_phrases(doc, params))
        who_candidates.extend(_detect_explicit_responsibility(doc, params))
        who_candidates.extend(_detect_temporal_patterns(text, params))

        if not who_candidates:
            who_candidates.extend(_fallback_noun_chunk_analysis(doc, params))

        if not who_candidates:
            who_candidates.extend(_fallback_regex_patterns(text, params))

        who_candidates = _validate_candidates(who_candidates, verb_objects, nlp_model)

        return _assemble_final_result(who_candidates, params)

    except Exception as e:
        print(f"Error in enhanced WHO detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "primary": None,
            "secondary": [],
            "confidence": 0,
            "message": f"Error: {str(e)}"
        }


def _detect_config_keywords(text: str, params: DetectionParameters) -> List[Dict]:
    """Direct search for configuration-provided keywords"""
    if not params.existing_keywords:
        return []

    config_candidates = []
    for keyword in params.existing_keywords:
        # Look for this specific keyword in the text, case-insensitively
        # Use regex with word boundaries
        pattern = r'\b' + re.escape(keyword) + r'\b'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Found a direct match from config
            entity_text = match.group(0)  # Preserve original capitalization

            # Try to determine if this is a main part of the control
            surrounding_context = text[max(0, match.start() - 30):min(len(text), match.end() + 30)]

            # Check if a control verb appears near the keyword
            has_nearby_verb = any(verb in surrounding_context.lower() for verb in CONTROL_VERBS)

            # Give a strong confidence score, especially if near a verb
            base_confidence = 0.8  # Already quite high
            if has_nearby_verb:
                base_confidence = 0.9  # Even higher with a verb

            config_candidates.append({
                "text": entity_text,
                "verb": "config_match",
                "type": "human",  # Assume human for config keywords
                "score": base_confidence,
                "position": match.start(),
                "role": "primary",
                "detection_method": "direct_config_match"
            })

    return config_candidates


def _identify_verb_objects(doc, params: DetectionParameters) -> List[Dict]:
    """Find all control verbs and identify objects of control verbs to exclude them as candidates"""
    # Add custom keywords from config if available
    control_verbs = CONTROL_VERBS.copy()
    if params.existing_keywords:
        for keyword in params.existing_keywords:
            # Extract verbs from keywords (if they contain multiple words)
            words = keyword.lower().split()
            for word in words:
                # Check if it's likely a verb
                if word.endswith(('s', 'ed', 'ing')) or word in control_verbs:
                    if word not in control_verbs:
                        control_verbs.append(word)

    verb_objects = []
    for token in doc:
        if token.lemma_.lower() in control_verbs:
            # Find all objects of this verb
            for child in token.children:
                if child.dep_ in ["dobj", "pobj"]:
                    # Get the full noun phrase
                    for chunk in doc.noun_chunks:
                        if child.i >= chunk.start and child.i < chunk.end:
                            verb_objects.append({
                                "text": chunk.text.lower(),
                                "span": (chunk.start, chunk.end),
                                "verb": token.lemma_
                            })

    return verb_objects


def _find_main_subjects(doc, verb_objects: List[Dict], params: DetectionParameters) -> List[Dict]:
    """Handle complex sentences by finding main subjects"""
    main_subjects = []

    for sent in doc.sents:
        # Find the root verbs and other main verbs of the sentence
        main_verbs = []
        for token in sent:
            # Include ROOT verbs and their direct verbal dependents
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                main_verbs.append(token)
                # Look for coordinated verbs
                for child in token.children:
                    if child.dep_ == "conj" and child.pos_ == "VERB":
                        main_verbs.append(child)

            # Also include common verbal constructions like "has processes to monitor"
            elif token.pos_ == "VERB" and token.head.pos_ == "VERB" and token.dep_ in ["xcomp", "advcl", "ccomp"]:
                # Get the highest verb in this chain
                current = token.head
                while current.head.pos_ == "VERB" and current.dep_ in ["xcomp", "advcl", "ccomp"]:
                    current = current.head

                # See if this parent verb is a ROOT
                if current.dep_ == "ROOT":
                    main_verbs.append(current)

        # Now find subjects of these main verbs
        for verb in main_verbs:
            for child in verb.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    # Get the full noun phrase
                    for chunk in doc.noun_chunks:
                        if child.i >= chunk.start and child.i < chunk.end:
                            # Check that this isn't a misidentified object
                            if not any(chunk.text.lower() == obj["text"] for obj in verb_objects):
                                main_subjects.append({
                                    "text": chunk.text,
                                    "span": (chunk.start, chunk.end),
                                    "is_passive": child.dep_ == "nsubjpass",
                                    "verb": verb.text,
                                    "verb_lemma": verb.lemma_
                                })

    return main_subjects


def _process_main_subjects(main_subjects: List[Dict], params: DetectionParameters) -> List[Dict]:
    """Process main subjects as high-quality candidates"""
    candidates = []

    for subject in main_subjects:
        subject_text = subject["text"]

        # Skip common non-performers
        if any(term in subject_text.lower() for term in NON_PERFORMER_TERMS):
            continue

        # Classify the entity
        entity_type = classify_entity_type(subject_text, None, params.existing_keywords)

        # Skip if classified as a non-performer
        if entity_type == "non-performer":
            continue

        # Score this candidate - boost for main subjects
        confidence = calculate_who_confidence({
            "text": subject_text,
            "type": entity_type,
            "is_passive": subject.get("is_passive", False)
        }, params.control_type, params.frequency) * MAIN_SUBJECT_BOOST

        # Determine role
        role = "primary"
        # If this is a passive subject, it's likely an object being acted upon, not an actor
        if subject.get("is_passive", False):
            # Check if it seems like a legitimate performer despite passive voice
            if entity_type == "human" and any(term in subject_text.lower() for term in
                                              ["manager", "director", "team", "staff", "specialist",
                                               "committee"]):
                role = "primary"  # Still consider it primary if it's clearly a person/role
            else:
                role = "secondary"

                # Extra penalty for passive non-performers
                if entity_type != "human":
                    confidence *= 0.5

        # Create candidate
        candidate = {
            "text": subject_text,
            "verb": subject["verb"],
            "type": entity_type,
            "score": confidence,
            "position": subject["span"][0],
            "role": role,
            "detection_method": "main_subject_of_verb"
        }

        candidates.append(candidate)

    return candidates


def _detect_prepositional_phrases(doc, params: DetectionParameters) -> List[Dict]:
    """Find performers in prepositional phrases (e.g., 'by the team')"""
    by_phrases = []
    for token in doc:
        if token.text.lower() == "by" and token.dep_ == "prep":
            for child in token.children:
                if child.dep_ == "pobj":
                    # Get the full noun phrase
                    for chunk in doc.noun_chunks:
                        if child.i >= chunk.start and child.i < chunk.end:
                            by_phrases.append({
                                "text": chunk.text,
                                "span": (chunk.start, chunk.end),
                                "verb": token.head.text if token.head.pos_ == "VERB" else None
                            })

    # Process "by X" phrases as high-quality candidates
    candidates = []
    for phrase in by_phrases:
        # Classify the entity
        entity_text = phrase["text"]
        entity_type = classify_entity_type(entity_text, None, params.existing_keywords)

        # Skip if classified as a non-performer
        if entity_type == "non-performer":
            continue

        # High confidence since "by X" is a strong indicator
        confidence = calculate_who_confidence({
            "text": entity_text,
            "type": entity_type,
            "is_passive": False
        }, params.control_type, params.frequency) * BY_PHRASE_BOOST

        candidate = {
            "text": entity_text,
            "verb": phrase.get("verb", "passive_by_phrase"),
            "type": entity_type,
            "score": min(MAXIMUM_CONFIDENCE, confidence),
            "position": phrase["span"][0],
            "role": "primary",  # "by X" usually indicates the primary performer
            "detection_method": "by_prepositional_phrase"
        }

        candidates.append(candidate)

    return candidates


def _detect_explicit_responsibility(doc, params: DetectionParameters) -> List[Dict]:
    """Find explicit role declarations (e.g., 'X is responsible for...')"""
    candidates = []

    for token in doc:
        if token.lemma_.lower() in ["responsible", "accountable", "tasked"]:
            # Check for "is responsible for" pattern
            if token.head.lemma_.lower() in ["be", "is", "are", "was", "were"]:
                for child in token.head.children:
                    if child.dep_ == "nsubj":
                        # Get the full noun phrase
                        for chunk in doc.noun_chunks:
                            if child.i >= chunk.start and child.i < chunk.end:
                                entity_text = chunk.text
                                entity_type = classify_entity_type(entity_text, None, params.existing_keywords)

                                # Skip if classified as a non-performer
                                if entity_type == "non-performer":
                                    continue

                                # This is a very strong indicator of a performer
                                confidence = calculate_who_confidence({
                                    "text": entity_text,
                                    "type": entity_type,
                                    "is_passive": False
                                }, params.control_type, params.frequency) * RESPONSIBILITY_BOOST

                                candidate = {
                                    "text": entity_text,
                                    "verb": "responsible/accountable",
                                    "type": entity_type,
                                    "score": min(MAXIMUM_CONFIDENCE, confidence),
                                    "position": chunk.start,
                                    "role": "primary",
                                    "detection_method": "explicit_responsibility"
                                }

                                candidates.append(candidate)
                                break

    return candidates


def _detect_temporal_patterns(text: str, params: DetectionParameters) -> List[Dict]:
    """Handle special cases like temporal prefixes"""
    candidates = []

    # Check if the text starts with a temporal prefix
    text_lower = text.lower()
    has_temporal_prefix = False
    matched_prefix = None

    for prefix in TEMPORAL_PREFIXES:
        if text_lower.startswith(prefix) or text_lower.startswith(prefix + ","):
            has_temporal_prefix = True
            matched_prefix = prefix
            break

    if has_temporal_prefix and matched_prefix:
        # Find where the temporal prefix ends (including any comma)
        prefix_end = len(matched_prefix)
        if text_lower[prefix_end:prefix_end + 1] == ",":
            prefix_end += 1

        # Look for "the [team name]" after the prefix
        team_match = re.search(r'\s+the\s+([A-Z][A-Za-z0-9]*\s+(?:team|group|committee|department|unit|office))',
                               text[prefix_end:], re.IGNORECASE)

        if team_match:
            # Extract just the team name (including "the")
            team_text = "the " + team_match.group(1)

            # Classify and score
            entity_type = classify_entity_type(team_text, None, params.existing_keywords)

            confidence = calculate_who_confidence({
                "text": team_text,
                "type": entity_type,
                "is_passive": False
            }, params.control_type, params.frequency) * TEMPORAL_BOOST

            candidate = {
                "text": team_text,  # Just "the [Team Name]"
                "verb": "temporal_structure",
                "type": entity_type,
                "score": min(MAXIMUM_CONFIDENCE, confidence),
                "position": prefix_end + team_match.start(),
                "role": "primary",
                "detection_method": "temporal_prefix_explicit"
            }

            candidates.append(candidate)

    return candidates


def _fallback_noun_chunk_analysis(doc, params: DetectionParameters) -> List[Dict]:
    """If we found no candidates at all, try more selective noun chunk analysis"""
    candidates = []

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text

        # Apply strict filtering to avoid false positives
        # Only consider noun chunks that have clear human/organization indicators
        human_indicators = [
            "manager", "director", "team", "supervisor", "analyst", "specialist",
            "officer", "committee", "department", "board", "staff", "personnel"
        ]

        # Check for exact matches, not partial matches
        matches_indicator = False
        for indicator in human_indicators:
            if re.search(r'\b' + re.escape(indicator) + r'\b', chunk_text.lower()):
                matches_indicator = True
                break

        if matches_indicator:
            # Classify entity type with stricter criteria
            entity_type = classify_entity_type(chunk_text, None, params.existing_keywords)

            # Only accept human or system entities
            if entity_type in ["human", "system"]:
                # Calculate confidence with a penalty since this is a fallback
                confidence = calculate_who_confidence({
                    "text": chunk_text,
                    "type": entity_type,
                    "is_passive": False
                }, params.control_type, params.frequency) * FALLBACK_PENALTY

                if confidence > MINIMUM_FALLBACK_CONFIDENCE:
                    candidate = {
                        "text": chunk_text,
                        "verb": "unknown",
                        "type": entity_type,
                        "score": confidence,
                        "position": chunk.start,
                        "role": "primary",  # Assign primary since we have no other candidates
                        "detection_method": "noun_chunk_fallback"
                    }

                    candidates.append(candidate)

    return candidates


def _fallback_regex_patterns(text: str, params: DetectionParameters) -> List[Dict]:
    """If still no candidates, use regex for team names and acronyms (with stricter filtering)"""
    candidates = []

    # Use regex to find different types of team names and acronyms
    org_patterns = [
        # Multi-word capitalized names like "Risk Management Team"
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:Team|Group|Department|Unit|Committee|Office))',

        # Acronym team names like "MCO Team"
        r'([A-Z]{2,}\s+(?:Team|Group|Department|Unit|Committee|Office))',

        # Variations with "the" prefix
        r'the\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:Team|Group|Department|Unit|Committee|Office))',
        r'the\s+([A-Z]{2,}\s+(?:Team|Group|Department|Unit|Committee|Office))',
    ]

    for pattern in org_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            match_text = match.group(0)  # Use full match including "the" if present
            entity_type = classify_entity_type(match_text, None, params.existing_keywords)

            # Calculate confidence with a penalty since this is a fallback
            confidence = calculate_who_confidence({
                "text": match_text,
                "type": entity_type,
                "is_passive": False
            }, params.control_type, params.frequency) * REGEX_FALLBACK_PENALTY

            if confidence > MINIMUM_FALLBACK_CONFIDENCE:
                candidate = {
                    "text": match_text,
                    "verb": "unknown",
                    "type": entity_type,
                    "score": confidence,
                    "position": match.start(),
                    "role": "primary",
                    "detection_method": "regex_fallback"
                }

                candidates.append(candidate)
                break  # Only take the first match from regex to avoid duplicates

        if candidates:
            break  # Stop looking for patterns if we found any

    return candidates


def _validate_candidates(who_candidates: List[Dict], verb_objects: List[Dict], nlp_model) -> List[Dict]:
    """Do a final validation of candidates to filter out false positives"""
    if not who_candidates:
        return []

    validated_candidates = []

    for candidate in who_candidates:
        # Skip candidates that are actually objects of control verbs
        if any(candidate["text"].lower() == obj["text"] for obj in verb_objects):
            continue

        # Validate against common false positive patterns
        false_positive_patterns = [
            r'\brisk\b', r'\baccess levels\b', r'\bensure compliance\b', r'\bmonitor\b',
            r'\blimit\b', r'\bprocess\b', r'\bcontrol\b'
        ]

        is_false_positive = False
        for pattern in false_positive_patterns:
            if re.search(pattern, candidate["text"].lower()):
                is_false_positive = True
                break

        if is_false_positive:
            continue

        # Apply more validation for noun chunk candidates
        if candidate["detection_method"] in ["noun_chunk_fallback", "regex_fallback"]:
            # Check if this appears to be a legitimate performer
            if not is_likely_performer(candidate["text"], nlp_model):
                continue

        # Add validated candidate
        validated_candidates.append(candidate)

    return validated_candidates


def _assemble_final_result(who_candidates: List[Dict], params: DetectionParameters) -> Dict:
    """Generate message and assemble final result"""
    # Return default if no performers found
    if not who_candidates:
        return {
            "primary": {
                "text": "Unknown Performer",
                "verb": "unknown",
                "type": "unknown",
                "score": 0.0
            },
            "secondary": [],
            "confidence": 0,
            "message": "No performer detected"
        }

    # Sort candidates by score, then by position
    who_candidates.sort(key=lambda x: (-x["score"], x["position"]))

    # Get primary and secondary performers
    primary = who_candidates[0]
    secondary = who_candidates[1:]

    # Filter out duplicates from secondary
    unique_secondary = []
    primary_text = primary["text"].lower()

    for candidate in secondary:
        # Skip if duplicate of primary or already in unique_secondary
        if candidate["text"].lower() == primary_text:
            continue

        if not any(candidate["text"].lower() == s["text"].lower() for s in unique_secondary):
            unique_secondary.append(candidate)

    # Generate message
    message = _generate_consistency_message(primary, params.control_type)

    return {
        "primary": primary,
        "secondary": unique_secondary[:3],  # Limit to top 3 secondary performers
        "confidence": primary["score"],
        "message": message,
        "detection_methods": list(set(c["detection_method"] for c in who_candidates))
    }


def _generate_consistency_message(primary: Dict, control_type: Optional[str]) -> str:
    """Generate message checking for consistency with control type"""
    message = ""

    if control_type and primary["type"] != "unknown":
        control_type_lower = control_type.strip().lower()
        performer_type = primary["type"]

        if control_type_lower == "automated" and performer_type == "human":
            message = "Warning: Human performer detected for automated control"
        elif control_type_lower == "manual" and performer_type == "system":
            message = "Warning: System performer detected for manual control"

    return message


def is_likely_performer(text, nlp_model):
    """
    Uses syntactic analysis to distinguish performers from objects/concepts.

    Determine if a candidate text is likely to be a performer rather than
    an object or concept being acted upon.
    """
    # Check for common non-performer patterns
    non_performer_patterns = [
        r'\brisk\b', r'\baccess levels\b', r'\bensure compliance\b', r'\bmonitor\b',
        r'\blimit\b', r'\bprocess\b', r'\bcontrol\b', r'\bcompliance\b', r'\baccess\b',
        r'\blevels\b', r'\bensure\b'
    ]

    for pattern in non_performer_patterns:
        if re.search(pattern, text.lower()):
            return False

    # Check for syntactic patterns that suggest real performers
    doc = nlp_model(text)

    # A real performer is typically:
    # 1. A person, role, or organization
    # 2. Not an abstract concept or object

    # Check for performer characteristics
    has_determiner = any(token.pos_ == "DET" for token in doc)
    has_proper_noun = any(token.pos_ == "PROPN" for token in doc)
    has_performer_indicator = any(token.text.lower() in [
        "manager", "director", "supervisor", "analyst", "specialist",
        "officer", "coordinator", "team", "staff", "committee",
        "department", "group", "unit", "division"
    ] for token in doc)

    # Organizations often have capital letters
    has_capitalized_words = any(token.text[0].isupper() and len(token.text) > 1 for token in doc)

    # Real performers often:
    # - Have determiners (the, a)
    # - Have proper nouns or capitals
    # - Contain performer indicators
    return has_performer_indicator or (has_determiner and (has_proper_noun or has_capitalized_words))


def classify_entity_type(text, nlp_model, custom_keywords=None):
    """
    Classifies entities using keyword matching and NLP entity recognition.

    Classify an entity as human, system, or non-performer with support for custom keywords.
    """
    normalized_text = text.lower()

    # Check for problem patterns first
    if _is_problem_phrase(normalized_text):
        return "non-performer"

    # Check for non-performer indicators
    if _contains_non_performer_indicators(normalized_text):
        return "non-performer"

    # Check for human indicators
    if _contains_human_indicators(normalized_text, custom_keywords):
        return "human"

    # Check for system indicators
    if _contains_system_indicators(normalized_text):
        return "system"

    # Use NLP to check for person or organization entities
    if nlp_model:
        doc = nlp_model(text)
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG"):
                return "human"

    # Additional check for acronyms that might be team names (like "MCO")
    if re.match(r'\b[A-Z]{2,}\b', text) and len(text) <= 5:
        # Short uppercase sequence - likely an acronym for a team or department
        return "human"

    # Default to unknown if no clear classification
    return "unknown"


def _is_problem_phrase(normalized_text: str) -> bool:
    """Check for explicit non-performer problem phrases"""
    problem_phrases = [
        "monitor risk", "ensure compliance", "access levels", "risk limits",
        "compliance", "access", "limit", "process", "control"
    ]
    return any(phrase in normalized_text for phrase in problem_phrases)


def _contains_non_performer_indicators(normalized_text: str) -> bool:
    """Check if the entity contains non-performer indicators using word boundaries"""
    for indicator in NON_PERFORMER_INDICATORS:
        pattern = r'\b' + re.escape(indicator) + r'\b'
        if re.search(pattern, normalized_text):
            return True
    return False


def _contains_human_indicators(normalized_text: str, custom_keywords: Optional[List[str]]) -> bool:
    """Check if the entity contains human role indicators with custom keywords support"""
    # Base human indicators
    human_indicators = HUMAN_INDICATORS.copy()

    # Add custom keywords if provided
    if custom_keywords:
        for keyword in custom_keywords:
            kw_lower = keyword.lower()
            # If the keyword contains a human indicator, add it to human indicators
            if any(indicator in kw_lower for indicator in
                   ["team", "group", "committee", "manager", "director", "officer"]):
                if kw_lower not in human_indicators:
                    human_indicators.append(kw_lower)

            # Look for acronyms followed by team indicators
            matches = re.findall(r'([A-Z]{2,})\s+(?:team|group|committee)', keyword, re.IGNORECASE)
            if matches:
                for match in matches:
                    if match.lower() not in human_indicators:
                        human_indicators.append(match.lower() + " team")

    # Check if the entity contains human role indicators (with word boundaries)
    for indicator in human_indicators:
        pattern = r'\b' + re.escape(indicator) + r'\b'
        if re.search(pattern, normalized_text):
            return True
    return False


def _contains_system_indicators(normalized_text: str) -> bool:
    """Check if the entity contains system indicators using word boundaries"""
    for indicator in SYSTEM_INDICATORS:
        pattern = r'\b' + re.escape(indicator) + r'\b'
        if re.search(pattern, normalized_text):
            return True
    return False


def calculate_who_confidence(entity_info, control_type=None, frequency=None):
    """
    Calculates confidence score based on entity characteristics and context.

    Calculate confidence score for a WHO entity with improved scoring.
    """
    # Check for known problem patterns
    if _has_problem_patterns(entity_info["text"]):
        return MINIMUM_CONFIDENCE

    # Start with base score
    base_score = BASE_CONFIDENCE_SCORE

    # Apply entity type adjustments
    base_score = _apply_entity_type_scoring(base_score, entity_info)

    # Apply passive voice penalty
    if entity_info.get("is_passive", False):
        base_score -= PASSIVE_VOICE_PENALTY

    # Apply control type consistency adjustments
    base_score = _apply_control_type_consistency(base_score, entity_info, control_type)

    # Apply frequency consistency adjustments
    base_score = _apply_frequency_consistency(base_score, entity_info, frequency)

    # Ensure score is within valid range
    return max(MINIMUM_CONFIDENCE, min(MAXIMUM_CONFIDENCE, base_score))


def _has_problem_patterns(text: str) -> bool:
    """Check for known problem patterns"""
    for pattern in PROBLEM_PATTERNS:
        if re.search(pattern, text.lower()):
            return True
    return False


def _apply_entity_type_scoring(base_score: float, entity_info: Dict) -> float:
    """Apply scoring adjustments based on entity type"""
    if entity_info["type"] == "human":
        base_score += HUMAN_TYPE_BOOST

        # Extra boost for specific managerial roles
        if any(term in entity_info["text"].lower() for term in ["manager", "director", "supervisor", "officer"]):
            base_score += MANAGERIAL_ROLE_BOOST

        # Even higher boost for very specific finance roles
        if any(term in entity_info["text"].lower() for term in [
            "accounts receivable manager", "accounts payable manager",
            "finance manager", "financial controller", "accounting manager"
        ]):
            base_score += SPECIFIC_FINANCE_ROLE_BOOST

        # Boost for teams
        if "team" in entity_info["text"].lower():
            base_score += TEAM_BOOST

        # Boost for committees
        if "committee" in entity_info["text"].lower():
            base_score += COMMITTEE_BOOST

        # Acronym team boost - recognize patterns like "MCO team"
        if re.search(r'[A-Z]{2,}\s+team', entity_info["text"], re.IGNORECASE):
            base_score += ACRONYM_TEAM_BOOST

    elif entity_info["type"] == "system":
        base_score += SYSTEM_TYPE_BOOST
    elif entity_info["type"] == "non-performer":
        base_score = MINIMUM_CONFIDENCE

    return base_score


def _apply_control_type_consistency(base_score: float, entity_info: Dict, control_type: Optional[str]) -> float:
    """Apply adjustments for control type consistency"""
    if not control_type:
        return base_score

    control_type_lower = control_type.lower()

    if "automated" in control_type_lower and entity_info["type"] == "system":
        base_score += AUTOMATED_CONTROL_BOOST
    elif "automated" in control_type_lower and entity_info["type"] == "human":
        base_score -= AUTOMATED_CONTROL_BOOST
    elif "manual" in control_type_lower and entity_info["type"] == "human":
        base_score += MANUAL_CONTROL_BOOST
    elif "manual" in control_type_lower and entity_info["type"] == "system":
        base_score -= MANUAL_CONTROL_BOOST

    return base_score


def _apply_frequency_consistency(base_score: float, entity_info: Dict, frequency: Optional[str]) -> float:
    """Apply adjustments for frequency consistency"""
    if not frequency:
        return base_score

    frequency_lower = frequency.lower()

    if any(term in frequency_lower for term in ["daily", "continuous"]):
        if entity_info["type"] == "system":
            base_score += CONTINUOUS_SYSTEM_BOOST
        elif "director" in entity_info["text"].lower() or "executive" in entity_info["text"].lower():
            base_score -= EXECUTIVE_FREQUENCY_PENALTY

    return base_score