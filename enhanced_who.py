import re
from typing import List, Optional


def enhanced_who_detection_v2(text: str, nlp, control_type: Optional[str] = None,
                              frequency: Optional[str] = None, existing_keywords: Optional[List[str]] = None):
    """
    Enhanced WHO detection with dependency parsing focus for more robust performer identification.

    This function uses spaCy's dependency parsing to identify performers in control descriptions,
    handling various sentence structures including:
    - Active voice: "The Manager reviews..."
    - Passive voice: "The review is performed by the Manager..."
    - Temporal prefixes: "Annually, the Team reviews..."
    - Responsibility clauses: "The Committee is responsible for..."

    Args:
        text: The control description text
        nlp: The spaCy NLP model
        control_type: Optional control type for context (e.g., "manual", "automated")
        frequency: Optional frequency information for context
        existing_keywords: Optional list of custom keywords from configuration

    Returns:
        Dictionary with detection results including primary and secondary performers
    """
    if not text or text.strip() == '':
        return {
            "primary": None,
            "secondary": [],
            "confidence": 0,
            "message": "No text provided"
        }

    try:
        doc = nlp(text)

        # Initialize candidates list
        who_candidates = []

        # Step 1: Find all control verbs in the text
        # Start with default verbs
        control_verbs = [
            "review", "approve", "verify", "check", "validate", "reconcile",
            "examine", "analyze", "evaluate", "assess", "monitor", "track",
            "investigate", "inspect", "audit", "oversee", "supervise", "ensure",
            "perform", "execute", "conduct", "disable", "enforce", "generate",
            "address", "compare", "maintain", "identify", "correct", "update",
            "submit", "complete", "prepare", "provide", "confirm"
        ]

        # Add custom keywords from config if available
        if existing_keywords:
            for keyword in existing_keywords:
                # Extract verbs from keywords (if they contain multiple words)
                words = keyword.lower().split()
                for word in words:
                    # Check if it's likely a verb
                    if word.endswith(('s', 'ed', 'ing')) or word in control_verbs:
                        if word not in control_verbs:
                            control_verbs.append(word)

        # Find all verbs and their subjects
        subjects_by_verb = {}
        for token in doc:
            # Check if this token is a verb we care about
            if token.lemma_.lower() in control_verbs or token.text.lower() in control_verbs:
                # Look for subjects of this verb
                subjects = []
                for child in token.children:
                    # Look for nominal subjects (active voice) and passive subjects
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        # Get the full noun phrase this token is part of
                        noun_phrase_found = False
                        for chunk in doc.noun_chunks:
                            if child.i >= chunk.start and child.i < chunk.end:
                                subjects.append({
                                    "text": chunk.text,
                                    "span": (chunk.start, chunk.end),
                                    "root": child.text,
                                    "is_passive": child.dep_ == "nsubjpass"
                                })
                                noun_phrase_found = True
                                break

                        # If not found in chunks, just use the token
                        if not noun_phrase_found:
                            subjects.append({
                                "text": child.text,
                                "span": (child.i, child.i + 1),
                                "root": child.text,
                                "is_passive": child.dep_ == "nsubjpass"
                            })

                # Store subjects for this verb
                if subjects:
                    subjects_by_verb[token.i] = {
                        "verb": token.text,
                        "lemma": token.lemma_,
                        "subjects": subjects,
                        "token": token
                    }

        # Step 2: Find performers in prepositional phrases (e.g., "by the team")
        preps_by_verb = {}
        for token in doc:
            if token.lemma_.lower() in control_verbs or token.text.lower() in control_verbs:
                # Look for prepositional phrases
                preps = []
                for child in token.children:
                    if child.dep_ == "prep" and child.text.lower() == "by":
                        # Find the object of the preposition
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                # Get the full noun phrase
                                noun_phrase_found = False
                                for chunk in doc.noun_chunks:
                                    if grandchild.i >= chunk.start and grandchild.i < chunk.end:
                                        preps.append({
                                            "text": chunk.text,
                                            "span": (chunk.start, chunk.end),
                                            "root": grandchild.text
                                        })
                                        noun_phrase_found = True
                                        break

                                # If not found in chunks, just use the token
                                if not noun_phrase_found:
                                    preps.append({
                                        "text": grandchild.text,
                                        "span": (grandchild.i, grandchild.i + 1),
                                        "root": grandchild.text
                                    })

                if preps:
                    preps_by_verb[token.i] = {
                        "verb": token.text,
                        "lemma": token.lemma_,
                        "preps": preps,
                        "token": token
                    }

        # Also check for "by" phrases at the sentence level
        for sent in doc.sents:
            for token in sent:
                if token.text.lower() == "by" and token.dep_ == "prep":
                    for child in token.children:
                        if child.dep_ == "pobj":
                            # This is likely a performer in passive voice
                            # Get the full noun phrase
                            for chunk in doc.noun_chunks:
                                if child.i >= chunk.start and child.i < chunk.end:
                                    # Classify the entity
                                    entity_text = chunk.text
                                    entity_type = classify_entity_type(entity_text, nlp, existing_keywords)

                                    # High confidence since "by X" is a strong indicator
                                    confidence = calculate_who_confidence({
                                        "text": entity_text,
                                        "type": entity_type,
                                        "is_passive": False
                                    }, control_type, frequency) * 1.3  # Apply a 30% boost

                                    candidate = {
                                        "text": entity_text,
                                        "verb": "passive_by_phrase",
                                        "type": entity_type,
                                        "score": min(1.0, confidence),  # Cap at 1.0
                                        "position": chunk.start,
                                        "role": "primary",  # "by X" usually indicates the primary performer
                                        "detection_method": "passive_by_phrase"
                                    }

                                    who_candidates.append(candidate)
                                    break

        # Step 3: Process control verb subjects
        for verb_idx, verb_info in subjects_by_verb.items():
            for subject in verb_info["subjects"]:
                subject_text = subject["text"]

                # Skip common non-performers
                non_performers = ["exception", "error", "transaction", "record", "issue",
                                  "item", "reconciliation", "document", "report",
                                  "questionnaire", "form", "template", "results"]
                if any(term in subject_text.lower() for term in non_performers):
                    continue

                # Classify the entity
                entity_type = classify_entity_type(subject_text, nlp, existing_keywords)

                # Score this candidate
                confidence = calculate_who_confidence({
                    "text": subject_text,
                    "type": entity_type,
                    "is_passive": subject.get("is_passive", False)
                }, control_type, frequency)

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
                            confidence *= 0.7

                # Create candidate
                candidate = {
                    "text": subject_text,
                    "verb": verb_info["verb"],
                    "type": entity_type,
                    "score": confidence,
                    "position": subject["span"][0],
                    "role": role,
                    "detection_method": "subject_of_verb"
                }

                who_candidates.append(candidate)

        # Step 4: Process prepositional phrase performers
        for verb_idx, verb_info in preps_by_verb.items():
            for prep in verb_info["preps"]:
                prep_text = prep["text"]

                # Skip common non-performers
                non_performers = ["exception", "error", "transaction", "record", "issue",
                                  "item", "reconciliation", "document", "report",
                                  "questionnaire", "form", "template", "results"]
                if any(term in prep_text.lower() for term in non_performers):
                    continue

                # Classify the entity
                entity_type = classify_entity_type(prep_text, nlp, existing_keywords)

                # Score this candidate with a boost since "by X" is a strong indicator
                confidence = calculate_who_confidence({
                    "text": prep_text,
                    "type": entity_type,
                    "is_passive": False
                }, control_type, frequency) * 1.2  # Apply a 20% boost

                # Create candidate
                candidate = {
                    "text": prep_text,
                    "verb": verb_info["verb"],
                    "type": entity_type,
                    "score": min(1.0, confidence),  # Cap at 1.0
                    "position": prep["span"][0],
                    "role": "primary",  # "by X" usually indicates the primary performer
                    "detection_method": "by_prepositional_phrase"
                }

                who_candidates.append(candidate)

        # Step 5: Find explicit role declarations (e.g., "X is responsible for...")
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
                                    entity_type = classify_entity_type(entity_text, nlp, existing_keywords)

                                    # This is a very strong indicator of a performer
                                    confidence = calculate_who_confidence({
                                        "text": entity_text,
                                        "type": entity_type,
                                        "is_passive": False
                                    }, control_type, frequency) * 1.3  # Apply a 30% boost

                                    candidate = {
                                        "text": entity_text,
                                        "verb": "responsible/accountable",
                                        "type": entity_type,
                                        "score": min(1.0, confidence),  # Cap at 1.0
                                        "position": chunk.start,
                                        "role": "primary",
                                        "detection_method": "explicit_responsibility"
                                    }

                                    who_candidates.append(candidate)
                                    break

        # Step 6: Handle special cases like temporal prefixes
        # Example: "Annually, the X team reviews..."
        temporal_prefixes = ["annually", "monthly", "quarterly", "weekly", "daily", "periodically", "regularly"]

        # Check if the text starts with a temporal prefix
        text_lower = text.lower()
        has_temporal_prefix = False
        matched_prefix = None

        for prefix in temporal_prefixes:
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
                entity_type = classify_entity_type(team_text, nlp, existing_keywords)

                confidence = calculate_who_confidence({
                    "text": team_text,
                    "type": entity_type,
                    "is_passive": False
                }, control_type, frequency) * 1.2  # Boost for clear structure

                candidate = {
                    "text": team_text,  # Just "the [Team Name]"
                    "verb": "temporal_structure",
                    "type": entity_type,
                    "score": min(1.0, confidence),
                    "position": prefix_end + team_match.start(),
                    "role": "primary",
                    "detection_method": "temporal_prefix_explicit"
                }

                who_candidates.append(candidate)

        # Step 7: If we found no candidates at all, try more aggressive noun chunk analysis
        if not who_candidates:
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text

                # Look for strong performer indicators in the chunk
                if any(indicator in chunk_text.lower() for indicator in
                       ["team", "manager", "director", "committee", "officer", "department", "group", "unit"]):
                    entity_type = classify_entity_type(chunk_text, nlp, existing_keywords)

                    # Lower confidence since we're not sure about the verb relationship
                    confidence = calculate_who_confidence({
                        "text": chunk_text,
                        "type": entity_type,
                        "is_passive": False
                    }, control_type, frequency) * 0.7  # Apply a 30% penalty

                    if confidence > 0.3:  # Only add if reasonable confidence
                        candidate = {
                            "text": chunk_text,
                            "verb": "unknown",
                            "type": entity_type,
                            "score": confidence,
                            "position": chunk.start,
                            "role": "unknown",
                            "detection_method": "noun_chunk_fallback"
                        }

                        who_candidates.append(candidate)

        # Step 8: If we still have no candidates, use regex for team names and acronyms
        if not who_candidates:
            # Use regex to find different types of team names and acronyms
            org_patterns = [
                # Multi-word capitalized names like "Risk Management Team"
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:Team|Group|Department|Unit|Committee|Office))',

                # Acronym team names like "MCO Team"
                r'([A-Z]{2,}\s+(?:Team|Group|Department|Unit|Committee|Office))',

                # Variations with "the" prefix
                r'the\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:Team|Group|Department|Unit|Committee|Office))',
                r'the\s+([A-Z]{2,}\s+(?:Team|Group|Department|Unit|Committee|Office))',

                # Names without team designator like "Finance Department"
                r'([A-Z][a-z]+\s+(?:Management|Administration|Operations|Finance|Audit|Risk|Compliance|Security))',

                # Special case for financial roles
                r'((?:Chief|Senior|Junior)\s+[A-Z][a-z]+\s+[A-Z][a-z]+)'
            ]

            for pattern in org_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    match_text = match.group(1) if match.lastindex == 1 else match.group(0)
                    entity_type = classify_entity_type(match_text, nlp, existing_keywords)

                    # Lower confidence since we're using regex
                    confidence = calculate_who_confidence({
                        "text": match_text,
                        "type": entity_type,
                        "is_passive": False
                    }, control_type, frequency) * 0.6  # Apply a 40% penalty

                    if confidence > 0.3:  # Only add if reasonable confidence
                        candidate = {
                            "text": match_text,
                            "verb": "unknown",
                            "type": entity_type,
                            "score": confidence,
                            "position": match.start(),
                            "role": "unknown",
                            "detection_method": "regex_fallback"
                        }

                        who_candidates.append(candidate)

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
        message = ""

        # Check for consistency with control type
        if control_type and primary["type"] != "unknown":
            control_type_lower = control_type.strip().lower()
            performer_type = primary["type"]

            if control_type_lower == "automated" and performer_type == "human":
                message = "Warning: Human performer detected for automated control"
            elif control_type_lower == "manual" and performer_type == "system":
                message = "Warning: System performer detected for manual control"

        return {
            "primary": primary,
            "secondary": unique_secondary[:3],  # Limit to top 3 secondary performers
            "confidence": primary["score"],
            "message": message,
            "detection_methods": list(set(c["detection_method"] for c in who_candidates))
        }

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


def classify_entity_type(text, nlp, custom_keywords=None):
    """
    Classify an entity as human, system, or non-performer with support for custom keywords.

    Args:
        text: The text to classify
        nlp: The spaCy NLP model
        custom_keywords: Optional list of custom keywords from configuration

    Returns:
        Classification as one of: "human", "system", "non-performer", or "unknown"
    """
    text_lower = text.lower()

    # Base human indicators
    human_indicators = [
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

    # System indicators remain the same
    system_indicators = [
        "system", "application", "software", "platform", "database", "server",
        "program", "script", "job", "batch", "workflow", "algorithm", "automated",
        "automatic", "tool", "module", "interface", "api", "service", "function",
        "process", "scheduled", "timer", "daemon", "bot", "routine", "task"
    ]

    # Non-performer indicators expanded
    non_performer_indicators = [
        "limit", "threshold", "policy", "procedure", "standard", "regulation",
        "account", "transaction", "balance", "report", "document", "record",
        "exception", "error", "issue", "finding", "discrepancy", "review",
        "control", "entry", "activity", "access", "instance", "item",
        "questionnaire", "form", "template", "results", "assessment"
    ]

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

    # Check if the entity contains human role indicators
    if any(indicator in text_lower for indicator in human_indicators):
        return "human"

    # Check if the entity contains system indicators
    if any(indicator in text_lower for indicator in system_indicators):
        return "system"

    # Check if the entity contains non-performer indicators
    if any(indicator in text_lower for indicator in non_performer_indicators):
        return "non-performer"

    # Use NLP to check for person or organization entities
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG"):
            return "human"

    # Additional check for acronyms that might be team names (like "MCO")
    if re.match(r'\b[A-Z]{2,}\b', text) and len(text) <= 5:
        # Short uppercase sequence - likely an acronym for a team or department
        return "human"

    # Default to unknown if no clear classification
    return "unknown"


def calculate_who_confidence(entity, control_type=None, frequency=None):
    """
    Calculate confidence score for a WHO entity with improved scoring.

    Args:
        entity: Entity information dictionary
        control_type: Optional control type for context
        frequency: Optional frequency information for context

    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Increased base score
    base_score = 0.7  # Start with higher base score

    # Adjust for entity type
    if entity["type"] == "human":
        base_score += 0.2

        # Extra boost for specific managerial roles
        if any(term in entity["text"].lower() for term in ["manager", "director", "supervisor", "officer"]):
            base_score += 0.2  # Extra boost for specific roles

        # Even higher boost for very specific finance roles
        if any(term in entity["text"].lower() for term in [
            "accounts receivable manager", "accounts payable manager",
            "finance manager", "financial controller", "accounting manager"
        ]):
            base_score += 0.3  # Additional boost for specific finance roles

        # Boost for teams
        if "team" in entity["text"].lower():
            base_score += 0.1

        # Boost for committees
        if "committee" in entity["text"].lower():
            base_score += 0.15

        # Acronym team boost - recognize patterns like "MCO team"
        if re.search(r'[A-Z]{2,}\s+team', entity["text"], re.IGNORECASE):
            base_score += 0.2

    elif entity["type"] == "system":
        base_score += 0.1
    elif entity["type"] == "non-performer":
        base_score = 0.1  # Very low confidence for non-performers

    # Reduced penalty for passive voice
    if entity.get("is_passive", False):
        base_score -= 0.1  # Reduced penalty for passive voice

    # Adjust for control type consistency
    if control_type:
        control_type_lower = control_type.lower()

        if "automated" in control_type_lower and entity["type"] == "system":
            base_score += 0.2
        elif "automated" in control_type_lower and entity["type"] == "human":
            base_score -= 0.2
        elif "manual" in control_type_lower and entity["type"] == "human":
            base_score += 0.2
        elif "manual" in control_type_lower and entity["type"] == "system":
            base_score -= 0.2

    # Adjust for frequency consistency
    if frequency:
        frequency_lower = frequency.lower()

        if any(term in frequency_lower for term in ["daily", "continuous"]):
            if entity["type"] == "system":
                base_score += 0.1
            elif "director" in entity["text"].lower() or "executive" in entity["text"].lower():
                base_score -= 0.1

    # Ensure score is within valid range
    return max(0.1, min(1.0, base_score))


def identify_control_action_subjects(doc):
    """
    Identify the subjects of control action verbs in the text.

    This function is used by the old WHO detection approach but is kept
    for backward compatibility.

    Args:
        doc: spaCy document

    Returns:
        List of subject dictionaries
    """
    control_verbs = [
        "review", "approve", "verify", "check", "validate", "reconcile",
        "examine", "analyze", "evaluate", "assess", "monitor", "track",
        "investigate", "inspect", "audit", "oversee", "supervise", "ensure",
        "perform", "execute", "conduct", "disable", "enforce", "generate",
        "address", "compare", "maintain", "identify", "correct", "update",
        "submit", "complete", "prepare", "provide", "confirm"
    ]

    subjects = []

    for token in doc:
        if token.lemma_.lower() in control_verbs:
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    for chunk in doc.noun_chunks:
                        if child.i >= chunk.start and child.i < chunk.end:
                            subjects.append({
                                "text": chunk.text,
                                "verb": token.text,
                                "verb_lemma": token.lemma_,
                                "is_passive": child.dep_ == "nsubjpass",
                                "start": chunk.start,
                                "end": chunk.end
                            })
                            break

    return subjects


def identify_performer_roles(subjects, text):
    """
    Categorize performers as primary, secondary, or escalation.

    This function is used by the old WHO detection approach but is kept
    for backward compatibility.

    Args:
        subjects: List of subject dictionaries
        text: The full text

    Returns:
        Dictionary of categorized roles
    """
    roles = {
        "primary": [],
        "secondary": [],
        "escalation": []
    }

    # Look for escalation patterns
    escalation_patterns = [
        r'escalated to (?:the\s+)?([a-zA-Z\s]+(?:manager|director|supervisor|analyst|officer))',
        r'reported to (?:the\s+)?([a-zA-Z\s]+(?:manager|director|supervisor|analyst|officer))',
        r'elevated to (?:the\s+)?([a-zA-Z\s]+(?:manager|director|supervisor|analyst|officer))',
        r'notified to (?:the\s+)?([a-zA-Z\s]+(?:manager|director|supervisor|analyst|officer))',
        r'alert(?:s|ed)? (?:the\s+)?([a-zA-Z\s]+(?:manager|director|supervisor|analyst|officer))'
    ]

    escalation_performers = []
    for pattern in escalation_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            escalation_performers.append(match.group(1).strip())

    # Look for preparation patterns
    preparation_patterns = [
        r'prepared by (?:the\s+)?([a-zA-Z\s]+)',
        r'completed by (?:the\s+)?([a-zA-Z\s]+)',
        r'generated by (?:the\s+)?([a-zA-Z\s]+)',
        r'performed by (?:the\s+)?([a-zA-Z\s]+)',
        r'created by (?:the\s+)?([a-zA-Z\s]+)'
    ]

    preparation_performers = []
    for pattern in preparation_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            preparation_performers.append(match.group(1).strip())

    # Categorize subjects
    for subject in subjects:
        subject_text = subject["text"].strip().lower()

        # Check if this is an escalation performer
        if any(performer.lower() in subject_text for performer in escalation_performers):
            roles["escalation"].append(subject)
        # Check if this is a preparation performer
        elif any(performer.lower() in subject_text for performer in preparation_performers):
            roles["secondary"].append(subject)
        # If associated with strong control verbs, likely primary
        elif subject["verb_lemma"].lower() in ["review", "approve", "verify", "validate", "reconcile", "examine",
                                               "analyze"]:
            roles["primary"].append(subject)
        else:
            # Default to secondary if role is unclear
            roles["secondary"].append(subject)

    return roles

def detect_control_structure(text):
    """
    Detect common control description structures and identify primary performers.

    This function uses regex patterns to identify common control description structures.
    It's enhanced to better handle acronym-based team names like "MCO team".

    Args:
        text: The control description text

    Returns:
        List of primary performer dictionaries
    """
    # Original patterns for standard role names
    primary_performer_patterns = [
        r'(?:the\s+)?([a-zA-Z\s]+(?:manager|director|supervisor|analyst|officer|administrator|controller|accountant|auditor|team))\s+(reviews|approves|verifies|reconciles|validates|examines|analyzes|monitors)',
        r'(?:the\s+)?([a-zA-Z\s]+(?:manager|director|supervisor|analyst|officer|administrator|controller|accountant|auditor|team))\s+is\s+responsible\s+for',
        r'(?:the\s+)?([a-zA-Z\s]+(?:manager|director|supervisor|analyst|officer|administrator|controller|accountant|auditor|team))\s+ensures'
    ]

    # New patterns specifically for acronym-based team names
    acronym_team_patterns = [
        # Match patterns like "MCO team", "IT team", "HR department"
        r'(?:the\s+)?([A-Z]{2,}\s+(?:team|group|department|unit|committee|office))\s+(review(?:s|ed)?|approve(?:s|d)?|verify|verifie(?:s|d)?|reconcile(?:s|d)?|validate(?:s|d)?|examine(?:s|d)?|analyze(?:s|d)?|monitor(?:s|ed)?|submit(?:s|ted)?|ensure(?:s|d)?|perform(?:s|ed)?|conduct(?:s|ed)?|execute(?:s|d)?)',

        # Match with "is responsible for" pattern
        r'(?:the\s+)?([A-Z]{2,}\s+(?:team|group|department|unit|committee|office))\s+is\s+responsible\s+for',

        # Match with "ensures" pattern
        r'(?:the\s+)?([A-Z]{2,}\s+(?:team|group|department|unit|committee|office))\s+ensures'
    ]

    # Patterns for mixed-case acronym teams (like "AML Team" or "IT Support Team")
    mixed_case_team_patterns = [
        # Match patterns like "AML Team", "IT Support Team"
        r'(?:the\s+)?([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\s+(?:team|group|department|unit|committee|office))\s+(review(?:s|ed)?|approve(?:s|d)?|verify|verifie(?:s|d)?|reconcile(?:s|d)?|validate(?:s|d)?|examine(?:s|d)?|analyze(?:s|d)?|monitor(?:s|ed)?|submit(?:s|ted)?|ensure(?:s|d)?|perform(?:s|ed)?|conduct(?:s|ed)?|execute(?:s|d)?)',

        # Match with "is responsible for" pattern
        r'(?:the\s+)?([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\s+(?:team|group|department|unit|committee|office))\s+is\s+responsible\s+for',

        # Match with "ensures" pattern
        r'(?:the\s+)?([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\s+(?:team|group|department|unit|committee|office))\s+ensures'
    ]

    # Patterns specifically for sentences that begin with temporal modifiers
    temporal_prefix_patterns = [
        # Handle cases like "Annually, the MCO team reviews..."
        # The important part is that we're capturing just group 1 (the team) in the primary_performers list
        r'(?:annually|monthly|quarterly|weekly|daily|periodically|regularly),?\s+(?:the\s+)?([A-Z]{2,}\s+(?:team|group|department|unit|committee|office))\s+(review(?:s|ed)?|approve(?:s|d)?|submit(?:s|ted)?|ensure(?:s|d)?|perform(?:s|ed)?)',

        # Similar pattern but for full team names
        r'(?:annually|monthly|quarterly|weekly|daily|periodically|regularly),?\s+(?:the\s+)?([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\s+(?:team|group|department|unit|committee|office))\s+(review(?:s|ed)?|approve(?:s|d)?|submit(?:s|ted)?|ensure(?:s|d)?|perform(?:s|ed)?)',

        # Same pattern but for regular team names
        r'(?:annually|monthly|quarterly|weekly|daily|periodically|regularly),?\s+(?:the\s+)?([a-zA-Z\s]+(?:manager|director|supervisor|analyst|officer|administrator|controller|accountant|auditor|team))\s+(review(?:s|ed)?|approve(?:s|d)?|submit(?:s|ted)?|ensure(?:s|d)?|perform(?:s|ed)?)'
    ]

    # Combine all patterns
    all_patterns = primary_performer_patterns + acronym_team_patterns + mixed_case_team_patterns + temporal_prefix_patterns

    primary_performers = []

    # Use the combined pattern list
    for pattern in all_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            primary_performers.append({
                "text": match.group(1).strip(),
                "verb": match.group(2) if len(match.groups()) > 1 else "responsible",
                "position": match.start()
            })

    return primary_performers

def calculate_role_adjusted_confidence(entity, role_type):
    """
    Apply role-specific adjustments to confidence scores.

    Args:
        entity: Entity information dictionary
        role_type: Role classification (primary, secondary, escalation)

    Returns:
        Adjusted confidence score
    """
    if role_type == "primary":
        # Boost primary performers
        return min(1.0, entity["score"] * 1.2)
    elif role_type == "secondary":
        # Secondary performers (preparers, generators) should have less impact
        return entity["score"] * 0.5
    elif role_type == "escalation":
        # Escalation performers should have minimal impact on WHO scoring
        return entity["score"] * 0.3
    else:
        return entity["score"]

def is_valid_performer(text):
    """
    Check if a text segment is likely to be a valid performer.

    Args:
        text: Text to check

    Returns:
        Boolean indicating if the text is likely a valid performer
    """
    # Filter out common false positives
    false_positives = ["changes", "results", "distributions", "communications",
                       "requirements", "transactions", "documents"]

    if any(fp.lower() == text.lower() for fp in false_positives):
        return False

    # Should have a role or organizational indicator
    role_indicators = ["manager", "director", "team", "staff", "officer", "group",
                       "department", "committee", "analyst", "specialist"]
    has_role = any(indicator in text.lower() for indicator in role_indicators)

    # Check for acronym team patterns like "MCO team"
    has_acronym_team = bool(re.search(r'[A-Z]{2,}\s+(?:team|group)', text, re.IGNORECASE))

    # Should not be just a verb or action
    action_only = re.match(r'^\w+ed$', text.strip()) is not None

    return (has_role or has_acronym_team) and not action_only

def is_valid_performer_in_context(entity, full_text, nlp):
    """
    Check if an entity is likely to be a control performer based on surrounding context.

    Args:
        entity: Entity text to check
        full_text: The full control description text
        nlp: The spaCy NLP model

    Returns:
        Boolean indicating if the entity is likely a performer in context
    """
    # Look for verb patterns around the entity
    # Find the position of the entity in the text
    entity_pos = full_text.find(entity)
    if entity_pos == -1:
        return False

    # Get text window around entity
    start_pos = max(0, entity_pos - 100)
    end_pos = min(len(full_text), entity_pos + len(entity) + 100)
    context_window = full_text[start_pos:end_pos]

    # Check for control verb patterns
    control_patterns = [
        r'(review|approve|verify|monitor|check|validate|reconcile)\s+by',
        r'(reviewed|approved|verified|monitored|checked|validated|reconciled)\s+by',
        r'performed\s+by',
        r'conducted\s+by',
        r'executed\s+by',
        r'is\s+responsible\s+for'
    ]

    for pattern in control_patterns:
        if re.search(pattern, context_window, re.IGNORECASE):
            return True

    # Check if the entity appears to be the subject of a control verb
    doc = nlp(context_window)
    for token in doc:
        if token.dep_ == "nsubj" and token.head.lemma_ in ["review", "approve", "verify", "monitor"]:
            for chunk in doc.noun_chunks:
                if token.i >= chunk.start and token.i < chunk.end:
                    chunk_text = chunk.text.lower()
                    if entity.lower() in chunk_text:
                        return True

    return False

def detect_performer_coreferences(candidates, text):
    """
    Detect when different terms refer to the same performer and boost the score.

    Args:
        candidates: List of performer candidates
        text: The control description text

    Returns:
        Updated list of performer candidates with coreference information
    """
    if len(candidates) < 2:
        return candidates

    # Special case for "The manager validates" pattern
    if len(candidates) > 0:
        # Look for "the manager validates" pattern which strongly indicates a primary performer
        manager_pattern = re.search(r'the\s+manager\s+(validates|ensures|confirms|reviews)', text, re.IGNORECASE)
        if manager_pattern:
            # Find primary candidate with "manager" in title
            for candidate in candidates:
                if "manager" in candidate["text"].lower():
                    candidate["role"] = "primary"  # Ensure it's marked as primary
                    candidate["score"] = min(1.0, candidate["score"] * 1.5)  # Strong boost
                    break

    # Check for common patterns like "Manager" followed by "The manager"
    for i, candidate in enumerate(candidates):
        if i == 0:
            continue

        # Get the last word of the first performer (usually the title)
        primary_performer = candidates[0]["text"].lower()
        primary_title = primary_performer.split()[-1]
        current_text = candidate["text"].lower()

        # Check if this is a reference like "the manager" to the first performer
        if current_text == "the " + primary_title or current_text == primary_title:
            # This is likely a reference to the first performer
            candidates[0]["score"] = min(1.0, candidates[0]["score"] + 0.3)  # Increased boost
            # Mark this candidate as a coreference
            candidate["is_coreference"] = True
            candidate["refers_to"] = primary_performer

    # Also check for explicit references in the text
    for candidate in candidates:
        if "manager" in candidate["text"].lower():
            match = re.search(r'the\s+manager\s+validates|the\s+manager\s+confirms|the\s+manager\s+ensures',
                              text, re.IGNORECASE)
            if match:
                candidate["score"] = min(1.0, candidate["score"] + 0.2)

    return candidates