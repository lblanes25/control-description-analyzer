import spacy
from typing import Dict, List, Any, Optional, Tuple, Union
import re


def enhance_what_detection(text: str, nlp, existing_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Enhanced WHAT detection with improved verb categorization, strength analysis, and context handling

    Args:
        text: The control description text to analyze
        nlp: spaCy NLP model
        existing_keywords: Optional list of action keywords to consider

    Returns:
        Dictionary containing detailed analysis of action elements:
        - actions: List of detected actions with details
        - primary_action: The main action identified in the control
        - score: Overall score for the WHAT element quality
        - verb_strength: Assessment of the strength of control verbs used
        - is_process: Flag indicating if this appears to be a process description rather than a control
        - voice: Dominant voice used (active/passive)
        - suggestions: Suggestions for improvement
    """
    if not text or text.strip() == '':
        return {
            "actions": [],
            "primary_action": None,
            "score": 0,
            "verb_strength": 0,
            "is_process": False,
            "voice": None,
            "suggestions": ["No text provided to analyze"]
        }

    try:
        # Process the text
        doc = nlp(text.lower())

        # Categorize verbs by strength with default scores
        high_strength_verbs = {
            "approve": 1.0, "authorize": 1.0, "reconcile": 1.0, "validate": 1.0,
            "certify": 1.0, "sign-off": 1.0, "verify": 1.0, "confirm": 0.9,
            "test": 0.9, "enforce": 0.9, "ensure": 0.9, "authenticate": 0.9,
            "audit": 0.9, "inspect": 0.9, "investigate": 0.9, "scrutinize": 0.9
        }

        medium_strength_verbs = {
            "review": 0.7, "examine": 0.7, "analyze": 0.7, "compare": 0.7,
            "evaluate": 0.7, "assess": 0.7, "track": 0.7, "document": 0.7,
            "record": 0.7, "maintain": 0.6, "prepare": 0.6, "generate": 0.6,
            "update": 0.6, "calculate": 0.6, "process": 0.6, "recalculate": 0.7
        }

        low_strength_verbs = {
            "check": 0.4, "look": 0.3, "monitor": 0.5, "observe": 0.4,
            "view": 0.3, "consider": 0.3, "watch": 0.3, "note": 0.4,
            "see": 0.2, "handle": 0.3, "manage": 0.4, "coordinate": 0.4,
            "facilitate": 0.3, "oversee": 0.5, "run": 0.4, "perform": 0.5
        }

        # Combine and extend with existing keywords if provided
        all_verbs = {**high_strength_verbs, **medium_strength_verbs, **low_strength_verbs}

        if existing_keywords:
            # Add any missing keywords with a default medium score
            for kw in existing_keywords:
                if kw.lower() not in all_verbs:
                    all_verbs[kw.lower()] = 0.6

        # Action detection logic
        actions = []
        process_indicators = 0
        active_voice_count = 0
        passive_voice_count = 0
        sequence_markers = ["then", "after", "before", "next", "subsequently", "finally", "lastly", "following"]

        # Count sequence markers as process indicators
        for marker in sequence_markers:
            if marker in text.lower():
                process_indicators += 1

        # Detect actions using dependency parsing
        for sent in doc.sents:
            # Find all verbs in the sentence
            for token in sent:
                # Check if token is a verb
                if token.pos_ == "VERB":
                    # Get the lemmatized form of the verb
                    verb_lemma = token.lemma_.lower()

                    # Skip auxiliary verbs and common verbs that don't represent control actions
                    if verb_lemma in ["be", "have", "do", "can", "could", "would", "should", "may", "might"]:
                        continue

                    # Get the verb phrase (verb + objects)
                    verb_phrase = get_verb_phrase(token)

                    # Determine if this is passive voice
                    is_passive = is_passive_construction(token)

                    if is_passive:
                        passive_voice_count += 1
                    else:
                        active_voice_count += 1

                    # Get the subject of the action
                    subject = get_subject(token)

                    # Determine verb strength
                    verb_strength = 0.5  # Default medium strength
                    verb_category = "medium"

                    if verb_lemma in high_strength_verbs:
                        verb_strength = high_strength_verbs[verb_lemma]
                        verb_category = "high"
                    elif verb_lemma in medium_strength_verbs:
                        verb_strength = medium_strength_verbs[verb_lemma]
                        verb_category = "medium"
                    elif verb_lemma in low_strength_verbs:
                        verb_strength = low_strength_verbs[verb_lemma]
                        verb_category = "low"

                    # Calculate confidence score
                    # Factors affecting confidence:
                    # 1. Verb strength
                    # 2. Voice (prefer active)
                    # 3. Subject clarity
                    # 4. Position in the text (earlier = more important)
                    confidence = verb_strength

                    # Adjust for voice
                    if is_passive:
                        confidence *= 0.9

                    # Adjust for subject clarity
                    if subject:
                        confidence *= 1.1

                    # Adjust for position (normalize by text length)
                    position_factor = 1.0 - (token.i / len(doc)) * 0.2
                    confidence *= position_factor

                    # Cap confidence at 1.0
                    confidence = min(1.0, confidence)

                    # Add to actions
                    actions.append({
                        "verb": token.text,
                        "verb_lemma": verb_lemma,
                        "full_phrase": verb_phrase,
                        "subject": subject,
                        "is_passive": is_passive,
                        "strength": verb_strength,
                        "strength_category": verb_category,
                        "confidence": confidence,
                        "span": [token.i, token.i + 1],
                        "sentence": sent.text
                    })

        # Sort actions by confidence
        actions.sort(key=lambda x: x["confidence"], reverse=True)

        # Identify the primary action (highest confidence)
        primary_action = actions[0] if actions else None

        # Calculate verb strength metrics
        avg_verb_strength = sum(a["strength"] for a in actions) / len(actions) if actions else 0

        # Determine if this is describing a process rather than a single control
        # Indicators:
        # 1. More than 3 distinct action verbs
        # 2. Presence of sequence markers
        # 3. Length of text (very long descriptions often describe processes)
        distinct_verbs = set(a["verb_lemma"] for a in actions)
        if len(distinct_verbs) > 3:
            process_indicators += 2

        if len(text.split()) > 50:  # If more than 50 words
            process_indicators += 1

        is_process = process_indicators >= 3

        # Determine dominant voice
        if active_voice_count > passive_voice_count:
            dominant_voice = "active"
        elif passive_voice_count > active_voice_count:
            dominant_voice = "passive"
        else:
            dominant_voice = "mixed" if active_voice_count > 0 else "unknown"

        # Calculate final score (0-1 scale)
        # Components:
        # 1. Quality of primary action (50%)
        # 2. Average verb strength (25%)
        # 3. Voice preference (active preferred) (15%)
        # 4. Process vs. control clarity (10%)
        primary_action_score = primary_action["confidence"] if primary_action else 0
        verb_strength_score = avg_verb_strength

        voice_score = 0.8 if dominant_voice == "active" else 0.6 if dominant_voice == "mixed" else 0.4
        process_score = 0.9 if not is_process else 0.3

        final_score = (
                primary_action_score * 0.5 +
                verb_strength_score * 0.25 +
                voice_score * 0.15 +
                process_score * 0.1
        )

        # Generate suggestions
        suggestions = []

        if primary_action and primary_action["strength_category"] == "low":
            suggestions.append(
                f"Replace weak verb '{primary_action['verb']}' with a stronger control verb like 'verify', 'approve', or 'reconcile'")

        if dominant_voice == "passive":
            suggestions.append("Consider using active voice to clearly indicate who performs the control")

        if is_process:
            suggestions.append(
                "This appears to describe a process rather than a specific control action; consider breaking into separate controls")

        if not primary_action:
            suggestions.append("No clear control action detected; add a specific verb describing what the control does")

        return {
            "actions": actions,
            "primary_action": primary_action,
            "score": final_score,
            "verb_strength": avg_verb_strength,
            "is_process": is_process,
            "voice": dominant_voice,
            "suggestions": suggestions
        }

    except Exception as e:
        print(f"Error in WHAT detection: {str(e)}")
        # Return default empty results on error
        return {
            "actions": [],
            "primary_action": None,
            "score": 0,
            "verb_strength": 0,
            "is_process": False,
            "voice": None,
            "suggestions": [f"Error analyzing text: {str(e)}"]
        }


def get_verb_phrase(verb_token) -> str:
    """
    Extract the complete verb phrase (verb + objects) from a verb token

    Args:
        verb_token: A spaCy token that is a verb

    Returns:
        The complete verb phrase as a string
    """
    # Start with the verb itself
    phrase_tokens = [verb_token]

    # Add direct and indirect objects
    for token in verb_token.children:
        if token.dep_ in ["dobj", "iobj", "pobj", "attr"]:
            # Include the token and its children (to get the complete phrase)
            subtree = list(token.subtree)
            phrase_tokens.extend(subtree)

    # Sort tokens by their position in the original text
    phrase_tokens.sort(key=lambda x: x.i)

    # Combine into a phrase
    return " ".join(token.text for token in phrase_tokens)


def get_subject(verb_token) -> Optional[str]:
    """
    Find the subject of a verb

    Args:
        verb_token: A spaCy token that is a verb

    Returns:
        The subject as a string, or None if no subject is found
    """
    for token in verb_token.children:
        if token.dep_ in ["nsubj", "nsubjpass"]:
            # Return the complete noun phrase, including any modifiers
            return " ".join(t.text for t in token.subtree)

    # If no direct subject found, look for a governing verb's subject (for compound verbs)
    if verb_token.dep_ == "xcomp" and verb_token.head.pos_ == "VERB":
        return get_subject(verb_token.head)

    return None


def is_passive_construction(verb_token) -> bool:
    """
    Determine if a verb is in passive voice

    Args:
        verb_token: A spaCy token that is a verb

    Returns:
        True if the verb is in passive voice, False otherwise
    """
    # Check for passive subject
    if any(token.dep_ == "nsubjpass" for token in verb_token.children):
        return True

    # Check for passive auxiliary (be + past participle)
    if verb_token.tag_ == "VBN":  # Past participle
        # Look for form of "be" as an auxiliary
        for ancestor in verb_token.ancestors:
            if ancestor.lemma_ == "be" and ancestor.pos_ == "AUX":
                return True

    return False


def mark_possible_standalone_controls(description: str, nlp) -> List[Dict[str, Any]]:
    """
    Analyze a potentially multi-control description and mark potential standalone controls

    Args:
        description: The control description text to analyze
        nlp: spaCy NLP model

    Returns:
        List of potential standalone controls with their spans and scores
    """
    doc = nlp(description)
    potential_controls = []

    # Split by common sentence separators
    sentences = list(doc.sents)

    # Analyze each sentence
    for i, sent in enumerate(sentences):
        # Analyze this sentence
        result = enhance_what_detection(sent.text, nlp)

        # If it has a clear action, it might be a separate control
        if result["primary_action"] and result["score"] > 0.5 and not result["is_process"]:
            potential_controls.append({
                "text": sent.text,
                "span": [sent.start, sent.end],
                "score": result["score"],
                "action": result["primary_action"]["full_phrase"] if result["primary_action"] else None
            })

    return potential_controls