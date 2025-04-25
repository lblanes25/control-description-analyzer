import spacy
from typing import Dict, List, Any, Optional, Tuple, Union
import re


def enhance_what_detection(text: str, nlp, existing_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Enhanced WHAT detection with improved verb categorization, strength analysis, and context handling
    With fixes to properly distinguish between actions (WHAT) and purposes (WHY)

    Args:
        text: The control description text to analyze
        nlp: spaCy NLP model
        existing_keywords: Optional list of action keywords to consider

    Returns:
        Dictionary containing detailed analysis of action elements:
        - actions: List of detected actions with details
        - primary_action: The main action identified in the control
        - secondary_actions: Additional significant actions identified
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
            "secondary_actions": [],
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
            "test": 0.9, "enforce": 0.9, "authenticate": 0.9,
            "audit": 0.9, "inspect": 0.9, "investigate": 0.9, "scrutinize": 0.9,
            "compare": 0.9  # Moved from medium to high strength
        }

        medium_strength_verbs = {
            "review": 0.7, "examine": 0.7, "analyze": 0.7,
            "evaluate": 0.7, "assess": 0.7, "track": 0.7, "document": 0.7,
            "record": 0.7, "maintain": 0.6, "prepare": 0.6, "generate": 0.6,
            "update": 0.6, "calculate": 0.6, "process": 0.6, "recalculate": 0.7
        }

        # Significantly lowered strength values for weak verbs
        low_strength_verbs = {
            "check": 0.3, "look": 0.2, "monitor": 0.4, "observe": 0.3,
            "view": 0.2, "consider": 0.2, "watch": 0.2, "note": 0.3,
            "see": 0.1, "handle": 0.2, "manage": 0.3, "coordinate": 0.3,
            "facilitate": 0.2, "oversee": 0.4, "run": 0.3, "perform": 0.3,  # Reduced "perform" from 0.5 to 0.3
            "address": 0.2  # Added "address" with very low score
        }

        # Combine and extend with existing keywords if provided
        all_verbs = {**high_strength_verbs, **medium_strength_verbs, **low_strength_verbs}

        if existing_keywords:
            # Add any missing keywords with a default medium score
            for kw in existing_keywords:
                if kw.lower() not in all_verbs:
                    all_verbs[kw.lower()] = 0.6

        # ----------------
        # NEW: Purpose phrase detection to identify WHY segments and exclude from WHAT
        # ----------------
        purpose_phrases = [
            r'to\s+(ensure|verify|confirm|validate|prevent|detect|mitigate|comply|adhere|demonstrate|maintain|support|achieve|provide)',
            r'in\s+order\s+to\s+[^\.;,]*',
            r'for\s+the\s+purpose\s+of\s+[^\.;,]*',
            r'designed\s+to\s+[^\.;,]*',
            r'intended\s+to\s+[^\.;,]*',
            r'so\s+that\s+[^\.;,]*'
        ]

        # Find all purpose phrases to exclude from WHAT candidates
        purpose_spans = []
        for pattern in purpose_phrases:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                purpose_spans.append((match.start(), match.end()))

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
                    # Skip if this verb is part of a purpose phrase (WHY not WHAT)
                    if any(token.idx >= start and token.idx < end for start, end in purpose_spans):
                        continue

                    # Get the lemmatized form of the verb
                    verb_lemma = token.lemma_.lower()

                    # Skip auxiliary verbs and common verbs that don't represent control actions
                    if verb_lemma in ["be", "have", "do", "can", "could", "would", "should", "may", "might"]:
                        continue

                    # Skip "ensure" when used as a standalone verb, as it's typically WHY not WHAT
                    if verb_lemma == "ensure" and not is_action_ensure(token):
                        continue

                    # Get the verb phrase (verb + objects) with improved extraction
                    verb_phrase = get_verb_phrase_improved(token)

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
                    # 5. Object specificity (new)
                    # 6. Vague term penalty (new)
                    confidence = verb_strength

                    # Adjust for voice
                    if is_passive:
                        confidence *= 0.8  # Increased penalty for passive voice (was 0.9)

                    # Adjust for subject clarity
                    if subject:
                        confidence *= 1.1

                    # Adjust for position (normalize by text length)
                    position_factor = 1.0 - (token.i / len(doc)) * 0.2
                    confidence *= position_factor

                    # NEW: Adjust for object specificity
                    object_specificity = assess_object_specificity(token)
                    confidence *= (1.0 + object_specificity * 0.2)

                    # NEW: Apply vague term penalty
                    # Check sentence for vague terms like "as needed", "when appropriate"
                    vague_terms = ["as needed", "when necessary", "as appropriate",
                                   "when appropriate", "as required", "if needed"]
                    sent_text = sent.text.lower()

                    if any(term in sent_text for term in vague_terms):
                        confidence *= 0.7  # Strong penalty for vague temporal terms

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
                        "sentence": sent.text,
                        "is_core_action": is_core_control_action(token, verb_lemma)
                        # NEW: Flag for core control actions
                    })

        # Sort actions by confidence
        actions.sort(key=lambda x: x["confidence"], reverse=True)

        # NEW: Filter for core control actions to identify primary and secondary
        core_actions = [a for a in actions if a.get("is_core_action", False)]

        # If no core actions found, fall back to all actions
        if not core_actions and actions:
            core_actions = actions

        # Identify primary and secondary actions
        primary_action = core_actions[0] if core_actions else None
        secondary_actions = core_actions[1:3] if len(core_actions) > 1 else []

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
            # NEW: Provide specific alternative suggestions based on context
            specific_alts = get_specific_alternatives(primary_action["verb_lemma"])
            suggestions.append(
                f"Replace weak verb '{primary_action['verb']}' with a stronger control verb like {specific_alts}")

        if dominant_voice == "passive":
            suggestions.append("Consider using active voice to clearly indicate who performs the control")

        if is_process:
            suggestions.append(
                "This appears to describe a process rather than a specific control action; consider breaking into separate controls")

        if not primary_action:
            suggestions.append("No clear control action detected; add a specific verb describing what the control does")

        # NEW: Suggestion for vague objects
        if primary_action and assess_object_specificity(doc[primary_action["span"][0]]) < 0.5:
            suggestions.append(f"Consider clarifying the object of '{primary_action['verb_lemma']}' to be more specific.")

        return {
            "actions": actions,
            "primary_action": primary_action,
            "secondary_actions": secondary_actions,  # NEW: Return secondary actions
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
            "secondary_actions": [],
            "score": 0,
            "verb_strength": 0,
            "is_process": False,
            "voice": None,
            "suggestions": [f"Error analyzing text: {str(e)}"]
        }


def get_verb_phrase_improved(verb_token) -> str:
    """
    Extract the complete verb phrase (verb + objects) from a verb token
    Improved to handle more complex verb phrases and collect more context

    Args:
        verb_token: A spaCy token that is a verb

    Returns:
        The complete verb phrase as a string
    """
    # Start with the verb itself
    phrase_tokens = [verb_token]

    # Add direct and indirect objects, plus prepositional phrases that modify the verb
    for token in verb_token.children:
        if token.dep_ in ["dobj", "iobj", "pobj", "attr", "oprd"]:
            # Include the token and its children (to get the complete phrase)
            subtree = list(token.subtree)
            phrase_tokens.extend(subtree)
        elif token.dep_ == "prep":
            # Include prepositional phrases
            subtree = list(token.subtree)
            phrase_tokens.extend(subtree)
        elif token.dep_ == "advcl" and token.pos_ != "VERB":
            # Include adverbial clauses that aren't separate verbs
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


def is_action_ensure(verb_token) -> bool:
    """
    Determine if "ensure" is being used as a primary action rather than as a purpose indicator

    Args:
        verb_token: A spaCy token for the "ensure" verb

    Returns:
        True if it appears to be a primary action, False if it's a purpose indicator
    """
    # If "ensure" is the main verb of the sentence, it might be a primary action
    if verb_token.dep_ == "ROOT":
        # Check if preceded by "to" which would indicate purpose
        prev_token = verb_token.doc[verb_token.i - 1] if verb_token.i > 0 else None
        if prev_token and prev_token.text.lower() == "to":
            return False
        return True

    # If it's not a ROOT verb, check if it's part of a purpose clause
    if verb_token.dep_ == "xcomp" or verb_token.dep_ == "advcl":
        # Check if preceded by "to"
        prev_token = verb_token.doc[verb_token.i - 1] if verb_token.i > 0 else None
        if prev_token and prev_token.text.lower() == "to":
            return False

    # Default to treating it as an action
    return True


def assess_object_specificity(verb_token) -> float:
    """
    Assess how specific the object of a verb is

    Args:
        verb_token: A spaCy token that is a verb

    Returns:
        Float score from 0.0 (vague) to 1.0 (very specific)
    """
    # Find direct objects
    objects = []
    for token in verb_token.children:
        if token.dep_ in ["dobj", "pobj", "attr"]:
            # Get the complete subtree
            objects.extend(list(token.subtree))

    if not objects:
        return 0.0

    # Assess specificity based on length, modifiers, and noun types
    score = min(1.0, len(objects) / 5.0)  # Longer object phrases tend to be more specific

    # Check for modifiers that make objects more specific
    specificity_indicators = ["specific", "certain", "particular", "defined", "exact"]
    if any(token.lemma_.lower() in specificity_indicators for token in objects):
        score += 0.2

    # Check for numeric modifiers (usually make things more specific)
    if any(token.pos_ == "NUM" for token in objects):
        score += 0.2

    # NEW: Check for vague generic terms that reduce specificity
    vague_object_terms = ["item", "thing", "stuff", "issue", "matter", "situation", "exception"]
    if any(token.lemma_.lower() in vague_object_terms for token in objects):
        score -= 0.3  # Significant penalty for vague objects

    return min(1.0, max(0.0, score))  # Ensure score is between 0 and 1


def is_core_control_action(verb_token, verb_lemma) -> bool:
    """
    Determine if a verb represents a core control action rather than a supporting process action

    Args:
        verb_token: A spaCy token that is a verb
        verb_lemma: The lemmatized form of the verb

    Returns:
        True if it appears to be a core control action, False otherwise
    """
    # Core control actions are typically the main verbs that directly mitigate risk
    core_control_verbs = {
        "review", "verify", "approve", "validate", "reconcile", "check",
        "confirm", "compare", "examine", "investigate", "audit", "analyze",
        "evaluate", "assess", "test", "monitor", "inspect"
    }

    # If it's a known core control verb, it's likely a core action
    if verb_lemma in core_control_verbs:
        return True

    # If it's the root verb of the sentence, it's more likely to be a core action
    if verb_token.dep_ == "ROOT":
        return True

    # Supporting actions are often in subordinate clauses
    if verb_token.dep_ in ["advcl", "relcl", "ccomp"]:
        return False

    # Default to False for uncertain cases
    return False


def get_specific_alternatives(verb_lemma) -> str:
    """
    Provide specific alternative verb suggestions based on the context

    Args:
        verb_lemma: The lemmatized form of the verb to replace

    Returns:
        String with 2-3 specific suggested replacements
    """
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
        "monitor": "'track', 'supervise', or 'observe'"
    }

    return alternatives.get(verb_lemma, "'verify', 'approve', or 'reconcile'")


def mark_possible_standalone_controls(description: str, nlp) -> List[Dict[str, Any]]:
    """
    Analyze a potentially multi-control description and mark potential standalone controls
    Improved to identify controls that should be separated

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
                "action": result["primary_action"]["full_phrase"] if result["primary_action"] else None,
                "control_type": determine_likely_control_type(sent.text)  # NEW: Determine likely control type
            })

    # NEW: Also look for controls that might be in the same sentence
    # Split long sentences with multiple verbs and conjunctions
    if len(sentences) == 1 and len(doc) > 20:  # If only one long sentence
        compound_control_candidates = identify_compound_controls(doc, nlp)
        if compound_control_candidates:
            potential_controls.extend(compound_control_candidates)

    return potential_controls


def determine_likely_control_type(text: str) -> str:
    """
    Determine the likely control type based on text analysis

    Args:
        text: Control description text

    Returns:
        Likely control type: "preventive", "detective", "corrective", or "unknown"
    """
    text_lower = text.lower()

    # Preventive control indicators
    if any(term in text_lower for term in ["prevent", "block", "stop", "prohibit", "restrict", "limit"]):
        return "preventive"

    # Detective control indicators
    if any(term in text_lower for term in ["detect", "identify", "discover", "find", "monitor", "review", "reconcile"]):
        return "detective"

    # Corrective control indicators
    if any(term in text_lower for term in ["correct", "resolve", "address", "fix", "remediate", "rectify"]):
        return "corrective"

    return "unknown"


def identify_compound_controls(doc, nlp) -> List[Dict[str, Any]]:
    """
    Identify potential compound controls within a single sentence

    Args:
        doc: spaCy document
        nlp: spaCy NLP model

    Returns:
        List of potential standalone controls
    """
    compound_controls = []

    # Look for clauses connected by coordinating conjunctions
    for token in doc:
        if token.pos_ == "CCONJ" and token.text.lower() in ["and", "or"]:
            # Check if connecting two verb phrases
            left_verbs = [t for t in doc[:token.i] if t.pos_ == "VERB"]
            right_verbs = [t for t in doc[token.i + 1:] if t.pos_ == "VERB"]

            if left_verbs and right_verbs:
                # Extract left and right clauses
                left_verb = left_verbs[-1]  # Last verb before conjunction
                right_verb = right_verbs[0]  # First verb after conjunction

                # Get rough clause spans
                left_span_end = token.i
                left_span_start = max(0, left_verb.i - 5)  # Approximate subject+verb start
                right_span_start = token.i + 1
                right_span_end = min(len(doc), right_verb.i + 5)  # Approximate verb+object end

                # Create potential control entries
                left_text = doc[left_span_start:left_span_end].text
                right_text = doc[right_span_start:right_span_end].text

                # Only add if they have sufficient length
                if len(left_text.split()) > 3:
                    compound_controls.append({
                        "text": left_text,
                        "span": [left_span_start, left_span_end],
                        "score": 0.6,  # Lower confidence due to extraction method
                        "action": left_verb.text,
                        "control_type": determine_likely_control_type(left_text)
                    })

                if len(right_text.split()) > 3:
                    compound_controls.append({
                        "text": right_text,
                        "span": [right_span_start, right_span_end],
                        "score": 0.6,  # Lower confidence due to extraction method
                        "action": right_verb.text,
                        "control_type": determine_likely_control_type(right_text)
                    })

    return compound_controls