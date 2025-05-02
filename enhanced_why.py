from typing import List
import re


def extract_risk_aspects(risk_text):
    """
    Extract different aspects or components from a risk description
    to enable partial matching when a risk is mitigated by multiple controls.

    Args:
        risk_text: The risk description text

    Returns:
        List of risk aspects/components
    """
    aspects = []

    # Split complex risks with multiple components
    if " and " in risk_text:
        parts = risk_text.split(" and ")
        for part in parts:
            if len(part.split()) > 3:  # Only consider substantial phrases
                aspects.append(part.strip())

    # Look for risks with multiple impacts
    impact_markers = ["resulting in", "leading to", "causing", "which may cause", "which could result in"]
    for marker in impact_markers:
        if marker in risk_text.lower():
            parts = risk_text.lower().split(marker)
            if len(parts) > 1:
                # Add the cause
                aspects.append(parts[0].strip())
                # Add the effect/impact
                aspects.append(parts[1].strip())

    # If no structural breakdown, try sentence-based splitting
    if not aspects:
        sentences = [s.strip() for s in re.split(r'[.;]', risk_text) if len(s.strip()) > 10]
        aspects.extend(sentences)

    # If still no aspects or just one, use the entire risk as a single aspect
    if not aspects:
        aspects = [risk_text]

    return aspects


def is_valid_why_match(match_text, full_description, nlp, similarity_threshold=0.9):
    """
    Check if a WHY match is sufficiently distinct from the full description

    Args:
        match_text: The matched WHY text
        full_description: The full control description
        nlp: The spaCy NLP model
        similarity_threshold: Threshold above which matches are considered too similar

    Returns:
        Boolean indicating if this is a valid WHY match
    """
    # Simple length check first
    if len(match_text) / len(full_description) > 0.7:
        return False

    # Skip similarity check for short matches
    if len(match_text.split()) < 5:
        return True

    # More sophisticated similarity check
    match_doc = nlp(match_text)
    description_doc = nlp(full_description)

    if not match_doc.has_vector or not description_doc.has_vector:
        # Fall back to simpler check if vectors aren't available
        return len(match_text) / len(full_description) < 0.5

    if match_doc.similarity(description_doc) > similarity_threshold:
        return False

    return True


def enhance_why_detection(text: str, nlp, risk_description: str = None, existing_keywords: List[str] = None,
                          control_id: str = None):
    """
    Enhanced WHY detection that evaluates alignment with mapped risks and includes specific improvements:
    1. Explicit risk tie-back analysis with support for partial risk coverage (multiple controls per risk)
    2. Vague/generic WHY phrase detection
    3. Success/failure criteria evaluation
    4. Distinction between procedural description and actual risk mitigation
    5. Control type alignment checking

    Args:
        text: The control description text
        nlp: The spaCy NLP model
        risk_description: The text of the mapped risk (if available)
        existing_keywords: Optional list of existing WHY keywords
        control_id: Optional control identifier for reference in feedback

    Returns:
        Dictionary with detection results including risk alignment
    """
    if not text or text.strip() == '':
        return {
            "explicit_why": [],
            "implicit_why": [],
            "top_match": None,
            "why_category": None,
            "score": 0,
            "risk_alignment_score": 0 if risk_description else None,
            "feedback": "No control description provided.",
            "extracted_keywords": [],
            "has_success_criteria": False,
            "vague_why_phrases": [],
            "is_actual_mitigation": False,
            "control_type_mismatch": False
        }

    # Process the text
    doc = nlp(text)

    # Default WHY keywords if none provided
    purpose_markers = existing_keywords or [
        "to ensure", "in order to", "for the purpose of", "designed to",
        "intended to", "so that", "purpose", "objective", "goal",
        "prevent", "detect", "mitigate", "risk", "error", "fraud",
        "misstatement", "compliance", "regulatory", "requirement",
        "accuracy", "completeness", "validity", "integrity"
    ]

    # Pattern-based WHY detection for explicit statements
    explicit_why_candidates = []

    # Look for purpose clauses with "to" + verb pattern - refined to limit match length
    purpose_patterns = [
        r'to\s+(ensure|verify|confirm|validate|prevent|detect|mitigate|comply|adhere|demonstrate|maintain|support|achieve|provide)\s+([^\.;,]{5,50})',
        r'in\s+order\s+to\s+([^\.;,]{5,50})',
        r'for\s+the\s+purpose\s+of\s+([^\.;,]{5,50})',
        r'designed\s+to\s+([^\.;,]{5,50})',
        r'intended\s+to\s+([^\.;,]{5,50})',
        r'so\s+that\s+([^\.;,]{5,50})'
    ]

    for pattern in purpose_patterns:
        matches = list(re.finditer(pattern, text.lower()))
        if not matches:
            print(f"[WHY DEBUG] No matches found for pattern: {pattern}")
        for match in matches:
            purpose_text = match.group(0)
            print(f"[WHY DEBUG] Found pattern match: '{purpose_text}'")
            if is_valid_why_match(purpose_text, text, nlp):
                explicit_why_candidates.append({
                    "text": purpose_text,
                    "method": "pattern_match",
                    "score": 0.9,
                    "span": [match.start(), match.end()],
                    "context": text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
                })

            # Only add if the match is valid (not too similar to full description)
            if is_valid_why_match(purpose_text, text, nlp):
                explicit_why_candidates.append({
                    "text": purpose_text,
                    "method": "pattern_match",
                    "score": 0.9,
                    "span": [match.start(), match.end()],
                    "context": text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
                })

    # Look for keyword-based WHY indicators
    for keyword in purpose_markers:
        if keyword.lower() in text.lower():
            # Find the position in text
            pos = text.lower().find(keyword.lower())

            # Get the sentence containing this keyword
            sentence = None
            for sent in doc.sents:
                if pos >= sent.start_char and pos < sent.end_char:
                    sentence = sent
                    break

            if sentence and is_valid_why_match(sentence.text, text, nlp):
                # Additional check: make sure the keyword is used as purpose indicator
                # not just as a word in another context
                keyword_position = sentence.text.lower().find(keyword.lower())
                if keyword_position == 0 or sentence.text[keyword_position - 1] in " .,;:(":
                    explicit_why_candidates.append({
                        "text": sentence.text,
                        "method": "keyword_match",
                        "score": 0.7,
                        "span": [sentence.start, sentence.end],
                        "context": text[max(0, sentence.start_char - 10):min(len(text), sentence.end_char + 10)]
                    })

    # Detect implicit WHY statements based on control actions
    implicit_why_candidates = []

    # Common control verbs and their implied purposes
    control_verb_purpose = {
        "review": "to ensure accuracy and completeness",
        "reconcile": "to ensure data integrity and accuracy",
        "approve": "to ensure proper authorization",
        "verify": "to confirm accuracy and validity",
        "validate": "to ensure compliance and accuracy",
        "monitor": "to detect anomalies or non-compliance",
        "check": "to identify errors or inconsistencies"
    }

    for token in doc:
        if token.lemma_.lower() in control_verb_purpose:
            # Get the complete verb phrase if possible
            verb_phrase = token.text

            # Try to get the object being acted upon
            obj_text = ""
            for child in token.children:
                if child.dep_ in ("dobj", "pobj"):
                    obj_text = child.text
                    # Try to get the full noun phrase
                    for chunk in doc.noun_chunks:
                        if child.i >= chunk.start and child.i < chunk.end:
                            obj_text = chunk.text
                            break
                    break

            implied_purpose = control_verb_purpose[token.lemma_.lower()]

            implicit_why_candidates.append({
                "text": f"{verb_phrase} {obj_text}",
                "implied_purpose": implied_purpose,
                "method": "action_inference",
                "score": 0.5,  # Lower score for implicit purposes
                "span": [token.i, token.i + len(obj_text.split()) + 1 if obj_text else 1],
                "context": f"Implied purpose from action: {implied_purpose}"
            })

    # Categorize the WHY (if found)
    categories = {
        "risk_mitigation": ["risk", "prevent", "mitigate", "reduce", "avoid", "minimize"],
        "compliance": ["comply", "compliance", "regulatory", "regulation", "requirement", "policy", "standard"],
        "accuracy": ["accuracy", "accurate", "correct", "error-free", "integrity", "reliable"],
        "completeness": ["complete", "completeness", "all", "comprehensive"],
        "authorization": ["authorize", "approval", "permission", "authorization"]
    }

    def categorize_why(why_text):
        scores = {cat: 0 for cat in categories}
        for cat, keywords in categories.items():
            for keyword in keywords:
                if keyword.lower() in why_text.lower():
                    scores[cat] += 1

        # Return the category with highest score, or None if all zeros
        max_cat = max(scores.items(), key=lambda x: x[1])
        return max_cat[0] if max_cat[1] > 0 else None

    # IMPROVEMENT 1: Improved risk tie-back analysis
    risk_specific_terms = set()
    risk_alignment_score = None
    risk_alignment_feedback = None
    risk_categories = set()
    risk_aspect_coverage = 0.0
    risk_aspects = []
    aspect_covered_index = -1

    if risk_description:
        # Extract key terms from risk description
        risk_doc = nlp(risk_description.lower())
        risk_specific_terms = set([token.lemma_ for token in risk_doc
                                   if
                                   not token.is_stop and not token.is_punct and token.pos_ in ('NOUN', 'VERB', 'ADJ')])

        # Identify risk categories
        for cat, keywords in categories.items():
            if any(kw in risk_description.lower() for kw in keywords):
                risk_categories.add(cat)

        # Break down risk into key aspects/components for partial coverage measurement
        # This helps evaluate if a control addresses at least one aspect of a multi-faceted risk
        risk_aspects = extract_risk_aspects(risk_description)

    # IMPROVEMENT 2: Detect vague/generic WHY phrases
    vague_why_terms = [
        "proper functioning", "appropriate", "adequately", "properly",
        "as needed", "as required", "as appropriate", "correct functioning",
        "effective", "efficient", "functioning", "operational", "successful",
        "appropriate action", "necessary action", "properly functioning"
    ]

    vague_why_phrases = []
    for candidate in explicit_why_candidates:
        why_text = candidate["text"].lower()
        for vague_term in vague_why_terms:
            if vague_term in why_text:
                vague_why_phrases.append({
                    "text": why_text,
                    "vague_term": vague_term,
                    "suggested_replacement": "specific risk, impact, or compliance requirement"
                })
                # Penalize score for vague phrases
                candidate["score"] *= 0.7

    # IMPROVEMENT 3: Check for success/failure criteria
    threshold_patterns = [
        r'(\$\d+[,\d]*|\d+\s*%|\d+\s*percent)',
        r'greater than|less than|at least|at most|minimum|maximum',
        r'threshold of|limit of|tolerance of',
        r'within \d+\s*(day|hour|minute|week|month)',
        r'criteria|criterion|standard|benchmark'
    ]

    has_success_criteria = False
    for pattern in threshold_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            has_success_criteria = True
            break

    # IMPROVEMENT 4: Evaluate if control is actually a mitigation vs just a procedure
    # Check for action verbs that culminate in risk mitigation
    mitigation_verbs = [
        "resolve", "correct", "address", "remediate", "fix", "prevent",
        "block", "stop", "deny", "restrict", "escalate", "alert",
        "notify", "disable", "lockout", "report"
    ]

    # Control should end with an action that mitigates, not just identifies
    is_actual_mitigation = False
    for verb in mitigation_verbs:
        if verb.lower() in text.lower():
            is_actual_mitigation = True
            break

    # IMPROVEMENT 5: Check for control type mismatch
    control_type_indicators = {
        "preventive": ["prevent", "block", "restrict", "deny", "before", "prior to"],
        "detective": ["detect", "identify", "discover", "find", "monitor", "review"],
        "corrective": ["correct", "remediate", "fix", "resolve", "address"]
    }

    # Extract implied control type from WHY statements
    implied_control_types = set()
    for candidate in explicit_why_candidates:
        for control_type, indicators in control_type_indicators.items():
            if any(indicator in candidate["text"].lower() for indicator in indicators):
                implied_control_types.add(control_type)

    control_type_mismatch = False
    if len(implied_control_types) > 1:
        control_type_mismatch = True  # Conflicting implied control types

    # Calculate alignment with mapped risk if provided
    if risk_description and (explicit_why_candidates or implicit_why_candidates):
        # Function to calculate semantic similarity
        def calculate_similarity(text1, text2, nlp):
            doc1 = nlp(text1)
            doc2 = nlp(text2)

            # Check if both have vectors
            if not doc1.has_vector or not doc2.has_vector:
                # Fallback to keyword matching
                keywords1 = set([token.lemma_.lower() for token in doc1
                                 if not token.is_stop and not token.is_punct])
                keywords2 = set([token.lemma_.lower() for token in doc2
                                 if not token.is_stop and not token.is_punct])

                # Calculate Jaccard similarity
                if keywords1 and keywords2:
                    return len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
                return 0.0

            return doc1.similarity(doc2)

        # Get best WHY statement
        best_why = None
        if explicit_why_candidates:
            best_why = max(explicit_why_candidates, key=lambda x: x["score"])["text"]
        elif implicit_why_candidates:
            best_why = max(implicit_why_candidates, key=lambda x: x["score"])["implied_purpose"]

        if best_why:
            # Calculate base semantic similarity
            base_similarity = calculate_similarity(best_why, risk_description, nlp)

            # Check for specific risk term matches
            why_doc = nlp(best_why.lower())
            why_terms = set([token.lemma_ for token in why_doc
                             if not token.is_stop and not token.is_punct and token.pos_ in ('NOUN', 'VERB', 'ADJ')])

            # Calculate term overlap with overall risk
            term_overlap = len(risk_specific_terms.intersection(why_terms)) / len(
                risk_specific_terms) if risk_specific_terms else 0

            # Calculate aspect-level alignment for partial risk coverage
            aspect_scores = []
            for aspect in risk_aspects:
                aspect_similarity = calculate_similarity(best_why, aspect, nlp)
                aspect_scores.append(aspect_similarity)

            # Use the best aspect alignment score - this allows a control to focus on just one part of a multi-faceted risk
            best_aspect_score = max(aspect_scores) if aspect_scores else 0
            aspect_covered_index = aspect_scores.index(best_aspect_score) if aspect_scores else -1

            # Calculate risk aspect coverage (what percentage of risk aspects are addressed by this control)
            if aspect_scores:
                # Consider an aspect addressed if similarity is above threshold
                aspects_addressed = sum(1 for score in aspect_scores if score > 0.5)
                risk_aspect_coverage = aspects_addressed / len(aspect_scores)

            # Weighted final score that considers:
            # 1. Overall semantic similarity (30%)
            # 2. Term overlap with full risk (20%)
            # 3. Best alignment with any single risk aspect (50%) - supports partial coverage model
            risk_alignment_score = (0.3 * base_similarity) + (0.2 * term_overlap) + (0.5 * best_aspect_score)

            # Additional penalty for vague phrases in risk context
            if vague_why_phrases:
                risk_alignment_score *= 0.8

            # Generate feedback based on alignment score
            if risk_alignment_score >= 0.7:
                if risk_aspect_coverage >= 0.8:
                    risk_alignment_feedback = "Strong alignment with the full mapped risk."
                else:
                    covered_aspect = risk_aspects[aspect_covered_index] if aspect_covered_index >= 0 else ""
                    risk_alignment_feedback = f"Strong alignment with part of the mapped risk: '{covered_aspect}'"
                    if len(risk_aspects) > 1:
                        risk_alignment_feedback += f", but may not address all aspects of: '{risk_description}'."
            elif risk_alignment_score >= 0.4:
                risk_alignment_feedback = "Moderate alignment with mapped risk."
                if len(risk_aspects) > 1 and aspect_covered_index >= 0:
                    risk_alignment_feedback += f" Primarily addresses: '{risk_aspects[aspect_covered_index]}'."
            else:
                risk_alignment_feedback = f"Weak alignment with mapped risk: '{risk_description}'. Consider explicitly addressing how this control mitigates this specific risk."

                # Add specific missing terms
                if risk_specific_terms:
                    missing_terms = risk_specific_terms - why_terms
                    if missing_terms:
                        top_missing = list(missing_terms)[:3]
                        risk_alignment_feedback += f" Consider including key terms like: {', '.join(top_missing)}."

                # Add reference to control ID if provided
                if control_id:
                    risk_alignment_feedback += f" (Control {control_id})"

    # Sort candidates by score
    explicit_why_candidates.sort(key=lambda x: x["score"], reverse=True)
    implicit_why_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Filter out any matches that are too similar to the original description
    valid_explicit_candidates = []
    for candidate in explicit_why_candidates:
        if is_valid_why_match(candidate["text"], text, nlp):
            valid_explicit_candidates.append(candidate)

    explicit_why_candidates = valid_explicit_candidates

    if explicit_why_candidates:
        print(f"[WHY DEBUG] Top explicit candidate: '{explicit_why_candidates[0]['text']}'")
        print(f"[WHY DEBUG] Score: {explicit_why_candidates[0]['score']}")

    # Determine overall WHY score and top match
    if explicit_why_candidates:
        top_match = explicit_why_candidates[0]
        base_score = top_match["score"]
        why_category = categorize_why(top_match["text"])
    elif implicit_why_candidates:
        top_match = implicit_why_candidates[0]
        base_score = top_match["score"] * 0.7  # Implicit scores are discounted
        why_category = categorize_why(top_match["implied_purpose"])
    else:
        top_match = None
        base_score = 0  # Ensure zero score when nothing found
        why_category = None

    print(f"[WHY DEBUG] Base score: {base_score}")
    print(f"[WHY DEBUG] WHY category: {why_category}")

    return {
        "explicit_why": explicit_why_candidates,
        "implicit_why": implicit_why_candidates,
        "top_match": top_match,
        "why_category": why_category,
        "score": base_score,
        "risk_alignment_score": risk_alignment_score,
        "feedback": risk_alignment_feedback,
        "extracted_keywords": [c["text"] for c in explicit_why_candidates],
        "has_success_criteria": has_success_criteria,
        "vague_why_phrases": vague_why_phrases,
        "is_actual_mitigation": is_actual_mitigation,
        "control_type_mismatch": control_type_mismatch
    }